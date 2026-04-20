import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import json
from datetime import datetime


class ETHZShapeClassesDataset(Dataset):
    def __init__(self, data_dir, categories, image_paths, labels, transform=None):
        self.data_dir = data_dir
        self.categories = categories
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


def load_ethz_dataset(data_dir):
    categories = []
    all_data = []

    for category_name in sorted(os.listdir(data_dir)):
        category_path = os.path.join(data_dir, category_name)
        if not os.path.isdir(category_path):
            continue

        categories.append(category_name)
        category_id = len(categories) - 1

        for filename in os.listdir(category_path):
            if filename.endswith('.jpg') and not filename.endswith('_edges.tif'):
                img_path = os.path.join(category_path, filename)

                mask_file = filename.rsplit('.', 1)[0] + '.mask.0.png'
                mask_path = os.path.join(category_path, mask_file)

                if os.path.exists(mask_path):
                    all_data.append({
                        'image_path': img_path,
                        'mask_path': mask_path,
                        'category': category_name,
                        'label': category_id
                    })

    return categories, all_data


def preprocess_data(all_data, test_size=0.2, random_state=42):
    image_paths = [d['image_path'] for d in all_data]
    labels = [d['label'] for d in all_data]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    return train_paths, val_paths, train_labels, val_labels


class DataPreprocessor:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_transforms(self):
        return self.train_transform, self.val_transform


class ShiftedWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x = self._window_partition(x, self.window_size)
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        x = self._window_reverse(x, self.window_size, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        return x

    def _window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, window_size * window_size, C)
        return x

    def _window_reverse(self, x, window_size, H, W, C):
        B = x.shape[0]
        h_num = H // window_size
        w_num = W // window_size
        x = x.view(B, h_num, w_num, window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, C)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ShiftedWindowAttention(dim, num_heads, window_size, shift_size)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)
        x = self.norm(x)
        x = x.view(B, H, W, C)
        return x


class SwinTransformerStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim, num_heads, window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SwinTransformerEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        self.stages = nn.ModuleList()
        for i, (depth, num_head) in enumerate(zip(depths, num_heads)):
            stage_dim = embed_dim * (2 ** i)
            self.stages.append(SwinTransformerStage(stage_dim, depth, num_head, mlp_ratio=mlp_ratio))

        self.downsample = nn.ModuleList([
            nn.Conv2d(embed_dim * (2 ** i), embed_dim * (2 ** (i + 1)), kernel_size=2, stride=2)
            for i in range(len(depths) - 1)
        ])

    def forward(self, x):
        x = self.patch_embed(x)
        features = []

        for i, stage in enumerate(self.stages):
            x = stage(x)
            B, H, W, C = x.shape
            features.append(x.view(B, H * W, C))

            if i < len(self.downsample):
                x = x.permute(0, 3, 1, 2).contiguous()
                x = self.downsample[i](x)
                H, W = H // 2, W // 2
                x = x.permute(0, 2, 3, 1).contiguous()

        return features


class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, embed_dim=96, num_classes=5):
        super().__init__()
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(embed_dim * 4, embed_dim * 8, kernel_size=3, padding=1)

        self.attention_conv = nn.Conv2d(embed_dim * 14, embed_dim * 14, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_dim * 14)
        self.relu = nn.ReLU(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(embed_dim * 14, num_classes)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))

        x_multi = torch.cat([x, x1, x2, x3], dim=1)

        attn = self.attention_conv(x_multi)
        attn = self.bn(attn)
        attn = self.relu(attn)

        x_fused = x_multi * attn

        x_pool = self.global_pool(x_fused)
        x_flat = x_pool.view(x_pool.size(0), -1)

        output = self.fc(x_flat)

        return output, x_fused


class CrossModalAttention(nn.Module):
    def __init__(self, visual_dim, text_dim, embed_dim):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim

        self.w_v_q = nn.Linear(visual_dim, embed_dim)
        self.w_t_k = nn.Linear(text_dim, embed_dim)
        self.w_t_v = nn.Linear(text_dim, embed_dim)

        self.scale = embed_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)

        self.fc = nn.Linear(embed_dim, visual_dim)

    def forward(self, visual_features, text_features):
        B, C_v = visual_features.shape

        q_v = self.w_v_q(visual_features)
        k_t = self.w_t_k(text_features)
        v_t = self.w_t_v(text_features)

        attn = (q_v @ k_t.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)

        fused = attn @ v_t
        output = self.fc(fused)

        return output


class IntegratedDesignRecognitionModel(nn.Module):
    def __init__(self, num_classes=2, img_size=224, embed_dim=96):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim

        self.swin_encoder = SwinTransformerEncoder(
            img_size=img_size,
            patch_size=4,
            embed_dim=embed_dim,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )

        self.fusion = MultiScaleFeatureFusion(embed_dim=embed_dim * 8, num_classes=num_classes)

        self.text_dim = 512
        self.text_embedding = nn.Linear(self.text_dim, embed_dim * 8)
        self.cross_attn = CrossModalAttention(embed_dim * 8, embed_dim * 8, embed_dim * 4)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 8 * 2, embed_dim * 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim * 8, num_classes)
        )

    def forward(self, x, text_features=None):
        swin_features = self.swin_encoder(x)

        visual_feat = swin_features[-1]
        B, H, W, C = visual_feat.shape
        visual_2d = visual_feat.permute(0, 2, 1).contiguous().view(B, C, H, W)

        fusion_output, fused_features = self.fusion(visual_2d)

        visual_global = fused_features.mean(dim=(2, 3))

        if text_features is not None:
            text_emb = self.text_embedding(text_features)
            cross_out = self.cross_attn(visual_global, text_emb)
            combined = torch.cat([visual_global, cross_out], dim=1)
        else:
            combined = torch.cat([visual_global, visual_global], dim=1)

        output = self.classifier(combined)

        return output


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


class ModelBuilder:
    @staticmethod
    def build_model(num_classes, img_size=224, embed_dim=96):
        model = IntegratedDesignRecognitionModel(
            num_classes=num_classes,
            img_size=img_size,
            embed_dim=embed_dim
        )
        return model


def dice_loss(pred, target, smooth=1.0):
    pred = torch.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()

    return dice_loss


def combined_loss(pred, target, lambda_seg=0.5):
    ce_loss = F.cross_entropy(pred, target)
    seg_loss = dice_loss(pred, target)
    return ce_loss + lambda_seg * seg_loss


def calculate_metrics(predictions, targets, num_classes):
    predictions = torch.argmax(predictions, dim=1)
    accuracy = (predictions == targets).float().mean().item()

    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)
    for c in range(num_classes):
        mask = (targets == c)
        per_class_total[c] = mask.sum().item()
        if per_class_total[c] > 0:
            per_class_correct[c] = ((predictions == c) & mask).sum().item()

    per_class_acc = torch.where(per_class_total > 0, per_class_correct / per_class_total, torch.zeros_like(per_class_total))

    return accuracy, per_class_acc.tolist()


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.criterion = combined_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            acc, _ = calculate_metrics(outputs, labels, self.model.num_classes)
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")

        self.scheduler.step()

        return total_loss / num_batches, total_acc / num_batches

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                acc, _ = calculate_metrics(outputs, labels, self.model.num_classes)
                total_loss += loss.item()
                total_acc += acc
                num_batches += 1

                all_predictions.append(outputs.cpu())
                all_labels.append(labels.cpu())

        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return total_loss / num_batches, total_acc / num_batches, all_predictions, all_labels

    def train(self, num_epochs, results_file):
        best_val_acc = 0.0
        training_history = []

        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ETHZShapeClasses Deep Learning Model Training Results\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, val_preds, val_labels = self.validate()

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"New best model saved! Val Acc: {best_val_acc:.4f}")

            history_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            }
            training_history.append(history_entry)

            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(f"\nEpoch {epoch + 1}/{num_epochs}\n")
                f.write(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}\n")
                f.write(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}\n")
                f.write(f"  Best Val Accuracy: {best_val_acc:.4f}\n")

        return training_history, best_val_acc


def calculate_f1_scores(predictions, labels, num_classes):
    predictions = torch.argmax(predictions, dim=1)

    f1_scores = []
    for c in range(num_classes):
        tp = ((predictions == c) & (labels == c)).sum().item()
        fp = ((predictions == c) & (labels != c)).sum().item()
        fn = ((predictions != c) & (labels == c)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_scores.append({
            'class': c,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

    macro_f1 = sum(f['f1_score'] for f in f1_scores) / num_classes
    micro_f1 = sum(f['tp'] for f in [
        {'tp': ((predictions == c) & (labels == c)).sum().item()} for c in range(num_classes)
    ]) / len(labels)

    return f1_scores, macro_f1, micro_f1


def model_validation(model, val_loader, device, categories, results_file):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            all_predictions.append(outputs.cpu())
            all_labels.append(labels)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    accuracy, per_class_acc = calculate_metrics(all_predictions, all_labels, len(categories))
    f1_scores, macro_f1, micro_f1 = calculate_f1_scores(all_predictions, all_labels, len(categories))

    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("MODEL VALIDATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")

        f.write("Per-Class Performance:\n")
        f.write("-" * 60 + "\n")
        for i, (cat, acc, f1) in enumerate(zip(categories, per_class_acc, f1_scores)):
            f.write(f"  Class {i} ({cat}):\n")
            f.write(f"    Accuracy: {acc:.4f}\n")
            f.write(f"    Precision: {f1['precision']:.4f}\n")
            f.write(f"    Recall: {f1['recall']:.4f}\n")
            f.write(f"    F1 Score: {f1['f1_score']:.4f}\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
        f.write(f"Micro F1 Score: {micro_f1:.4f}\n")

    print("\nValidation Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")

    return accuracy, f1_scores, macro_f1


def output_results_to_file(results_file, categories, training_history, final_metrics):
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("FINAL SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("Dataset: ETHZShapeClasses\n")
        f.write(f"Number of Classes: {len(categories)}\n")
        f.write(f"Classes: {', '.join(categories)}\n\n")

        f.write("Training Configuration:\n")
        f.write("-" * 40 + "\n")
        f.write("  Model: Swin Transformer + Multi-Scale Feature Fusion\n")
        f.write("  Image Size: 224x224\n")
        f.write("  Embed Dimension: 96\n")
        f.write("  Optimizer: AdamW\n")
        f.write("  Learning Rate: 0.0001\n")
        f.write("  Weight Decay: 0.01\n")
        f.write("  Scheduler: CosineAnnealing\n")
        f.write(f"  Epochs: {len(training_history)}\n\n")

        f.write("Final Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Best Validation Accuracy: {max(h['val_acc'] for h in training_history):.4f}\n")
        f.write(f"  Final Validation Accuracy: {training_history[-1]['val_acc']:.4f}\n")
        f.write(f"  Final Macro F1 Score: {final_metrics['macro_f1']:.4f}\n\n")

        f.write("Training History Summary:\n")
        f.write("-" * 40 + "\n")
        for h in training_history:
            f.write(f"  Epoch {h['epoch']}: Train Acc={h['train_acc']:.4f}, Val Acc={h['val_acc']:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Training completed at: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("=" * 80 + "\n")


def main():
    print("=" * 60)
    print("ETHZShapeClasses Deep Learning Model")
    print("Based on: Multi-Feature Design Elements Recognition Paper")
    print("=" * 60)

    data_dir = "c:/work_place/CCC/data/ETHZShapeClasses"
    results_file = "c:/work_place/CCC/training_results.txt"
    img_size = 224
    batch_size = 8
    num_epochs = 30
    learning_rate = 1e-4
    embed_dim = 96

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    print("\n[1] Loading Dataset...")
    categories, all_data = load_ethz_dataset(data_dir)
    print(f"Found {len(all_data)} samples across {len(categories)} categories")
    for cat, count in zip(categories, [sum(1 for d in all_data if d['category'] == c) for c in categories]):
        print(f"  - {cat}: {count} samples")

    print("\n[2] Preprocessing Data...")
    train_paths, val_paths, train_labels, val_labels = preprocess_data(all_data, test_size=0.2)
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    preprocessor = DataPreprocessor(img_size=img_size)
    train_transform, val_transform = preprocessor.get_transforms()

    train_dataset = ETHZShapeClassesDataset(data_dir, categories, train_paths, train_labels, train_transform)
    val_dataset = ETHZShapeClassesDataset(data_dir, categories, val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("\n[3] Building Model...")
    model = ModelBuilder.build_model(num_classes=len(categories), img_size=img_size, embed_dim=embed_dim)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n[4] Training Model...")
    trainer = ModelTrainer(model, train_loader, val_loader, device, learning_rate=learning_rate)
    training_history, best_val_acc = trainer.train(num_epochs, results_file)

    print("\n[5] Validating Model...")
    model.load_state_dict(torch.load('best_model.pth'))
    final_accuracy, f1_scores, macro_f1 = model_validation(model, val_loader, device, categories, results_file)

    print("\n[6] Saving Final Results...")
    final_metrics = {
        'best_val_acc': best_val_acc,
        'final_accuracy': final_accuracy,
        'macro_f1': macro_f1
    }
    output_results_to_file(results_file, categories, training_history, final_metrics)

    print(f"\nTraining complete! Results saved to: {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()