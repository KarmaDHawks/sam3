import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sam3.model_builder import build_sam3_video_model
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
import csv
import h5py


# CONFIG
TRAIN_NOISY_DIR = '/media/TBData/marco/Projects/VOST/features/ft_ICIP/SAM3_features/train/noisy'
TRAIN_CLEAN_DIR = '/media/TBData/marco/Projects/VOST/features/ft_ICIP/SAM3_features/train/clean'
VAL_NOISY_DIR = "/media/TBData/marco/Projects/VOST/features/ft_ICIP/SAM3_features/val/noisy"
VAL_CLEAN_DIR = "/media/TBData/marco/Projects/VOST/features/ft_ICIP/SAM3_features/val/clean"
SAVE_DIR = '/media/TBData/marco/Projects/VOST/FeatureCorrector_Training/ECCV/SAM3TC/CNN/3x3/MSE/IoU/delta_5'
mask_save_dir = '/media/TBData/marco/Projects/VOST/FeatureCorrector_Training/ECCV/SAM3TC/CNN/3x3/MSE/IoU/delta_5/masks'
DEVICE = 'cuda:0'
BATCH_SIZE = 8
EPOCHS = 10
NUM_WORKERS = 4

GT_MASK_PATH_TRAIN = '/media/TBData/marco/Projects/VOST/SingleObject/Train_binary/Annotations'
GT_MASK_PATH_VAL = '/media/TBData/marco/Projects/VOST/SingleObject/Val_binary/Annotations'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(mask_save_dir, exist_ok=True)


# TC kernel 3x3 - ResNet style
class FeatureCorrector(nn.Module):
    def __init__(self, C=256, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, C, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class FeaturePairDataset(Dataset):
    def __init__(self, file_list, noisy_base_dir):
        self.file_list = file_list
        self.noisy_base_dir = noisy_base_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        paths = self.file_list[idx]
        clean_path = paths['clean'].replace('.pt', '.h5')
        noisy_path = paths['noisy'].replace('.pt', '.h5')

        # Carica le feature clean e noisy dal file HDF5
        with h5py.File(clean_path, "r") as f:
            f_clean = torch.from_numpy(f["pix_feat"][:]).float()  # [1, 256, H, W]
        with h5py.File(noisy_path, "r") as f:
            f_noisy = torch.from_numpy(f["pix_feat"][:]).float()

        shape = f_clean.shape
        video_name = os.path.basename(os.path.dirname(clean_path))
        frame_idx = int(os.path.splitext(os.path.basename(clean_path))[0])

        high_res_features = None
        try:
            input_feat_path = os.path.join(
                self.noisy_base_dir,
                video_name, 'inputs', f"{frame_idx:05d}.h5"
            )
            if os.path.exists(input_feat_path):
                with h5py.File(input_feat_path, "r") as f:
                    high_res_features = []
                    i = 0
                    while f"high_res_features_{i}" in f:
                        arr = torch.from_numpy(f[f"high_res_features_{i}"][:]).float()
                        high_res_features.append(arr)
                        i += 1
                    if len(high_res_features) == 0:
                        high_res_features = None
        except Exception as e:
            print(f"Warning: Could not load high_res_features from {input_feat_path}: {e}")
            high_res_features = None

        meta = {
            'original_shape': shape,
            'video_name': video_name,
            'frame_idx': frame_idx,
            'high_res_features': high_res_features,
        }
        return f_noisy, f_clean, meta


def collect_file_pairs(noisy_dir, clean_dir, max_videos=None):
    file_list = []
    for video_id in sorted(os.listdir(clean_dir))[:max_videos]:
        clean_video_dir = os.path.join(clean_dir, video_id)
        noisy_video_dir = os.path.join(noisy_dir, video_id)
        if not os.path.isdir(clean_video_dir): 
            continue

        for fname in os.listdir(clean_video_dir):
            if fname.endswith(".h5") and not fname.startswith("00000"):
                clean_path = os.path.join(clean_video_dir, fname)
                noisy_path = os.path.join(noisy_video_dir, fname)
                if os.path.exists(noisy_path):
                    file_list.append({'clean': clean_path, 'noisy': noisy_path})
    return file_list


# Load SAM3 model - come in vos_inference.py
print("Loading SAM3 model...")
sam3_model = build_sam3_video_model(device=DEVICE)
sam3_tracker = sam3_model.tracker
# Congela i parametri di SAM3 così i gradienti fluiscono solo verso il correttore
sam3_tracker.eval()
for p in sam3_tracker.parameters():
    p.requires_grad = False

print("SAM3 model loaded successfully")

# Dataset
train_file_list = collect_file_pairs(TRAIN_NOISY_DIR, TRAIN_CLEAN_DIR)
val_file_list = collect_file_pairs(VAL_NOISY_DIR, VAL_CLEAN_DIR)

train_dataset = FeaturePairDataset(train_file_list, TRAIN_NOISY_DIR)
val_dataset = FeaturePairDataset(val_file_list, VAL_NOISY_DIR)

from torch.utils.data._utils.collate import default_collate


def collate_with_meta(batch):
    xs, ys, metas = zip(*batch)
    return default_collate(xs), default_collate(ys), list(metas)


train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_with_meta
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_with_meta
)

# Model
model = FeatureCorrector()
model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        target = target.float()
        intersection = (pred * target).sum()
        union = (pred + target - pred * target).sum()
        return 1 - (intersection + self.eps) / (union + self.eps)


def compute_iou(pred_mask, gt_mask):
    """Compute IoU between prediction and ground truth."""
    pred = (pred_mask > 0).float()
    gt = (gt_mask > 0).float()
    intersection = (pred * gt).sum()
    union = ((pred + gt) > 0).float().sum()
    if union == 0:
        return torch.tensor(1.0 if intersection == 0 else 0.0, device=pred_mask.device)
    return intersection / union


def save_mask_as_png(mask_tensor, save_path):
    """Save mask tensor as PNG image."""
    mask = (mask_tensor > 0).to(torch.uint8)
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    img = to_pil_image(mask.cpu() * 255)
    img.save(save_path)


mask_loss_fn = IoULoss()
lambda_mask = 5.0

checkpoint_path = os.path.join(SAVE_DIR, "last_checkpoint.pt")
start_epoch = 0
best_val_iou = 0.0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["mlp_corrector"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    best_val_iou = checkpoint["best_val_iou"]
    print(f"✔️ Ripristinato training da checkpoint (epoch {start_epoch}, best_val_iou={best_val_iou:.4f})")

# Training loop
best_val_loss = float('inf')
print("Inizio training...")

metrics_csv_path = os.path.join(SAVE_DIR, "training_metrics.csv")
with open(metrics_csv_path, mode='a' if start_epoch > 0 else 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if start_epoch == 0:
        writer.writerow([
            "epoch", "train_loss", "train_mse", "train_iou_loss", "val_loss", "val_mIoU"
        ])
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_mask_loss = 0.0
        
        for x, y, meta in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)  # x, y: [B, 1, 256, H, W]
            x = x.squeeze(1)  # [B, 256, H, W]
            y = y.squeeze(1)
            
            optimizer.zero_grad()
            output = model(x)  # [B, 256, H, W]
            mse_loss = criterion(output, y)
            total_mask_loss = 0.0

            for i in range(x.size(0)):
                shape = meta[i]['original_shape']  # [1, 256, H, W]
                C, H, W = shape[1], shape[2], shape[3]
                video_name = meta[i]['video_name']
                frame_idx = meta[i]['frame_idx']
                high_res_features = meta[i]['high_res_features']

                pix_feat = output[i].unsqueeze(0).to(DEVICE)  # [1, 256, H, W]

                if isinstance(high_res_features, list):
                    high_res = [f.to(DEVICE) for f in high_res_features]
                else:
                    high_res = None

                # Predizione mask usando SAM3
                # SAM3TrackerBase._forward_sam_heads richiede:
                # - backbone_features: [B, C, H, W] ✓ (abbiamo pix_feat)
                # - point_inputs: None (non abbiamo punti)
                # - mask_inputs: None (non abbiamo maschere di input)
                # - high_res_features: lista di feature opzionali ✓ (abbiamo high_res)
                # - multimask_output: True/False
                with torch.no_grad():
                    sam_outputs = sam3_tracker._forward_sam_heads(
                        backbone_features=pix_feat,
                        point_inputs=None,
                        mask_inputs=None,
                        high_res_features=high_res,
                        multimask_output=True
                    )
                
                # sam_outputs è una tupla: 
                # (low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, object_score_logits)
                # Indice 4 = high_res_masks [B, 1, H, W]
                high_res_masks = sam_outputs[4]  # [1, 1, H_orig, W_orig]
                mask_pred = high_res_masks[0, 0]  # [H_orig, W_orig]

                # Carica GT mask
                gt_mask_path = os.path.join(
                    GT_MASK_PATH_TRAIN,
                    video_name,
                    f"{frame_idx:05d}.png"
                )
                
                if os.path.exists(gt_mask_path):
                    gt_mask = torch.from_numpy(np.array(Image.open(gt_mask_path))).to(mask_pred.device)
                    if gt_mask.max() > 1:
                        gt_mask = (gt_mask > 0).float()

                    # Resize pred -> GT shape
                    mask_pred_resized = F.interpolate(
                        mask_pred.unsqueeze(0).unsqueeze(0),
                        size=gt_mask.shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)

                    # IoU Loss
                    mask_loss = mask_loss_fn(mask_pred_resized, gt_mask)
                    total_mask_loss += mask_loss
                else:
                    print(f"Warning: GT mask not found at {gt_mask_path}")

            # Media delle loss
            total_mask_loss = total_mask_loss / max(x.size(0), 1)

            # Loss combinata
            loss = mse_loss + lambda_mask * total_mask_loss

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            train_mse_loss += mse_loss.item() * x.size(0)
            train_mask_loss += total_mask_loss.item() * x.size(0)

        # Medie sulle metriche di training
        train_loss /= len(train_loader.dataset)
        train_mse_loss /= len(train_loader.dataset)
        train_mask_loss /= len(train_loader.dataset)

        # Validazione
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_count = 0

        epoch_mask_dir = os.path.join(mask_save_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_mask_dir, exist_ok=True)

        with torch.no_grad():
            for x, y, meta in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = x.squeeze(1)
                y = y.squeeze(1)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item() * x.size(0)

                for i in range(x.size(0)):
                    shape = meta[i]['original_shape']
                    C, H, W = shape[1], shape[2], shape[3]
                    video_name = meta[i]['video_name']
                    frame_idx = meta[i]['frame_idx']
                    high_res_features = meta[i]['high_res_features']

                    pix_feat = output[i].unsqueeze(0)  # [1, 256, H, W]
                    if isinstance(high_res_features, list):
                        high_res = [f.to(DEVICE) for f in high_res_features]
                    else:
                        high_res = None

                    # SAM3 inference
                    sam_outputs = sam3_tracker._forward_sam_heads(
                        backbone_features=pix_feat.to(DEVICE),
                        point_inputs=None,
                        mask_inputs=None,
                        high_res_features=high_res,
                        multimask_output=True
                    )
                    high_res_masks = sam_outputs[4]  # [B, 1, H, W]
                    mask = high_res_masks[0, 0]  # [H, W]

                    # Salva PNG
                    out_dir = os.path.join(epoch_mask_dir, video_name)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"{frame_idx:05d}.png")
                    save_mask_as_png(mask, out_path)

                    # Calcolo IoU con GT
                    gt_mask_path = os.path.join(GT_MASK_PATH_VAL, video_name, f"{frame_idx:05d}.png")
                    
                    if os.path.exists(gt_mask_path):
                        gt_mask = torch.from_numpy(np.array(Image.open(gt_mask_path))).to(mask.device)
                        if gt_mask.max() > 1:
                            gt_mask = (gt_mask > 0).float()

                        # Ridimensiona pred -> GT shape
                        mask_resized = F.interpolate(
                            mask.unsqueeze(0).unsqueeze(0),
                            size=gt_mask.shape,
                            mode='bilinear',
                            align_corners=False
                        )
                        mask_resized = torch.sigmoid(mask_resized)
                        mask_resized = (mask_resized > 0.5).float()
                        mask_resized = mask_resized.squeeze()

                        iou = compute_iou(mask_resized, gt_mask)
                        val_iou += iou.item()
                        val_count += 1

        val_loss /= len(val_loader.dataset)
        mean_iou = val_iou / max(val_count, 1)
        
        print(f"[Epoch {epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Train MSE: {train_mse_loss:.6f} | "
            f"Train IoU Loss: {train_mask_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val mIoU: {mean_iou:.4f}")

        # Salva metriche su CSV
        writer.writerow([
            epoch + 1,
            train_loss,
            train_mse_loss,
            train_mask_loss,
            val_loss,
            mean_iou
        ])
        csvfile.flush()

        # Salva best checkpoint se IoU migliora
        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            best_checkpoint_path = os.path.join(SAVE_DIR, "best_checkpoint.pt")
            torch.save({
                "mlp_corrector": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_iou": best_val_iou,
            }, best_checkpoint_path)
            print(f"✔️ Salvato best checkpoint a epoch {epoch+1} con val_mIoU {mean_iou:.4f}")

        # Salva sempre checkpoint dell'ultima epoca
        last_checkpoint_path = os.path.join(SAVE_DIR, "last_checkpoint.pt")
        torch.save({
            "mlp_corrector": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_val_iou": best_val_iou,
        }, last_checkpoint_path)

# Final Save
torch.save({
    "mlp_corrector": model.state_dict(),
}, os.path.join(SAVE_DIR, "mlp_corrector_final.pt"))

print("Training completato!")