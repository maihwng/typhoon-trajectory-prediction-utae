import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset, random_split


# ======================================================
# CONFIG
# ======================================================
CSV_PATH = Path("ibtracs.WP.list.v04r01.csv")
ERA5_DIR = Path("ERA5_Data")
CHECKPOINT_PATH = Path("utae_era5_ibtracs.pt")

LON_MIN, LON_MAX = 100.0, 125.0
LAT_MIN, LAT_MAX = 0.0, 25.0
YEARS = range(2021, 2025)
MONTH_MIN, MONTH_MAX = 6, 12

IMG_SIZE = 64
T_INPUT = 5
HORIZON = 6
HEATMAP_SIGMA = 2.0

SURFACE_VARS = ["msl", "sst", "u10", "v10", "tcwv"]
PRESSURE_VARS = ["r", "vo", "u", "v", "z"]
PRESSURE_LEVELS = [850.0, 500.0, 200.0]
INPUT_CHANNELS = len(SURFACE_VARS) + len(PRESSURE_VARS) * len(PRESSURE_LEVELS)


# ======================================================
# GEO / HEATMAP UTILS
# ======================================================
def coord_to_pixel(lat, lon, size=IMG_SIZE):
    x = (lon - LON_MIN) / (LON_MAX - LON_MIN) * (size - 1)
    y = (LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * (size - 1)
    return x, y


def pixel_to_coord(x, y, size=IMG_SIZE):
    lon = LON_MIN + x / (size - 1) * (LON_MAX - LON_MIN)
    lat = LAT_MAX - y / (size - 1) * (LAT_MAX - LAT_MIN)
    return lat, lon


def gaussian_heatmap(lat, lon, size=IMG_SIZE, sigma=HEATMAP_SIGMA):
    cx, cy = coord_to_pixel(lat, lon, size)
    yy, xx = np.mgrid[0:size, 0:size]
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


def heatmap_to_center(heatmap):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return pixel_to_coord(float(x), float(y), heatmap.shape[-1])


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


# ======================================================
# DATA LOADING
# ======================================================
def load_ibtracs_tracks(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    for col in ["LAT", "LON", "USA_WIND"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        df["ISO_TIME"].dt.year.isin(YEARS)
        & df["ISO_TIME"].dt.month.between(MONTH_MIN, MONTH_MAX)
        & df["ISO_TIME"].dt.hour.isin([0, 6, 12, 18])
        & df["LAT"].between(LAT_MIN, LAT_MAX)
        & df["LON"].between(LON_MIN, LON_MAX)
        & (df["NATURE"] != "ET")
    ].copy()

    df = df.dropna(subset=["SID", "ISO_TIME", "LAT", "LON"])
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    return df


def open_era5(era5_dir=ERA5_DIR):
    surface_files = sorted(era5_dir.glob("surface_*.nc"))
    pressure_files = sorted(era5_dir.glob("pressure_*.nc"))
    if not surface_files or not pressure_files:
        raise FileNotFoundError("Can not find ERA5 surface_*.nc and pressure_*.nc files.")

    surface_ds = xr.open_mfdataset(surface_files, combine="by_coords")
    pressure_ds = xr.open_mfdataset(pressure_files, combine="by_coords")
    return surface_ds, pressure_ds


def make_track_samples(df, t_input=T_INPUT, horizon=HORIZON):
    samples = []
    needed = t_input + horizon
    for sid, group in df.groupby("SID"):
        group = group.sort_values("ISO_TIME").reset_index(drop=True)
        for start in range(len(group) - needed + 1):
            window = group.iloc[start : start + needed]
            dt_hours = window["ISO_TIME"].diff().dropna().dt.total_seconds() / 3600
            if not (dt_hours == 6).all():
                continue
            samples.append(
                {
                    "sid": sid,
                    "name": str(window["NAME"].iloc[0]),
                    "input_times": window["ISO_TIME"].iloc[:t_input].tolist(),
                    "target_times": window["ISO_TIME"].iloc[t_input:].tolist(),
                    "target_lats": window["LAT"].iloc[t_input:].to_numpy(np.float32),
                    "target_lons": window["LON"].iloc[t_input:].to_numpy(np.float32),
                }
            )
    return samples


class ERA5IBTrACSDataset(Dataset):
    def __init__(self, samples, surface_ds, pressure_ds):
        self.samples = samples
        self.surface_ds = surface_ds
        self.pressure_ds = pressure_ds

    def __len__(self):
        return len(self.samples)

    def _standardize(self, arr):
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)
        std = arr.std()
        if std < 1e-6:
            return arr * 0.0
        return (arr - arr.mean()) / std

    def _resize_field(self, field):
        tensor = torch.from_numpy(field.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return tensor.squeeze().numpy()

    def _era5_frame(self, time):
        time = np.datetime64(pd.Timestamp(time))
        surf = self.surface_ds.sel(valid_time=time, method="nearest")
        pres = self.pressure_ds.sel(valid_time=time, method="nearest")

        channels = []
        for var in SURFACE_VARS:
            field = surf[var].values
            channels.append(self._standardize(self._resize_field(field)))

        for level in PRESSURE_LEVELS:
            level_frame = pres.sel(pressure_level=level, method="nearest")
            for var in PRESSURE_VARS:
                field = level_frame[var].values
                channels.append(self._standardize(self._resize_field(field)))

        return np.stack(channels).astype(np.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = np.stack([self._era5_frame(t) for t in sample["input_times"]])
        y = np.stack(
            [
                gaussian_heatmap(lat, lon)
                for lat, lon in zip(sample["target_lats"], sample["target_lons"])
            ]
        )
        centers = np.stack([sample["target_lats"], sample["target_lons"]], axis=1).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(centers), idx


# ======================================================
# MODEL: ERA5 + IBTRACS -> UTAE/LTAE -> HEATMAPS
# ======================================================
class LTAE2d(nn.Module):
    def __init__(self, channels=64, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B,T,C,H,W)
        b, t, c, h, w = x.shape
        seq = x.permute(0, 3, 4, 1, 2).contiguous().view(b * h * w, t, c)
        out, _ = self.attn(seq, seq, seq)
        out = self.norm(out + seq)
        out = out.mean(dim=1)
        return out.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class UTAE(nn.Module):
    def __init__(self, in_channels=INPUT_CHANNELS, hidden=64, horizon=HORIZON):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.ltae = LTAE2d(hidden)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, horizon, 1),
        )

    def forward(self, x):
        # x: (B,T,C,H,W), output logits: (B,HORIZON,H,W)
        feats = [self.encoder(x[:, t]) for t in range(x.shape[1])]
        feats = torch.stack(feats, dim=1)
        temporal = self.ltae(feats)
        return self.decoder(temporal)


class HeatmapLoss(nn.Module):
    def __init__(self, pos_weight=20.0, dice_weight=0.5):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        pred = torch.sigmoid(logits)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()
        return bce + self.dice_weight * dice


# ======================================================
# CENTER EXTRACTION / TRAJECTORY / EVALUATION
# ======================================================
def extract_centers_from_logits(logits):
    probs = torch.sigmoid(logits)
    batch_centers = []
    for sample in probs:
        centers = [heatmap_to_center(step) for step in sample]
        batch_centers.append(centers)
    return batch_centers


def trajectory_km_errors(pred_centers, true_centers):
    errors = []
    for (pred_lat, pred_lon), (true_lat, true_lon) in zip(pred_centers, true_centers):
        errors.append(haversine_km(pred_lat, pred_lon, float(true_lat), float(true_lon)))
    return errors


def evaluate(model, loader, device):
    model.eval()
    all_errors = []
    with torch.no_grad():
        for x, _, centers, _ in loader:
            x = x.to(device)
            logits = model(x)
            pred_centers = extract_centers_from_logits(logits)
            for pred, true in zip(pred_centers, centers.numpy()):
                all_errors.append(trajectory_km_errors(pred, true))

    errors = np.array(all_errors, dtype=np.float32)
    return {
        "mean_km_error": float(errors.mean()),
        "median_km_error": float(np.median(errors)),
        "per_step_mean_km": errors.mean(axis=0),
    }


def visualize_prediction(model, dataset, sample_idx=0, device="cpu"):
    model.eval()
    x, y, centers, original_idx = dataset[sample_idx]
    with torch.no_grad():
        logits = model(x.unsqueeze(0).to(device))[0].cpu()
    probs = torch.sigmoid(logits)
    pred_centers = extract_centers_from_logits(logits.unsqueeze(0))[0]
    true_centers = centers.numpy()
    errors = trajectory_km_errors(pred_centers, true_centers)

    fig, axes = plt.subplots(2, HORIZON, figsize=(18, 6), constrained_layout=True)
    for i in range(HORIZON):
        axes[0, i].imshow(y[i], origin="upper", extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], cmap="magma")
        axes[0, i].scatter(true_centers[i, 1], true_centers[i, 0], c="cyan", s=35, edgecolor="black")
        axes[0, i].set_title(f"True +{(i + 1) * 6}h")

        axes[1, i].imshow(probs[i], origin="upper", extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], cmap="magma")
        axes[1, i].scatter(pred_centers[i][1], pred_centers[i][0], c="lime", s=35, edgecolor="black")
        axes[1, i].scatter(true_centers[i, 1], true_centers[i, 0], c="cyan", s=25, edgecolor="black")
        axes[1, i].set_title(f"Pred err={errors[i]:.0f} km")

        for ax in axes[:, i]:
            ax.set_xlim(LON_MIN, LON_MAX)
            ax.set_ylim(LAT_MIN, LAT_MAX)
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")

    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(true_centers[:, 1], true_centers[:, 0], "o-", label="IBTrACS true", color="cyan")
    ax.plot([p[1] for p in pred_centers], [p[0] for p in pred_centers], "o-", label="UTAE pred", color="lime")
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(f"6-point trajectory | sample {int(original_idx)}")
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    ax.legend()
    plt.show()


# ======================================================
# TRAIN PIPELINE
# ======================================================
def build_loaders(batch_size=2, val_ratio=0.2):
    storm_df = load_ibtracs_tracks()
    samples = make_track_samples(storm_df)
    print(f"IBTrACS rows: {len(storm_df)}")
    print(f"Trajectory samples: {len(samples)}")
    if not samples:
        raise RuntimeError("No continuous 5-input + 6-target samples found.")

    surface_ds, pressure_ds = open_era5()
    dataset = ERA5IBTrACSDataset(samples, surface_ds, pressure_ds)

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, dataset


def train(epochs=5, batch_size=2, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    train_loader, val_loader, dataset = build_loaders(batch_size=batch_size)

    model = UTAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = HeatmapLoss().to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y, _, _ in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={total_loss / len(train_loader):.4f} | "
            f"val_mean_km={metrics['mean_km_error']:.1f} | "
            f"val_median_km={metrics['median_km_error']:.1f} | "
            f"per_step={np.round(metrics['per_step_mean_km'], 1)}"
        )

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved checkpoint: {CHECKPOINT_PATH}")
    visualize_prediction(model, dataset, sample_idx=0, device=device)
    return model


if __name__ == "__main__":
    train()
