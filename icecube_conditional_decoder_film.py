"""
Large FiLM-conditioned decoder for CamSim
"""
# ---------- Imports ----------
import sys, os, json, math, csv, argparse
from pathlib import Path
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from torch.amp import GradScaler, autocast  # new AMP API

# add repo root to path (for relative imports if needed later)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))




def _vis_stretch(x: torch.Tensor, eps: float = 1e-8):
    """
    x: (1,1,H,W) float tensor (or (B,1,H,W) works too)
    returns: stretched to [0,1] using per-image max
    """
    x = x.clone()
    m = x.max().clamp_min(eps)
    return (x / m).clamp(0, 1)


def vis_log(x: torch.Tensor, eps: float = 1e-6, gamma: float = 0.4):
    """
    Log + gamma visualization for sparse images.
    Input: (B,1,H,W) or (1,1,H,W) in [0,1] (or any nonnegative).
    Output: uint8 tensor in [0,255] with same shape.
    """
    x = x.clamp_min(0)
    x = torch.log(x + eps)
    x = x - x.amin(dim=(2, 3), keepdim=True)
    x = x / (x.amax(dim=(2, 3), keepdim=True) + 1e-8)
    x = x ** gamma
    return (255 * x).byte()




def _draw_dot(im: Image.Image, x: float, y: float, color=(255, 0, 0), r: int = 3):
    """
    Draw a filled dot at (x,y) on a PIL image.
    im should be RGB.
    """
    W, H = im.size
    x = int(round(x))
    y = int(round(y))
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    draw = ImageDraw.Draw(im)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    return im





def save_val_grid(model, dataset, device, out_path: Path, k: int = 8):
    """
    Saves a simple PNG grid:
    each row: [target | prediction | abs diff]
    for first k samples in val set (deterministic).
    """
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for i in range(min(k, len(dataset))):
            img, p, xy = dataset[i]          # xy is (2,) raw pixel coords after resize
            img = img.unsqueeze(0).to(device)
            p   = p.unsqueeze(0).to(device)

            pred = model(p).clamp(0, 1).cpu()     # (1,1,H,W)
            tgt  = img.clamp(0, 1).cpu()          # (1,1,H,W)
            diff = (pred - tgt).abs()

            # --- compute predicted centroid in pixel coords ---
            c_pred = soft_centroid(pred.to(torch.float32), power=4.0)[0]  # (2,) [cx, cy]
            cx, cy = float(c_pred[0].cpu()), float(c_pred[1].cpu())

            # --- GT xy ---
            gx, gy = float(xy[0].cpu()), float(xy[1].cpu())

            # --- VIS: log+gamma so sparse dots show up ---
            tgt_u8  = vis_log(tgt)[0, 0].numpy()   # (H,W) uint8
            pred_u8 = vis_log(pred)[0, 0].numpy()
            diff_u8 = vis_log(diff)[0, 0].numpy()

            # --- convert to RGB and overlay dots ---
            tgt_im  = Image.fromarray(tgt_u8).convert("RGB")
            pred_im = Image.fromarray(pred_u8).convert("RGB")
            diff_im = Image.fromarray(diff_u8).convert("RGB")

            # draw GT (green) and predicted centroid (red)
            tgt_im  = _draw_dot(tgt_im,  gx, gy, color=(0, 255, 0), r=3)
            pred_im = _draw_dot(pred_im, gx, gy, color=(0, 255, 0), r=3)   # GT in pred tile too
            pred_im = _draw_dot(pred_im, cx, cy, color=(255, 0, 0), r=3)   # predicted centroid

            # concatenate 3 panels into one row
            row = np.concatenate([np.array(tgt_im), np.array(pred_im), np.array(diff_im)], axis=1)
            rows.append(row)

    grid = np.concatenate(rows, axis=0)

    Image.fromarray(grid).save(str(out_path))






@torch.no_grad()
def conditioning_sanity(model, dataset, device, out_path: Path, idx: int = 0, dx: float = 10.0):
    """
    Compare predictions with slightly shifted conditioning.
    Row: [pred(x,y) | pred(x+dx,y) | abs diff]
    """
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _, p, _ = dataset[idx]
    p = p.unsqueeze(0).to(device)

    p_shift = p.clone()
    p_shift[0, 0] += float(dx) / float(dataset.p_std[0])  # shift x in standardized units

    pred1 = model(p).clamp(0, 1).detach().cpu()
    pred2 = model(p_shift).clamp(0, 1).detach().cpu()
    diff  = (pred1 - pred2).abs()

    grid = torch.cat([vis_log(pred1), vis_log(pred2), vis_log(diff)], dim=3)

    grid = grid.squeeze(0).permute(1, 2, 0).numpy()
    if grid.shape[2] == 1:
        grid = grid[:, :, 0]  # (H, W*3)

    Image.fromarray(grid).save(str(out_path))






# ---------- Utils ----------


def _nearest_odd(n: int) -> int:
    return n if (n % 2 == 1) else (n + 1)

@lru_cache(maxsize=16)
def _gaussian_kernel_cached(win: int, sigma: float) -> np.ndarray:
    """Return a cached (win x win) Gaussian kernel as float32 numpy array (sum=1)."""
    coords = np.arange(win, dtype=np.float32) - win // 2
    g = np.exp(-(coords**2) / (2.0 * sigma * sigma))
    g /= g.sum()
    return np.outer(g, g).astype(np.float32)

def _gaussian_window(window_size: int, sigma: float, device, dtype=torch.float32) -> torch.Tensor:
    """Torch tensor (win x win) on device/dtype, built from cached numpy kernel."""
    k2d = _gaussian_kernel_cached(int(window_size), float(sigma))
    return torch.from_numpy(k2d).to(device=device, dtype=dtype)

def ssim(img1: torch.Tensor,
         img2: torch.Tensor,
         window_size: int | None = None,
         sigma: float | None = None,
         L: float = 1.0,
         pad_mode: str | None = "reflect") -> torch.Tensor:
    """
    SSIM for (B,C,H,W) tensors in [0, L], numerically-stable.
    Dynamic window: odd ~= min(H,W)/12, floor=3, and valid for reflect padding.
    """
    assert img1.shape == img2.shape, "SSIM inputs must have the same shape"

    # Force compute in fp32 to avoid fp16/CPU autocast instabilities
    x1 = img1.to(dtype=torch.float32)
    x2 = img2.to(dtype=torch.float32)

    B, C, H, W = x1.shape
    device = x1.device

    # dynamic window sizing
    if window_size is None:
        base = max(3, int(round(min(H, W) / 12)))
        window_size = _nearest_odd(base)
        # reflect padding validity (p = win//2 <= size-1  => win <= 2*size-2)
        max_win = 2 * min(H, W) - 2
        if window_size > max_win:
            window_size = _nearest_odd(max(3, max_win))

    if sigma is None:
        sigma = window_size / 3.0

    k2d = _gaussian_window(window_size, sigma, device, torch.float32)         # (win, win)
    kernel = k2d.view(1, 1, window_size, window_size).expand(C, 1, -1, -1)    # (C,1,win,win)

    def blur(x: torch.Tensor) -> torch.Tensor:
        if pad_mode:
            p = window_size // 2
            x = F.pad(x, (p, p, p, p), mode=pad_mode)
            return F.conv2d(x, kernel, groups=C)
        return F.conv2d(x, kernel, padding=window_size // 2, groups=C)

    mu1 = blur(x1)
    mu2 = blur(x2)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = blur(x1 * x1) - mu1_sq
    sigma2_sq = blur(x2 * x2) - mu2_sq
    sigma12   = blur(x1 * x2) - mu1_mu2

    sigma1_sq = sigma1_sq.clamp_min(0.0)
    sigma2_sq = sigma2_sq.clamp_min(0.0)

    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = (num / den.clamp_min(1e-12)).clamp(0.0, 1.0)
    return ssim_map.mean()

# ---------- Dataset ----------
class CamSimDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 img_size: int | None = None,     # auto-detect if None
                 no_header: bool = False,
                 standardize_params: bool = True,
                 base_dir: str | None = None,
                 p_mean: np.ndarray | list | None = None,   # reuse train stats for val/test
                 p_std:  np.ndarray | list | None = None,
                 interp: str = "auto",            # 'auto' | 'bicubic' | 'bilinear' | 'nearest'
                 augment: bool = False,
                 as_gray: bool = True):
        """
        CSV: [image_path, x, y]
        """
        # load rows
        raw_rows = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            if not no_header:
                next(reader, None)
            for r in reader:
                if r and len(r) >= 3:
                    raw_rows.append(r)

        # resolve paths & filter missing
        rows = []
        for r in raw_rows:
            p = Path(r[0])
            if base_dir and not p.is_absolute():
                p = Path(base_dir) / p
            if p.exists():
                rows.append([str(p), r[1], r[2]])  # x, y
        if not rows:
            raise RuntimeError(f"No valid image paths found in {csv_path}")
        self.rows = rows

        # auto-detect image size if needed (then align to multiple of 8 for the decoder)
        if img_size is None:
            try:
                with Image.open(self.rows[0][0]) as im:
                    w, h = im.size
                    img_size = min(w, h)
                    img_size = max(8, (img_size // 8) * 8)  # nearest multiple of 8 (floor), min 8
            except Exception as e:
                raise RuntimeError(f"Cannot auto-detect image size: {e}")
        self.img_size = int(img_size)

        # interpolation mode
        if interp == "auto":
            imode = InterpolationMode.BICUBIC if self.img_size <= 128 else InterpolationMode.BILINEAR
        elif interp == "bicubic":
            imode = InterpolationMode.BICUBIC
        elif interp == "bilinear":
            imode = InterpolationMode.BILINEAR
        else:
            imode = InterpolationMode.NEAREST

        self.resize = transforms.Resize((self.img_size, self.img_size),
                                        interpolation=imode, antialias=True)
        self.to_tensor = transforms.ToTensor()
        self.as_gray = bool(as_gray)

        # param stats
        arr = np.asarray([[float(r[1]), float(r[2])] for r in self.rows], dtype=np.float32)  # x, y
        if standardize_params:
            if p_mean is not None and p_std is not None:
                self.p_mean = np.asarray(p_mean, dtype=np.float32)
                self.p_std  = np.asarray(p_std,  dtype=np.float32)
            else:
                self.p_mean = arr.mean(0)
                self.p_std  = arr.std(0) + 1e-6
        else:
            self.p_mean = np.zeros(2, dtype=np.float32)
            self.p_std  = np.ones(2,  dtype=np.float32)

        # light augmentation (optional; symmetry flips)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        img_path, x, y = self.rows[idx]

        with Image.open(img_path) as im:
            img = im.convert("L") if self.as_gray else im.convert("RGB")
            W0, H0 = img.size

            x = float(x)
            y = float(y)

            # --- rescale coords to resized image ---
            sx = self.img_size / W0
            sy = self.img_size / H0
            x *= sx
            y *= sy

            # --- IMPORTANT: CSV y is bottom-origin; image y is top-origin ---
            y = (self.img_size - 1) - y

            img = self.resize(img)



            # --- optional augmentation: flip image AND coords ---
            if self.augment:
                if torch.rand(()) < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    x = (self.img_size - 1) - x

                if torch.rand(()) < 0.5:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    y = (self.img_size - 1) - y

        img = self.to_tensor(img)

        params = torch.tensor([
            (x - self.p_mean[0]) / self.p_std[0],
            (y - self.p_mean[1]) / self.p_std[1],
        ], dtype=torch.float32)

        xy_raw = torch.tensor([x, y], dtype=torch.float32)

        return img, params, xy_raw

# ---------- Model ----------
class ParamEncoder(nn.Module):
    """
    Encodes the param vector into a conditioning embedding.
    """
    def __init__(self, in_dim: int = 4, hidden: int = 256, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.SiLU(),
            nn.Linear(128, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, p):
        return self.net(p)

class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation (FiLM): y = x * (1 + gamma) + beta"""
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, channels)
        self.to_beta  = nn.Linear(cond_dim, channels)
    def forward(self, x, cond):
        gamma = self.to_gamma(cond)[:, :, None, None]
        beta  = self.to_beta(cond)[:,  :, None, None]
        return x * (1 + gamma) + beta

class ResFiLMBlock(nn.Module):
    """Conv -> GN -> SiLU -> FiLM -> Conv -> GN -> residual add."""
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        groups = max(1, min(16, channels // 8))
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1   = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.film1 = FiLMBlock(channels, cond_dim)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2   = nn.GroupNorm(num_groups=groups, num_channels=channels)

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)
        h = self.film1(h, cond)
        h = self.conv2(h)
        h = self.gn2(h)
        return F.silu(h + x)

class UpsampleBlock(nn.Module):
    """
    Upsample×2 -> Conv(3×3) -> GN -> SiLU -> FiLM
    then (n_resblocks) of ResFiLMBlock for extra depth.
    """
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, up_mode: str = "bilinear", n_resblocks: int = 2):
        super().__init__()
        if up_mode == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        groups = max(1, min(16, out_ch // 8))
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_ch)
        self.film = FiLMBlock(out_ch, cond_dim)
        self.extra = nn.ModuleList([ResFiLMBlock(out_ch, cond_dim) for _ in range(n_resblocks)])

    def forward(self, x, cond):
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        x = F.silu(x)
        x = self.film(x, cond)
        for blk in self.extra:
            x = blk(x, cond)
        return x

class CondDecoder(nn.Module):
    """
    FiLM-conditioned decoder that grows resolution by powers of two.
    Output is grayscale (1 channel).
    """
    def __init__(self,
                 cond_dim:   int = 256,
                 start_res:  int = 4,
                 img_size:   int = 128,
                 base_ch:    int = 256,
                 ch_mults:   tuple = (1.0, 1.0, 0.75, 0.5, 0.5, 0.25),
                 up_mode:    str = "bilinear",
                 n_blocks_per_up: int = 2):
        super().__init__()
        out_channels = 1  # grayscale

        assert img_size % start_res == 0, "img_size must be a multiple of start_res"
        ups = int(math.log2(img_size // start_res))
        assert 2 ** ups == (img_size // start_res), "img_size/start_res must be a power of 2 (×2 upsampling)"

        c0 = int(base_ch * ch_mults[0])
        self.const = nn.Parameter(torch.randn(1, c0, start_res, start_res) * 0.02)

        chs = [int(base_ch * m) for m in ch_mults]
        while len(chs) < (ups + 1):
            chs.append(max(8, chs[-1] // 2))

        self.blocks = nn.ModuleList([
            UpsampleBlock(chs[i], chs[i + 1], cond_dim, up_mode=up_mode, n_resblocks=n_blocks_per_up)
            for i in range(ups)
        ])
        self.final = nn.Conv2d(chs[ups], out_channels, kernel_size=3, padding=1)

    def forward(self, cond):
        B = cond.size(0)
        x = self.const.expand(B, -1, -1, -1)
        for blk in self.blocks:
            x = blk(x, cond)
        x = self.final(x)
        return torch.sigmoid(x)

class TinyCondDecoder(nn.Module):
    """Full model = ParamEncoder + CondDecoder (grayscale)."""
    def __init__(self,
                 param_dim:  int = 4,
                 cond_dim:   int = 256,
                 base_ch:    int = 256,
                 start_res:  int = 4,
                 img_size:   int = 128,
                 ch_mults:   tuple = (1.0, 1.0, 0.75, 0.5, 0.5, 0.25),
                 up_mode:    str = "bilinear",
                 n_blocks_per_up: int = 2,
                 mlp_hidden: int = 256):
        super().__init__()
        self.penc = ParamEncoder(in_dim=param_dim, hidden=mlp_hidden, out_dim=cond_dim)
        self.dec  = CondDecoder(cond_dim=cond_dim,
                                start_res=start_res,
                                img_size=img_size,
                                base_ch=base_ch,
                                ch_mults=ch_mults,
                                up_mode=up_mode,
                                n_blocks_per_up=n_blocks_per_up)
    def forward(self, p):
        cond = self.penc(p)
        return self.dec(cond)

# ---------- Training ----------
def _parse_ch_mults(s: str | None, default=(1.0, 1.0, 0.75, 0.5, 0.5, 0.25)):
    if not s or s.strip() == "":
        return tuple(default)
    try:
        vals = [float(x) for x in s.split(",")]
        assert all(v > 0 for v in vals)
        return tuple(vals)
    except Exception:
        print(f"[warn] Could not parse --ch_mults='{s}', using default {default}")
        return tuple(default)

def soft_centroid(pred: torch.Tensor, eps: float = 1e-8, power: float = 1.0):
    """
    pred: (B,1,H,W) in [0,1]
    returns: (B,2) centroid in pixel coords (x,y)
    """
    assert pred.dim() == 4 and pred.size(1) == 1
    B, _, H, W = pred.shape

    w = pred.clamp_min(0.0)
    if power != 1.0:
        w = w.pow(power)

    wsum = w.sum(dim=(2,3), keepdim=True).clamp_min(eps)
    w = w / wsum

    xs = torch.linspace(0, W - 1, W, device=pred.device, dtype=pred.dtype).view(1,1,1,W)
    ys = torch.linspace(0, H - 1, H, device=pred.device, dtype=pred.dtype).view(1,1,H,1)

    cx = (w * xs).sum(dim=(2,3))
    cy = (w * ys).sum(dim=(2,3))
    return torch.cat([cx, cy], dim=1)





@torch.no_grad()
def check_xy_convention(dataset, n=200):
    # tests which mapping of (x,y) best matches centroid(img)
    # candidates: identity, swap, yflip, xflip, swap+yflip, swap+xflip
    errs = {k: [] for k in ["id", "swap", "yflip", "xflip", "swap_yflip", "swap_xflip"]}

    for i in range(min(n, len(dataset))):
        img, _, xy = dataset[i]             # img: (1,H,W), xy: (2,)
        H, W = img.shape[1], img.shape[2]

        tgt = img.unsqueeze(0)              # (1,1,H,W)
        c = soft_centroid(tgt.float(), power=4.0)[0]   # (cx,cy) in pixels from image itself

        x, y = float(xy[0]), float(xy[1])

        cand = {
            "id":        (x, y),
            "swap":      (y, x),
            "yflip":     (x, (H - 1) - y),
            "xflip":     ((W - 1) - x, y),
            "swap_yflip":(y, (H - 1) - x),
            "swap_xflip":((W - 1) - y, x),
        }

        for k, (xx, yy) in cand.items():
            dx = (c[0].item() - xx)
            dy = (c[1].item() - yy)
            errs[k].append(dx*dx + dy*dy)

    for k in errs:
        print(k, "MSE(px^2) =", sum(errs[k]) / max(1, len(errs[k])))

# Example use inside train() after build `va`:
# check_xy_convention(va, n=200)





@torch.no_grad()
def weight_checksum(model: nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        s += float(p.detach().abs().mean().cpu())
    return s


def loss_fn(pred, target, xy_raw, lambda_xy: float = 0.1):
    pred   = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)

    mse = F.mse_loss(pred, target)
    s   = ssim(pred, target, L=1.0, window_size=5, sigma=1.5)

    # --- xy supervision (normalize to [0,1]) ---
    B, _, H, W = pred.shape

    c_pred = soft_centroid(pred.to(torch.float32), power=4.0)  # pixels
    xy_t   = xy_raw.to(torch.float32)                          # pixels

    # normalize both to [0,1]
    c_pred[:, 0] = c_pred[:, 0] / (W - 1)
    c_pred[:, 1] = c_pred[:, 1] / (H - 1)
    xy_t[:, 0]   = xy_t[:, 0]   / (W - 1)
    xy_t[:, 1]   = xy_t[:, 1]   / (H - 1)

    loss_xy = F.mse_loss(c_pred, xy_t)

    loss = 0.98 * mse + 0.02 * (1.0 - s) + lambda_xy * loss_xy
    return loss, mse.item(), s.item(), float(loss_xy.detach().cpu())




@torch.no_grad()
def validate(model, loader, device, amp=False):
    model.eval()
    mse_sum = ssim_sum = xy_sum = 0.0
    n = 0
    use_amp = (amp and device.type == "cuda")

    for img, p, xy_raw in loader:
        img    = img.float().to(device, non_blocking=True)
        p      = p.float().to(device, non_blocking=True)
        xy_raw = xy_raw.float().to(device, non_blocking=True)


        # AMP only on CUDA; avoid autocast(device_type="cpu"/"mps") weirdness
        if use_amp:
            with autocast(device_type="cuda"):
                pred = model(p)
        else:
            pred = model(p)

        # ---- DEBUG: fingerprint first validation batch only ----
        if n == 0:
            fp_mean = float(pred.detach().mean().cpu())
            fp_std  = float(pred.detach().std().cpu())
            fp_abs  = float(pred.detach().abs().mean().cpu())
            print(
                f"[val dbg] pred mean={fp_mean:.10f} "
                f"std={fp_std:.10f} absmean={fp_abs:.10f}"
            )

        pred = pred.clamp(0.0, 1.0)
        img  = img.clamp(0.0, 1.0)

        mse_sum  += F.mse_loss(pred, img).item()
        ssim_sum += ssim(pred, img, L=1.0, window_size=5, sigma=1.5).item()


        B, _, H, W = pred.shape

        c_pred = soft_centroid(pred.to(torch.float32), power=4.0)  # pixels
        xy_t   = xy_raw.to(torch.float32)                          # pixels

        c_pred[:, 0] = c_pred[:, 0] / (W - 1)
        c_pred[:, 1] = c_pred[:, 1] / (H - 1)
        xy_t[:, 0]   = xy_t[:, 0]   / (W - 1)
        xy_t[:, 1]   = xy_t[:, 1]   / (H - 1)

        xy_sum += F.mse_loss(c_pred, xy_t).item()


        n += 1

    return mse_sum / max(n, 1), ssim_sum / max(n, 1), xy_sum / max(n, 1)

def train(args):
    # --- out_dir handling: auto-pick runs/expXX if not provided
    if not args.out_dir or args.out_dir.strip() == "":
        base = Path("runs"); base.mkdir(exist_ok=True)
        i = 0
        while (base / f"exp{i:02d}").exists():
            i += 1
        args.out_dir = str(base / f"exp{i:02d}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # device selection: cuda → mps → cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[info] device: {device}")

    # persist run info
    with open(Path(args.out_dir) / "info.json", "w") as f:
        json.dump({"device": str(device), "args": vars(args)}, f, indent=2)

    # --- Data ---
    ch_mults = _parse_ch_mults(args.ch_mults)
    tr = CamSimDataset(args.csv,
                       img_size=args.img_size,
                       no_header=args.no_header,
                       standardize_params=args.standardize_params,
                       base_dir=args.base_dir,
                       interp=args.interp,
                       augment=args.augment,
                       as_gray=args.as_gray)
    va = CamSimDataset(args.val_csv,
                       img_size=args.img_size,
                       no_header=args.no_header,
                       standardize_params=True,
                       base_dir=args.base_dir,
                       interp=args.interp,
                       augment=False,
                       as_gray=args.as_gray,
                       p_mean=tr.p_mean,
                       p_std=tr.p_std)

    tr_loader = DataLoader(tr,
                           batch_size=args.batch_size,
                           shuffle=True,
                           num_workers=args.num_workers,
                           pin_memory=args.pin_memory,
                           persistent_workers=(args.num_workers > 0))
    va_loader = DataLoader(va,
                           batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=args.num_workers,
                           pin_memory=args.pin_memory,
                           persistent_workers=(args.num_workers > 0))

    # --- Model ---
    model = TinyCondDecoder(
        param_dim=2,
        cond_dim=args.cond_dim,
        base_ch=args.base_ch,
        start_res=args.start_res,
        img_size=args.img_size,
        ch_mults=ch_mults,
        up_mode=args.up_mode,
        n_blocks_per_up=args.n_blocks_per_up,
        mlp_hidden=args.mlp_hidden
    ).to(device)

    # Optional resume
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        print(f"[info] resumed weights from {args.resume}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))

    w0 = weight_checksum(model)
    print(f"[dbg] init weight checksum: {w0:.6f}")

    best_ssim = -1.0

    for ep in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "mse": 0.0, "ssim": 0.0, "xy": 0.0, "n": 0}

        use_amp = (args.amp and device.type == "cuda")

        for img, p, xy_raw in tr_loader:
            img    = img.float().to(device, non_blocking=True)
            p      = p.float().to(device, non_blocking=True)
            xy_raw = xy_raw.float().to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # AMP only on CUDA; on CPU/MPS this runs as normal fp32
            if use_amp:
                with autocast(device_type="cuda"):
                    pred = model(p)
                    loss, mse_item, ssim_item, xy_item = loss_fn(pred, img, xy_raw, lambda_xy=1.0)
            else:
                pred = model(p)
                loss, mse_item, ssim_item, xy_item = loss_fn(pred, img, xy_raw, lambda_xy=1.0)

            running["xy"] += xy_item

            # backward
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

            running["loss"] += float(loss.detach().cpu())
            running["mse"]  += mse_item
            running["ssim"] += ssim_item
            running["n"]    += 1

        tr_loss = running["loss"] / max(running["n"], 1)
        tr_mse  = running["mse"]  / max(running["n"], 1)
        tr_ssim = running["ssim"] / max(running["n"], 1)
        tr_xy   = running["xy"]   / max(running["n"], 1)


        val_mse, val_ssim, val_xy = validate(model, va_loader, device, amp=args.amp)
        if ep in {1, 5, 10, 20}:
            conditioning_sanity(
                model, va, device,
                Path(args.out_dir) / f"cond_sanity_ep{ep:03d}.png",
                idx=0, dx=12.0
            )
            save_val_grid(
                model, va, device,
                Path(args.out_dir) / f"val_grid_ep{ep:03d}.png",
                k=8
            )


        w1 = weight_checksum(model)
        print(f"[dbg] ep{ep:03d} weight checksum: {w1:.6f} (delta {w1 - w0:+.6f})")
        w0 = w1

        print(
            f"Epoch {ep:03d} | "
            f"Train loss {tr_loss:.8f} (mse {tr_mse:.8f}, ssim {tr_ssim:.6f}, xy {tr_xy:.8f}) | "
            f"Val MSE {val_mse:.8f} | Val SSIM {val_ssim:.6f} | Val XY {val_xy:.8f}"
        )

        # save best
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), Path(args.out_dir) / "best.pt")

        # periodic checkpoints
        if args.ckpt_every and args.ckpt_every > 0 and (ep % args.ckpt_every == 0):
            if args.keep_all_ckpts:
                torch.save(model.state_dict(), Path(args.out_dir) / f"ep{ep:03d}.pt")
            else:
                torch.save(model.state_dict(), Path(args.out_dir) / "ep_latest.pt")

    # save last epoch weights
    torch.save(model.state_dict(), Path(args.out_dir) / "last_ep.pt")

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--base_dir", type=str, default="", help="Prefix to prepend to CSV image paths")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--no_header", action="store_true")
    ap.add_argument("--standardize_params", action="store_true")
    ap.add_argument("--interp", type=str, default="auto", choices=["auto","bicubic","bilinear","nearest"])
    ap.add_argument("--augment", action="store_true", help="Enable random H/V flips on training set")

    # NOTE: grayscale-only model (decoder outputs 1 channel)
    ap.add_argument("--as_gray", action="store_true", help="Use grayscale images (default)")
    ap.set_defaults(as_gray=True)

    # model capacity (Large defaults)
    ap.add_argument("--start_res", type=int, default=4, help="Const feature map size; must divide img_size")
    ap.add_argument("--base_ch", type=int, default=256)
    ap.add_argument("--cond_dim", type=int, default=256)
    ap.add_argument("--mlp_hidden", type=int, default=256)
    ap.add_argument("--ch_mults", type=str, default="1.0,1.0,0.75,0.5,0.5,0.25",
                    help='Comma list like "1.0,1.0,0.75,0.5,0.5,0.25".')
    ap.add_argument("--up_mode", type=str, default="bilinear", choices=["nearest", "bilinear"])
    ap.add_argument("--n_blocks_per_up", type=int, default=2, help="Extra ResFiLM blocks per upsample stage")

    # optimization
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Max grad norm; 0 = disabled")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only)")

    # io / runtime
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--ckpt_every", type=int, default=10, help="Save epXXX.pt every N epochs (0=disable)")
    ap.add_argument("--keep_all_ckpts", action="store_true", help="If set, keep all periodic checkpoints")
    ap.add_argument("--resume", type=str, default="", help="Path to .pt to load weights from")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")

    ap.add_argument(
        "--write_grid_only",
        action="store_true",
        help="Skip training; load --resume (or last_ep.pt in --out_dir) and write grids")

    return ap.parse_args()





if __name__ == "__main__":
    args = parse_args()

    def _ensure_out_dir(args):
        if not args.out_dir or args.out_dir.strip() == "":
            base = Path("runs")
            base.mkdir(exist_ok=True)
            i = 0
            while (base / f"exp{i:02d}").exists():
                i += 1
            args.out_dir = str(base / f"exp{i:02d}")
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # If wants grid only, don't train.
    if args.write_grid_only:
        _ensure_out_dir(args)

        # pick checkpoint: prefer --resume, else out_dir/last_ep.pt
        ckpt_path = Path(args.resume) if args.resume else (Path(args.out_dir) / "last_ep.pt")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tr = CamSimDataset(
            args.csv,
            img_size=args.img_size,
            no_header=args.no_header,
            standardize_params=True,
            base_dir=args.base_dir,
            interp=args.interp,
            augment=False,
            as_gray=args.as_gray
        )

        va = CamSimDataset(
            args.val_csv,
            img_size=args.img_size,
            no_header=args.no_header,
            standardize_params=True,
            base_dir=args.base_dir,
            interp=args.interp,
            augment=False,
            as_gray=args.as_gray,
            p_mean=tr.p_mean,
            p_std=tr.p_std
        )

        print("\n[XY CHECK] comparing CSV (x,y) to GT-image centroid\n")
        check_xy_convention(va, n=200)
        print()


        model = TinyCondDecoder(
            param_dim=2,
            cond_dim=args.cond_dim,
            base_ch=args.base_ch,
            start_res=args.start_res,
            img_size=args.img_size,
            ch_mults=_parse_ch_mults(args.ch_mults),
            up_mode=args.up_mode,
            n_blocks_per_up=args.n_blocks_per_up,
            mlp_hidden=args.mlp_hidden
        ).to(device)

        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)

        model.eval()
        with torch.no_grad():
            img0, p0, _ = va[0]
            p0 = p0.unsqueeze(0).to(device)
            pred0 = model(p0).clamp(0, 1).detach().cpu()
            print("[AFTER dbg] pred min/max/mean:",
                float(pred0.min()), float(pred0.max()), float(pred0.mean()))

        save_val_grid(
            model, va, device,
            Path(args.out_dir) / "val_grid_AFTER.png",
            k=8
        )

        conditioning_sanity(
            model, va, device,
            Path(args.out_dir) / "cond_sanity_AFTER.png",
            idx=0, dx=12.0
        )

        print(f"[info] wrote grids to {args.out_dir}")
        raise SystemExit(0)

    # otherwise: normal training
    train(args)






    
    