
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add repo root

from pathlib import Path
import torch, torchvision.utils as vutils
from icecube_conditional_decoder_film import TinyCondDecoder, CamSimDataset
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("--val_csv", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--img_size", type=int, default=64)
ap.add_argument("--out", type=str, default="runs/preview_grid.png")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--standardize_params", action="store_true")
args=ap.parse_args()

ds = CamSimDataset(args.val_csv, img_size=args.img_size,
                   standardize_params=args.standardize_params)
model = TinyCondDecoder(img_size=args.img_size)
model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
model.eval()

# take first n samples
imgs, preds = [], []
for i in range(min(args.n, len(ds))):
    gt, p = ds[i]
    with torch.no_grad():
        pr = model(p.unsqueeze(0)).squeeze(0)
    imgs.append(gt); preds.append(pr)

# stack grid: top row GT, bottom row Pred
grid = vutils.make_grid(torch.stack(imgs+preds,0), nrow=len(imgs), padding=2)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
vutils.save_image(grid, args.out)
print("Saved", args.out)
