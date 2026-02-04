import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add repo root

import argparse, numpy as np, matplotlib.pyplot as plt, torch
from icecube_conditional_decoder_film import TinyCondDecoder, CamSimDataset

ap=argparse.ArgumentParser()
ap.add_argument("--val_csv", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--img_size", type=int, default=64)
ap.add_argument("--idx", type=int, default=0)
ap.add_argument("--standardize_params", action="store_true")
ap.add_argument("--out", type=str, default="runs/exp00/radial_profile.png")
args=ap.parse_args()

ds = CamSimDataset(args.val_csv, img_size=args.img_size,
                   standardize_params=args.standardize_params)
gt, p = ds[args.idx]
model = TinyCondDecoder(img_size=args.img_size)
model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
model.eval()
with torch.no_grad(): pred = model(p.unsqueeze(0)).squeeze(0)

# center at brightest pixel (toy)
def radial_profile(img):
    img = img.squeeze().numpy()
    y0,x0 = np.unravel_index(np.argmax(img), img.shape)
    yy,xx = np.indices(img.shape)
    r = np.sqrt((yy-y0)**2+(xx-x0)**2).astype(np.int32)
    rmax = r.max()
    prof = np.bincount(r.ravel(), img.ravel(), minlength=rmax+1)
    count = np.bincount(r.ravel(), minlength=rmax+1)
    return prof/np.maximum(count,1)

rg = radial_profile(gt); rp = radial_profile(pred)
plt.figure()
plt.plot(rg, label="GT")
plt.plot(rp, label="Pred")
plt.xlabel("radius (pixels)"); plt.ylabel("mean intensity")
plt.legend(); plt.tight_layout(); plt.savefig(args.out)
print("Saved", args.out)
