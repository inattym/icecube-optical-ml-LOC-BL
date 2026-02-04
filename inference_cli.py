cat > infer_once.py <<'PY'
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from icecube_conditional_decoder_film import TinyCondDecoder, CamSimDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"])
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    info_path = run_dir / "info.json"
    ckpt_path = run_dir / args.ckpt
    out_dir = run_dir / "inference_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[info] device: {device}")

    # load hp for dataset paths (csv/base_dir) only
    with open(info_path, "r") as f:
        hp = json.load(f).get("args", {})
    img_size = int(hp.get("img_size", 128))
    as_gray  = bool(hp.get("as_gray", True))
    base_dir = hp.get("base_dir", "")
    train_csv = hp["csv"]; val_csv = hp["val_csv"]

    # dataset (reuse training stats)
    tr_ds = CamSimDataset(train_csv, img_size=img_size, no_header=hp.get("no_header", False),
                          standardize_params=True, base_dir=base_dir,
                          interp=hp.get("interp","auto"), augment=False, as_gray=as_gray)
    va_ds = CamSimDataset(val_csv,   img_size=img_size, no_header=hp.get("no_header", False),
                          standardize_params=True, base_dir=base_dir,
                          interp=hp.get("interp","auto"), augment=False, as_gray=as_gray,
                          p_mean=tr_ds.p_mean, p_std=tr_ds.p_std)
    va_loader = DataLoader(va_ds, batch_size=1, shuffle=False, num_workers=0)

    # load state and INFER the correct architecture
    print(f"[info] loading weights: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    C0 = state["dec.const"].shape[1]
    sr = state["dec.const"].shape[2]
    chs = [C0]; i = 0
    while f"dec.blocks.{i}.conv.weight" in state:
        chs.append(state[f"dec.blocks.{i}.conv.weight"].shape[0]); i += 1
    ch_mults = tuple(float(c)/float(C0) for c in chs)
    cond_dim = state.get("penc.net.4.weight", torch.empty(128,1)).shape[0]
    mlp_hidden = state.get("penc.net.2.weight", torch.empty(128,1)).shape[0]
    has_extra = any(k.startswith("dec.blocks.0.extra.") for k in state.keys())
    n_blocks_per_up = 1 if has_extra else 0

    params = dict(
        param_dim=4,
        cond_dim=int(cond_dim),
        base_ch=int(C0),
        start_res=int(sr),
        img_size=int(img_size),
        ch_mults=tuple(ch_mults),
        up_mode=hp.get("up_mode","bilinear"),
        n_blocks_per_up=int(n_blocks_per_up),
        mlp_hidden=int(mlp_hidden),
    )
    print("[info] inferred params:", params)

    model = TinyCondDecoder(**params).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    saved = 0
    for i, (img, p) in enumerate(va_loader):
        if saved >= args.n: break
        img = img.to(device); p = p.to(device)
        with torch.no_grad():
            pred = model(p).clamp(0,1)
        from torchvision.utils import save_image, make_grid
        (out_dir / "preds").mkdir(exist_ok=True)
        (out_dir / "grids").mkdir(exist_ok=True)
        save_image(pred, out_dir / "preds" / f"pred_{i:04d}.png")
        save_image(img,  out_dir / "preds" / f"gt_{i:04d}.png")
        grid = make_grid(torch.cat([img, pred], dim=0), nrow=2)
        save_image(grid, out_dir / "grids" / f"grid_{i:04d}.png")
        print(f"[save] grids/grid_{i:04d}.png")
        saved += 1
    print(f"[done] wrote {saved} samples → {out_dir}")

if __name__ == "__main__":
    main()
PY
