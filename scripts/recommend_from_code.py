#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, re, textwrap
from pathlib import Path

# ---------- regex helpers ----------
def rx_arg(name, type_="int"):
    if type_ == "int":
        return re.compile(rf'add_argument\(\s*["\']--{name}["\'].*?default\s*=\s*(\d+)', re.S)
    if type_ == "float":
        return re.compile(rf'add_argument\(\s*["\']--{name}["\'].*?default\s*=\s*([\d\.eE+-]+)', re.S)
    if type_ == "str" or type_ == "liststr":
        return re.compile(rf'add_argument\(\s*["\']--{name}["\'].*?default\s*=\s*["\']([^"\']+)["\']', re.S)
    raise ValueError("unknown type")

RX_SET_DEFAULTS_GRAY = re.compile(r'set_defaults\(\s*as_gray\s*=\s*(True|False)\s*\)')
RX_PARAMENC_SIG = re.compile(
    r'class\s+ParamEncoder\(.*?\):.*?def\s+__init__\s*\(\s*self\s*,\s*in_dim\s*:\s*int\s*=\s*(\d+)'
    r'\s*,\s*hidden\s*:\s*int\s*=\s*(\d+)\s*,\s*out_dim\s*:\s*int\s*=\s*(\d+)',
    re.S
)
RX_TINYCOND_SIG = re.compile(
    r'class\s+TinyCondDecoder\(.*?\):.*?def\s+__init__\s*\(.*?param_dim\s*:\s*int\s*=\s*(\d+)'
    r'.*?mlp_hidden\s*:\s*int\s*=\s*(\d+)',
    re.S
)

def extract_int(txt, pattern, fallback):
    m = pattern.search(txt)
    return int(m.group(1)) if m else fallback

def extract_str(txt, pattern, fallback):
    m = pattern.search(txt)
    return m.group(1).strip() if m else fallback

# ---------- parse generator ----------
def parse_generator(gen_path: Path):
    s = gen_path.read_text()
    img_size = extract_int(s, rx_arg("img_size","int"), None)
    gray = bool(re.search(r'convert\(\s*["\']L["\']\s*\)|Image\.new\(\s*["\']L["\']', s))
    channels = 1 if gray else 3
    return {"img_size": img_size, "channels_guess": channels}

# ---------- parse decoder ----------
def parse_decoder(dec_path: Path):
    s = dec_path.read_text()
    start_res = extract_int(s, rx_arg("start_res","int"), 4)
    base_ch   = extract_int(s, rx_arg("base_ch","int"), 256)
    cond_dim  = extract_int(s, rx_arg("cond_dim","int"), 256)
    n_blocks  = extract_int(s, rx_arg("n_blocks_per_up","int"), 2)
    ch_mults_str = extract_str(s, rx_arg("ch_mults","liststr"), "1.0,1.0,0.75,0.5,0.5,0.25")
    ch_mults = tuple(float(x) for x in ch_mults_str.split(","))
    mlp_hidden = extract_int(s, rx_arg("mlp_hidden","int"), None)

    param_dim = None
    m = RX_TINYCOND_SIG.search(s)
    if m:
        param_dim = int(m.group(1))
        if mlp_hidden is None:
            mlp_hidden = int(m.group(2))

    if param_dim is None or mlp_hidden is None:
        m = RX_PARAMENC_SIG.search(s)
        if m:
            if param_dim is None:  param_dim  = int(m.group(1))
            if mlp_hidden is None: mlp_hidden = int(m.group(2))

    if param_dim is None:  param_dim = 4
    if mlp_hidden is None: mlp_hidden = 256

    m = RX_SET_DEFAULTS_GRAY.search(s)
    as_gray_default = (m.group(1) == "True") if m else True

    num_workers_default = extract_int(s, rx_arg("num_workers","int"), 4)

    return {
        "start_res": start_res, "base_ch": base_ch, "cond_dim": cond_dim,
        "n_blocks_per_up": n_blocks, "ch_mults": ch_mults,
        "param_dim": param_dim, "mlp_hidden": mlp_hidden,
        "as_gray_default": as_gray_default,
        "num_workers_default": num_workers_default
    }

# ---------- helpers ----------
def nearest_multiple_of_8(n:int) -> int:
    return max(8, (n // 8) * 8)

def channels_schedule(base_ch:int, ch_mults:tuple[float,...], ups:int):
    chs = [int(base_ch*m) for m in ch_mults]
    while len(chs) < (ups+1):
        chs.append(max(8, chs[-1]//2))
    return chs[:ups+1]

# ---------- autosuggests (continuous) ----------
def autosuggest_num_workers(batch: int|None,
                            cpu_quota: int|None,
                            img_size: int,
                            grayscale: bool) -> int:
    # scaling with workload ~ batch * pixels * channels
    b   = 1 if (batch is None or batch < 1) else batch
    ch  = 1 if grayscale else 3
    eff = b * (img_size**2) * ch

    # Normalize to baseline (8 × 128 × 128 × gray)
    baseline = 8 * (128**2) * 1
    w_norm = max(1.0, eff / baseline)

    # ~1.8 workers at baseline; +0.65 per log2 step; small RGB bump
    workers_f = 1.8 + 0.65 * math.log2(w_norm) + (0.4 if ch == 3 else 0.0)
    nw = int(round(workers_f))
    nw = max(1, min(16, nw))

    # Respect cpu_quota (reserve 1 core for the main process)
    if cpu_quota is not None:
        nw = max(1, min(nw, max(1, cpu_quota - 1)))
    return nw

def autosuggest_cpus_from_workers(nw:int) -> int:
    # 1 main + N workers, soft-clamped
    return max(2, min(32, nw + 1))

def autosuggest_memG(nw:int, img_size:int, batch:int|None, grayscale:bool) -> int:
    # Continuous host-RAM estimate in GB
    b   = 1 if (batch is None or batch < 1) else batch
    ch  = 1 if grayscale else 3
    mpix = (img_size * img_size) / 1_000_000.0  # megapixels per image

    base = 4.5 + 0.15 * nw                # python/OS + light growth with workers
    per_worker_queues = 0.38 * nw         # dataloader queues & transforms
    prefetch = 0.20 * nw * mpix * ch * (1.0 + 0.5 * max(0.0, math.log2(b)))
    batch_queue = 0.030 * b * mpix * ch   # batch staging
    rgb_bump = 0.8 if ch == 3 else 0.0

    est = base + per_worker_queues + prefetch + batch_queue + rgb_bump
    memG = int(round(est))
    return max(6, min(64, memG))

# ---------- memory estimates ----------
def estimate_activation_bytes_per_sample(img_size:int, start_res:int, chs:list[int], n_blocks:int, amp:bool) -> int:
    bytes_per = 2 if amp else 4
    ups = int(math.log2(max(1, img_size // start_res)))
    H = W = start_res
    total = 0
    for i in range(ups):
        H *= 2; W *= 2
        C = chs[i+1]
        maps = 1 + 2*n_blocks
        total += maps * C * H * W * bytes_per
    return int(total * 1.2)

def estimate_param_opt_bytes(base_ch:int, chs:list[int], cond_dim:int, n_blocks:int, amp:bool,
                             param_dim:int, mlp_hidden:int) -> int:
    total_w = 0
    for i in range(len(chs)-1):
        c_in, c_out = chs[i], chs[i+1]
        total_w += (c_in*c_out*3*3) + c_out
        total_w += 2*c_out
        total_w += (2*(cond_dim*c_out) + 2*c_out)
        for _ in range(n_blocks):
            total_w += (c_out*c_out*3*3) + c_out
            total_w += 2*c_out
            total_w += (2*(cond_dim*c_out) + 2*c_out)
            total_w += (c_out*c_out*3*3) + c_out
            total_w += 2*c_out
    total_w += (param_dim*128 + 128)
    total_w += (128*mlp_hidden + mlp_hidden)
    total_w += (mlp_hidden*cond_dim + cond_dim)

    weight_bytes = total_w * (2 if amp else 4)
    opt_bytes = total_w * 12  # fp32 master + m + v
    return weight_bytes + opt_bytes

def human(n:int) -> str:
    g = n / (1024**3)
    if g >= 1: return f"{g:.2f} GB"
    m = n / (1024**2)
    return f"{m:.1f} MB"

def training_activations(act_forward_bytes:int, train_mult:float) -> int:
    return int(act_forward_bytes * train_mult)

def vram_needed_bytes(batch:int, act_per_train:int, params:int,
                      headroom:float, fixed_overhead:int) -> int:
    core = params + fixed_overhead + batch * max(1, act_per_train)
    return int(core / max(1e-9, (1.0 - headroom)))

# ---------- print ----------
def hrule(title: str = ""):
    print("\n" + "="*80)
    if title:
        print(title.upper())
        print("="*80)

def print_summary(out: dict, rec_bs: int | None, target_vram_gb: float | None, rows: list[dict] | None):
    hrule("Model + Data Summary")
    print(f"Image size:           {out['img_size_used']} px")
    print(f"Upsample stages:      {out['upsamples']}")
    print(f"Channels schedule:    {out['channels_schedule']}")
    print(f"AMP assumed:          {out['amp_assumed']}")
    print(f"Activation / sample (fwd):  {out['activation_forward']}")
    print(f"Training activations / sample: {out['activation_train']}")
    print(f"Params + Optimizer:   {out['params_plus_optimizer']}")
    print(f"Headroom:             {out['headroom']*100:.0f}%")
    print(f"Fixed overhead estimate: {out['fixed_overhead']}")
    if target_vram_gb is not None and rec_bs is not None:
        print(f"\nRecommended conservative batch for ~{target_vram_gb:g} GB GPU: {rec_bs}")
    hrule("Batch → Min VRAM (approx.)")
    if rows:
        print(f"{'Batch':>6} | {'Min VRAM Required':>18}")
        print("-"*30)
        for r in rows:
            print(f"{r['batch']:>6} | {r['min_vram_required']:>18}")
    hrule()

def print_slurm_snippet(args, decoder_name, img_size, rec_bs, cpus, memG, num_workers):
    acc_line  = f"#SBATCH --account={args.slurm_account}\n" if args.slurm_account else ""
    cuda_line = f"module load {args.slurm_cuda_module}\n" if args.slurm_cuda_module else ""
    hrule("SLURM Job Template")
    print(textwrap.dedent(f"""\
        #SBATCH --partition={args.slurm_partition}
        {acc_line}#SBATCH --gres={args.slurm_gres}
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --mem={memG}G
        #SBATCH --time={args.slurm_time}

        module purge
        {cuda_line}source {args.slurm_venv}

        python {decoder_name} \\
          --csv data/camsim/meta/train.csv \\
          --val_csv data/camsim/meta/val.csv \\
          --img_size {img_size} \\
          --standardize_params \\
          --batch_size {rec_bs} \\
          --epochs 50 \\
          --num_workers {num_workers} --pin_memory \\
          --amp \\
          --ckpt_every 5 \\
          --out_dir runs/exp_gpu{img_size}
    """))
    hrule()

# ---------- optional live probe ----------
def try_probe_cuda(batch_guess:int, img_size:int, channels:int, amp:bool, device:str) -> dict | None:
    try:
        import torch, torch.nn as nn
        if device != "cuda" or not torch.cuda.is_available():
            return None
        dev = torch.device("cuda:0")
        model = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU()
        ).to(dev)
        # Updated AMP API (no deprecation warning)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        def one_try(b):
            dtype = torch.float16 if amp else torch.float32
            x = torch.randn(b, channels, img_size, img_size, device=dev, dtype=dtype)
            y = torch.zeros_like(x)
            torch.cuda.reset_peak_memory_stats(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize(dev)
            return torch.cuda.max_memory_allocated(dev)

        b = min(max(1, batch_guess), 1024)
        peak_ok = 0
        while True:
            try:
                peak = one_try(b)
                peak_ok = peak
                b2 = b * 2 if b < 128 else b + max(8, b//8)
                if b2 > 4096: break
                b = b2
            except RuntimeError as e:
                if "cuda" in str(e).lower() and "out of memory" in str(e).lower():
                    break
                raise
        return {"max_batch_probe_ok": b, "peak_bytes_at_ok": peak_ok}
    except Exception:
        return None

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    # inputs
    ap.add_argument("--gen", type=str, default="scripts/make_fake_blobs.py")
    ap.add_argument("--decoder", type=str, default="icecube_conditional_decoder_film.py")
    ap.add_argument("--amp", action="store_true", help="assume AMP on for estimates")
    # realistic training multipliers & overheads
    ap.add_argument("--train_multiplier", type=float, default=2.6,
                    help="multiplier to convert forward activations to training-time cost")
    ap.add_argument("--workspace_mb", type=int, default=512,
                    help="extra cuDNN/cublas workspace & miscellany")
    ap.add_argument("--dataloader_mb_per_worker", type=int, default=200,
                    help="host+device buffer, pinned memory, queues per worker")
    ap.add_argument("--extra_fixed_mb", type=int, default=200,
                    help="logging, small tensors, miscellaneous")
    ap.add_argument("--headroom", type=float, default=0.20,
                    help="fractional VRAM reserved for fragmentation/spikes")
    # output controls
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--table-rows", type=int, default=0, help="print up to N batch rows (0 = no table)")
    ap.add_argument("--fallback_cap_vram_gb", type=float, default=80.0,
                    help="when no target VRAM, stop table when estimate exceeds this")
    # Slurm snippet controls
    ap.add_argument("--target_vram_gb", type=float, default=None, help="GPU VRAM to size for (e.g. 24)")
    ap.add_argument("--slurm_partition", type=str, default="notchpeak-gpu")
    ap.add_argument("--slurm_account", type=str, default=None)
    ap.add_argument("--slurm_time", type=str, default="04:00:00")
    ap.add_argument("--slurm_gres", type=str, default="gpu:1")
    ap.add_argument("--slurm_cpus", type=int, default=None, help="override cpus-per-task")
    ap.add_argument("--slurm_memG", type=int, default=None, help="override mem in GB")
    ap.add_argument("--slurm_cuda_module", type=str, default=None)
    ap.add_argument("--slurm_venv", type=str, default="~/icecube-ml/venv/bin/activate")
    # realism knobs
    ap.add_argument("--num_workers_est", type=int, default=None,
                    help="if set, override decoder default for dataloader worker count")
    ap.add_argument("--probe", action="store_true",
                    help="tiny CUDA dry-run to tighten batch rec; only affects recommendation")
    ap.add_argument("--device", type=str, default="cuda", choices=("cuda","cpu"),
                    help="device for probe only")
    args = ap.parse_args()

    gen = parse_generator(Path(args.gen))
    dec = parse_decoder(Path(args.decoder))

    img_size = gen["img_size"] if gen["img_size"] else 128
    img_size = nearest_multiple_of_8(img_size)

    start_res = dec["start_res"]
    ups = int(math.log2(max(1, img_size // start_res)))
    chs = channels_schedule(dec["base_ch"], dec["ch_mults"], ups)

    act_forward = estimate_activation_bytes_per_sample(img_size, start_res, chs, dec["n_blocks_per_up"], args.amp)
    act_train   = training_activations(act_forward, args.train_multiplier)

    params  = estimate_param_opt_bytes(dec["base_ch"], chs, dec["cond_dim"],
                                       dec["n_blocks_per_up"], args.amp,
                                       dec["param_dim"], dec["mlp_hidden"])

    # fixed overhead estimate (host/device misc; depends on workers)
    workers_default = dec["num_workers_default"]
    workers_hint = args.num_workers_est if args.num_workers_est is not None else workers_default
    fixed_mb = args.workspace_mb + args.extra_fixed_mb + (workers_hint * args.dataloader_mb_per_worker)
    fixed_overhead_bytes = fixed_mb * 1024**2

    # optional live probe
    probe_info = None
    if args.probe:
        probe_info = try_probe_cuda(
            batch_guess=16,
            img_size=img_size,
            channels=(gen["channels_guess"] if gen["channels_guess"] else (1 if dec["as_gray_default"] else 3)),
            amp=args.amp,
            device=args.device
        )
        if probe_info and "peak_bytes_at_ok" in probe_info:
            measured_floor = int(probe_info["peak_bytes_at_ok"] * 0.25)
            fixed_overhead_bytes = max(fixed_overhead_bytes, measured_floor)

    # recommend batch for target VRAM
    rec_bs = None
    if args.target_vram_gb is not None:
        vram_bytes = int(args.target_vram_gb * (1024**3))
        usable = int(vram_bytes * (1.0 - args.headroom)) - params - fixed_overhead_bytes
        rec_bs = max(1, usable // max(1, act_train)) if usable > 0 else 1
        if probe_info and probe_info.get("max_batch_probe_ok"):
            rec_bs = min(rec_bs, max(1, int(probe_info["max_batch_probe_ok"] * 0.8)))

    # optional compact table
    rows = None
    if args.table_rows > 0:
        rows = []
        cap_bytes = (args.target_vram_gb if args.target_vram_gb is not None else args.fallback_cap_vram_gb) * (1024**3)
        b = 1
        while len(rows) < args.table_rows:
            need = vram_needed_bytes(b, act_train, params, args.headroom, fixed_overhead_bytes)
            if need > cap_bytes and b > 1:
                break
            rows.append({"batch": b, "min_vram_required": human(need)})
            b += 1
        if rec_bs is not None and (not rows or rows[-1]["batch"] != rec_bs):
            need = vram_needed_bytes(rec_bs, act_train, params, args.headroom, fixed_overhead_bytes)
            rows.append({"batch": rec_bs, "min_vram_required": human(need)})

    # derive autosuggested workers/CPUs/mem for the template (continuous)
    is_gray = bool(gen["channels_guess"] == 1 or dec["as_gray_default"])
    cpu_quota = args.slurm_cpus
    if args.num_workers_est is not None:
        num_workers = args.num_workers_est
    elif dec["num_workers_default"] is not None:
        # honor a decoder default if present, but still clamp under any cpu_quota
        num_workers = dec["num_workers_default"]
        if cpu_quota is not None:
            num_workers = max(1, min(num_workers, max(1, cpu_quota - 1)))
    else:
        num_workers = autosuggest_num_workers(rec_bs, cpu_quota, img_size, is_gray)

    cpus = args.slurm_cpus if args.slurm_cpus else autosuggest_cpus_from_workers(num_workers)
    memG = args.slurm_memG if args.slurm_memG else autosuggest_memG(num_workers, img_size, rec_bs, is_gray)

    out = {
        "from_generator": gen,
        "from_decoder": dec,
        "img_size_used": img_size,
        "upsamples": ups,
        "channels_schedule": chs,
        "amp_assumed": bool(args.amp),
        "activation_forward": human(act_forward),
        "activation_train": human(act_train),
        "params_plus_optimizer": human(params),
        "headroom": args.headroom,
        "fixed_overhead": human(fixed_overhead_bytes),
        "probe": probe_info if probe_info else {}
    }

    print_summary(out, rec_bs, args.target_vram_gb, rows)

    if args.json:
        dump = {
            **out,
            "batch_table": rows if rows else [],
            "recommended_batch": rec_bs,
            "target_vram_gb": args.target_vram_gb,
            "suggested": {"num_workers": num_workers, "cpus": cpus, "memG": memG}
        }
        print("\n# Raw JSON\n" + json.dumps(dump, indent=2))

    if rec_bs is not None:
        print_slurm_snippet(args, Path(args.decoder).name, img_size, rec_bs, cpus, memG, num_workers)

if __name__ == "__main__":
    import re
    main()
    




