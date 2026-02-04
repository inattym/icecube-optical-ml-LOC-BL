# icecube-optical-ml-LOC-BL

**LOC-BL (Location Baseline)** — frozen reference version where the model first successfully learns spatial (x–y) location conditioning.

This repository is an **archival snapshot**, not an active development branch.



## What this repo contains

- Core training and inference code (`.py`)
- SLURM / job submission scripts
- Analysis and utility scripts
- Metadata and configuration files (`.csv`, `.json`, `.log`)
- Structured run history (`runs/**/info.json`)

All files tracked here are **text-based and human-readable**.



## What is intentionally NOT included

The following artifacts are excluded to keep the repository lightweight and cloneable:

- Model checkpoints (`.pt`, `.pth`, etc.)
- Images / figures (`.png`, `.jpg`, etc.)
- Large binary data products
- Virtual environments

These artifacts are preserved **verbatim** on CHPC.



## Location of full training artifacts (CHPC)

Full artifacts corresponding to this baseline live on CHPC at:
