# GNPS Local

Local replacement for ProteoSAFe + GNPS web interface.
Single-user, runs entirely in WSL2 on your machine.

## Setup (WSL2, one-time)

```bash
conda activate gnps
cd /mnt/d/GNPS_Local/local_runner
pip install -r requirements.txt
```

## Start

```bash
bash run.sh
# Then open http://localhost:8000 in your Windows browser
```

## What's here

```
local_runner/
├── app.py              FastAPI web app (upload, submit, download)
├── orchestrator.py     Job runner — replaces ProteoSAFe
├── workflows/
│   ├── molecular_networking.py   metabolomics-snets-v2 pipeline
│   ├── fbmn.py                   feature-based-molecular-networking pipeline
│   └── mshub_gc.py               mshub-gc deconvolution pipeline
├── templates/          Jinja2 HTML pages
├── run.sh              Start script
└── requirements.txt
```

Jobs are stored in `~/gnps_jobs/{job_id}/`:
- `input/`   — uploaded files
- `output/`  — all output files (downloadable from UI)
- `run.log`  — full step-by-step log
- `state.json` — job metadata (survives server restarts)

## Spectral libraries

Place downloaded GNPS library MGF files in:
```
/mnt/d/GNPS_Local/libraries/
```

Download from: https://gnps-external.ucsd.edu/gnpslibrary

## Supported workflows

| Workflow | Input | Key outputs |
|---|---|---|
| Molecular Networking | MGF / mzML files | `.graphml`, `clusterinfo.tsv`, `networkedges.tsv` |
| FBMN | MGF + feature table (MZmine/XCMS/etc.) | `.graphml`, `clusterinfo_summary_enriched.tsv` |
| GC-MS Deconvolution | mzML / CDF files | `specs_ms.mgf`, `clusterinfo.tsv`, `quantification.csv` |
| MetVAE Correlation-Based Molecular Network | Feature table as .csv or .tsv and metadata (as .csv or .tsv) if adjusting for covariates | `df_sparse_pval.csv` and/or `df_sparse_sec.csv`; optional: `correlation_graph_<method>_cutoff<cutoff>.graphml`, `df_corr.csv` |

## Notes

- Parallel steps from the original ProteoSAFe workflows are run **sequentially**
  (single-user machine — parallelism set to 1). For large datasets this is slower
  but simpler and more reliable.
- Steps that required UCSD infrastructure (MassIVE dataset matching, Dereplicator,
  QIIME2, ili 3D mapping) are **skipped** gracefully with a log warning.
- The binaries (`MsCluster_bin`, `main_execmodule`) are Linux ELF executables and
  must run inside WSL2 — they cannot run natively on Windows.

## Troubleshooting

**Binary permission denied:**
```bash
chmod +x /mnt/d/GNPS_Local/metabolomics-snets-v2/tools/metabolomicsnetsv2/binaries/*
```

**Script not found errors:**
The script names in `workflows/*.py` are derived from `tool.xml`. If a step fails
with "No such file", check the actual filename in the workflow's `tools/.../scripts/`
directory and update the corresponding `workflows/*.py` wrapper.

**Jobs directory:**
```bash
ls ~/gnps_jobs/
```