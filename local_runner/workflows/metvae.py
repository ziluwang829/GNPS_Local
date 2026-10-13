"""
MetVAE Correlation-Based Molecular Network workflow.
Builds a sparse metabolite correlation network from a feature table.

Pipeline (the steps timed in MetVAE's quickstart notebook; each runs
python -m workflows.metvae_stage --stage <stage>):
  1. metvae_initialize_model        read the table, MetVAE(...)
       inputs:  feature table (.csv or .tsv), optional metadata
  2. metvae_train_model             model.train()
  3. metvae_estimate_correlations   model.get_corr()
  4. metvae_pvalue_sparsification   model.sparse_by_p()     (one step per chosen method)
     metvae_sec_sparsification      model.sparse_by_sec()
       outputs: df_sparse_<method>.csv, df_corr.csv, correlation_graph_<method>_cutoff<c>.graphml,
                metvae/ (model_state.pth, p_values.csv, q_values.csv, ...)

Entirely Python, no binaries.
"""

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator import Job

STAGE_MODULE = "workflows.metvae_stage"
LOCAL_RUNNER_DIR = Path(__file__).resolve().parents[1]

# Training and SEC can run for hours on large tables
METVAE_TIMEOUT = 6 * 3600

SPARSE_STEPS = {"pval": "metvae_pvalue_sparsification", "sec": "metvae_sec_sparsification"}

# Outputs the form can turn off
OPTIONAL_OUTPUTS = ("graphml", "corr", "pvalues", "checkpoint")


def run(job: "Job") -> bool:
    p = job.params
    input_dir = job.input_dir
    out = job.output_dir

    python = sys.executable

    # Locate inputs
    quant_table = _find_input(input_dir, ["quantification_table.*"]) or \
        _find_input(input_dir, ["*.csv", "*.tsv"], exclude="metadata")
    if not quant_table:
        job.log("ERROR: No feature table found in input directory")
        return False
    metadata_file = input_dir / "metadata.csv"
    if not metadata_file.exists():
        metadata_file = input_dir / "metadata.tsv"

    categorical = _split_names(p.get("METVAE_CATEGORICAL_COVARIATES", ""))
    continuous = _split_names(p.get("METVAE_CONTINUOUS_COVARIATES", ""))
    covariates = continuous + categorical
    if covariates and not metadata_file.exists():
        job.log("ERROR: Covariates were given but no metadata table was uploaded")
        return False
    if metadata_file.exists() and not covariates:
        job.log("Metadata uploaded without covariates; MetVAE will not adjust for any covariate")

    job.log(f"Feature table: {quant_table}")
    job.log(f"Metadata: {metadata_file if covariates else 'not used'}")

    metvae_out = out / "metvae"
    metvae_out.mkdir(exist_ok=True)

    # ── MetVAE steps ──────────────────────────────────────────────────────────
    methods = [m for m in p.get("METVAE_SPARSE_METHODS", "pval").split(",") if m in SPARSE_STEPS]
    if not methods:
        job.log("ERROR: No sparsification method chosen")
        return False
    metvae_args = [
        "--data", str(quant_table),
        "--save_path", str(metvae_out),
        "--seed", p.get("METVAE_SEED", "0"),
        "--feature_zero_threshold", p.get("METVAE_FEATURE_ZERO_THRESHOLD", "0.3"),
        "--learning_rate", p.get("METVAE_LEARNING_RATE", "0.01"),
        "--max_epochs", p.get("METVAE_MAX_EPOCHS", "1000"),
        "--num_sim", p.get("METVAE_NUM_SIM", "100"),
    ]
    outputs = p.get("METVAE_OUTPUTS", ",".join(OPTIONAL_OUTPUTS)).split(",")
    if "graphml" in outputs:
        metvae_args += ["--export_graphml", "--graphml_cutoffs", p.get("METVAE_GRAPHML_CUTOFF", "0.7")]
    skipped = [o for o in OPTIONAL_OUTPUTS if o != "graphml" and o not in outputs]
    if p.get("METVAE_LATENT_DIM"):
        metvae_args += ["--latent_dim", p["METVAE_LATENT_DIM"]]
    if p.get("METVAE_USE_GPU") == "1":
        metvae_args += ["--use_gpu"]
    if covariates:
        metvae_args += ["--meta", str(metadata_file)]
        if continuous:
            metvae_args += ["--continuous_covariate_keys", *continuous]
        if categorical:
            metvae_args += ["--categorical_covariate_keys", *categorical]

    job.log(f"Methods: {', '.join(methods)} · latent size: {p.get('METVAE_LATENT_DIM') or 'min(samples, features)'} · "
            f"learning rate: {p.get('METVAE_LEARNING_RATE', '0.01')} · GPU: {'requested' if p.get('METVAE_USE_GPU') == '1' else 'no'}")

    stages = [
        ("metvae_initialize_model", "initialize", [], skipped),
        ("metvae_train_model", "train", [], skipped),
        ("metvae_estimate_correlations", "correlate", [], skipped),
    ]
    for i, m in enumerate(methods):
        # Only the first method writes df_corr.csv (the p-value method clips |r| slightly)
        skip = skipped + (["corr"] if i > 0 and "corr" not in skipped else [])
        stages.append((SPARSE_STEPS[m], "sparsify",
                       ["--sparse_method", m, "--graphml_prefix", f"correlation_graph_{m}_cutoff"], skip))
    try:
        for step_name, stage, extra, skip in stages:
            ok = job.run_step(
                step_name,
                [python, "-m", STAGE_MODULE, "--stage", stage, *metvae_args, *extra,
                 *(["--skip_outputs", *skip] if skip else [])],
                cwd=LOCAL_RUNNER_DIR,
                timeout=METVAE_TIMEOUT,
            )
            if not ok:
                return False
    finally:
        (metvae_out / "metvae_model.pt").unlink(missing_ok=True)

    # Move key files to the output root for easy download
    graphml_files = sorted(metvae_out.glob("*.graphml"))
    for f in [metvae_out / "df_corr.csv", *sorted(metvae_out.glob("df_sparse_*.csv")), *graphml_files]:
        if f.exists():
            f.replace(out / f.name)
    if "graphml" in outputs:
        for m in methods:
            if not any(g.name.startswith(f"correlation_graph_{m}_") for g in graphml_files):
                job.log(f"WARNING: No {m} correlation reached the GraphML cutoff of "
                        f"{p.get('METVAE_GRAPHML_CUTOFF', '0.7')}, so no {m} network file was written. "
                        "Lower the cutoff to get one.")

    job.log("Key outputs:")
    for m in methods:
        job.log(f"  Sparse correlations:   {out / f'df_sparse_{m}.csv'}")
    if "corr" in outputs:
        job.log(f"  All correlations:      {out / 'df_corr.csv'}")
    for g in graphml_files:
        job.log(f"  GraphML network:       {out / g.name}")
    job.log(f"  Other MetVAE files:    {metvae_out}")
    return True


def _split_names(value: str) -> list:
    return [v.strip() for v in re.split(r",", value or "") if v.strip()]


def _find_input(directory: Path, patterns: list, exclude: str = None, exclude_also: list = None) -> Path | None:
    import fnmatch
    candidates = list(directory.iterdir())
    for pattern in patterns:
        for f in candidates:
            if not f.is_file():
                continue
            # Case-insensitive glob match
            if not fnmatch.fnmatch(f.name.lower(), pattern.lower()):
                continue
            if exclude and exclude.lower() in f.name.lower():
                continue
            if exclude_also:
                skip = False
                for xpat in exclude_also:
                    if fnmatch.fnmatch(f.name.lower(), xpat.lower()):
                        skip = True
                        break
                if skip:
                    continue
            return f
    return None
