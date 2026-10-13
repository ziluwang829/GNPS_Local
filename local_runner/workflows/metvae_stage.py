"""
Run one stage of a MetVAE analysis, so GNPS Local can time each stage as its own step.

Stages (the steps timed in MetVAE's quickstart notebook):
  initialize  -> read the table, MetVAE(...)
  train       -> model.train()
  correlate   -> model.get_corr()
  sparsify    -> model.sparse_by_p() or model.sparse_by_sec(), plus the GraphML export

Takes metvae-cli's flags plus --stage and --skip_outputs; the model is kept in
<save_path>/metvae_model.pt between stages.

Usage (from local_runner/, as a module so workflows/metvae.py doesn't shadow MetVAE):
  python -m workflows.metvae_stage --stage initialize --data table.csv --save_path out [metvae-cli flags]
"""

import argparse
import csv
import os
import sys

import pandas as pd
import torch

try:
    from metvae import MetVAE
    from metvae.cli import build_parser
except ImportError:
    sys.exit("ERROR: could not import the MetVAE package. Run this as "
             "`python -m workflows.metvae_stage` from local_runner/, not as a file.")

STAGES = ("initialize", "train", "correlate", "sparsify")
MODEL_FILE = "metvae_model.pt"
SKIPPABLE = ("corr", "pvalues", "checkpoint")

PEAK_AREA = " Peak area"
# Columns (or sample file names as headers) that mean features are rows: a raw tool export
FEATURE_PROPERTY_COLUMNS = {
    "row m/z", "m/z", "mz", "average mz", "mass", "neutral mass", "pepmass",
    "row retention time", "retention time", "rt", "average rt(min)",
    "formula", "molecular_formula", "compound", "compound name", "adduct", "charge", "cas", "cas id",
    "mzmed", "mzmin", "mzmax", "rtmed", "rtmin", "rtmax", "npeaks",
}
SAMPLE_FILE_SUFFIXES = (".mzml", ".mzxml", ".raw", ".d", ".cdf", ".wiff")


def parse_args(argv=None):
    stage_parser = argparse.ArgumentParser(add_help=False)
    stage_parser.add_argument("--stage", required=True, choices=STAGES)
    stage_parser.add_argument("--skip_outputs", nargs="*", choices=SKIPPABLE, default=[])
    stage_args, rest = stage_parser.parse_known_args(argv)

    parser = build_parser()
    parser.set_defaults(latent_dim=None)   # MetVAE then uses min(samples, features)
    args = parser.parse_args(rest)
    if args.meta is None and (args.continuous_covariate_keys is not None
                              or args.categorical_covariate_keys is not None):
        parser.error("--continuous_covariate_keys and --categorical_covariate_keys require --meta")
    args.stage = stage_args.stage
    args.skip_outputs = set(stage_args.skip_outputs)
    return args


def _model_path(args):
    return os.path.join(args.save_path, MODEL_FILE)


def _save_model(model, args):
    torch.save(model, _model_path(args))


def _load_model(args):
    path = _model_path(args)
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found; run the earlier MetVAE stages first.")
    return torch.load(path, weights_only=False)


def _read_table(path):
    sep = "\t" if path.lower().endswith(".tsv") else ","
    try:
        try:
            encoding = "utf-8"
            df = pd.read_csv(path, sep=sep, encoding=encoding)
        except UnicodeDecodeError:
            encoding = "latin-1"
            df = pd.read_csv(path, sep=sep, encoding=encoding)
        if df.shape[1] == 1:   # another separator, e.g. semicolons
            df = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as e:
        sys.exit(f"ERROR: Could not read {os.path.basename(path)} as a CSV or tab-separated table "
                 f"({' '.join(str(e).split())}).")
    return df


def _row_id_table_to_samples(df):
    area_cols = [c for c in df.columns if str(c).endswith(PEAK_AREA)]
    data = df.set_index("row ID")[area_cols]
    try:
        data.index = [str(int(float(i))) for i in data.index]
    except ValueError:
        data.index = [str(i) for i in data.index]
    if data.index.duplicated().any():
        sys.exit("ERROR: The table has duplicate 'row ID' values.")
    data.columns = [str(c)[: -len(PEAK_AREA)] for c in area_cols]
    # Convert after transposing: same memory layout as a CSV read, so results match metvae-cli's
    return data.T.apply(pd.to_numeric, errors="coerce")


def _named_rows(df, features_as_rows):
    """First column = names, every other column numeric."""
    if not features_as_rows:
        has_properties = any(str(c).strip().lower() in FEATURE_PROPERTY_COLUMNS for c in df.columns)
        file_cols = sum(str(c).strip().lower().endswith(SAMPLE_FILE_SUFFIXES) for c in df.columns)
        if has_properties or file_cols >= 2:
            sys.exit("ERROR: This table is not supported.")
    data = df.set_index(df.columns[0])
    data = data.drop(columns=[c for c in data.columns if str(c).startswith("Unnamed") and data[c].isna().all()])
    data.index = [str(i) for i in data.index]
    numeric = data.apply(pd.to_numeric, errors="coerce")
    if any(numeric[c].isna().all() and data[c].notna().any() for c in data.columns):
        sys.exit("ERROR: This table is not supported.")
    return numeric.T if features_as_rows else numeric


def read_data(path, features_as_rows=False):
    df = _read_table(path)
    if "row ID" in df.columns and any(str(c).endswith(PEAK_AREA) for c in df.columns):
        print("Detected features as rows with 'row ID' and 'Peak area' columns (FBMN's layout)")
        data = _row_id_table_to_samples(df)
    else:
        print("Detected a table with features as rows" if features_as_rows else "Detected a samples x features table")
        data = _named_rows(df, features_as_rows)
    if data.index.duplicated().any():
        sys.exit("ERROR: Some sample names appear more than once.")
    if data.shape[1] < 3:
        sys.exit(f"ERROR: Found {data.shape[1]} feature(s); MetVAE needs at least 3.")
    if data.shape[0] < 2:
        sys.exit(f"ERROR: Found {data.shape[0]} sample(s); MetVAE needs at least 2.")
    zero_share = float((data.fillna(0) == 0).to_numpy().mean())
    print(f"Read {data.shape[0]} samples x {data.shape[1]} features ({zero_share:.0%} zeros)")
    return data


def read_metadata(path, samples, covariates):
    """Covariates per sample, matched on the 'filename' column (else the first)."""
    meta = _read_table(path)
    meta.columns = [str(c).strip() for c in meta.columns]
    name_col = next((c for c in meta.columns if c.lower() == "filename"), meta.columns[0])
    meta[name_col] = meta[name_col].astype(str).str.strip()
    meta = meta.set_index(name_col)
    missing_cols = [c for c in covariates if c not in meta.columns]
    if missing_cols:
        sys.exit(
            f"ERROR: Covariate column(s) not in the metadata: {', '.join(missing_cols)}. "
            f"Available columns: {', '.join(map(str, meta.columns))}"
        )
    missing = [s for s in samples if s not in meta.index]
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        sys.exit(
            f"ERROR: {len(missing)} sample(s) have no row in the metadata '{name_col}' column: {shown}. "
            "Metadata file names must match the sample names in the table exactly."
        )
    if meta.index.duplicated().any():
        sys.exit(f"ERROR: The metadata has duplicate values in its '{name_col}' column.")
    print(f"Read metadata for {len(samples)} samples; covariates: {', '.join(covariates) or 'none'}")
    return meta.loc[samples, covariates]


def initialize(args):
    feature_zero_threshold = None if args.no_feature_filter else args.feature_zero_threshold
    data = read_data(args.data, args.features_as_rows)
    meta = None
    if args.meta is not None:
        covariates = (args.continuous_covariate_keys or []) + (args.categorical_covariate_keys or [])
        meta = read_metadata(args.meta, list(data.index), covariates)
    activation = None if args.activation in ("linear", "none") else args.activation
    os.makedirs(args.save_path, exist_ok=True)
    model = MetVAE(
        data=data,
        features_as_rows=False,
        meta=meta,
        continuous_covariate_keys=args.continuous_covariate_keys,
        categorical_covariate_keys=args.categorical_covariate_keys,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        activation=activation,
        use_gpu=args.use_gpu,
        logging=args.logging,
        feature_zero_threshold=feature_zero_threshold,
        sample_zero_threshold=args.sample_zero_threshold,
        seed=args.seed,
    )
    _save_model(model, args)


def train(args):
    model = _load_model(args)
    model.train(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        max_grad_norm=None if (args.max_grad_norm is None or args.max_grad_norm < 0) else args.max_grad_norm,
        shuffle=True,
        deterministic=args.deterministic,
    )
    if "checkpoint" not in args.skip_outputs:
        optimizer = getattr(model, "optimizer", None)
        ckpt = {
            "model_state_dict": model.model.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "train_loss": getattr(model, "train_loss", []),
            "epoch": getattr(model, "current_epoch", None),
            "trained_epochs": len(getattr(model, "train_loss", [])),
            "learning_rate": args.learning_rate,
        }
        torch.save(ckpt, os.path.join(args.save_path, "model_state.pth"))
    _save_model(model, args)


def correlate(args):
    model = _load_model(args)
    model.get_corr(
        num_sim=args.num_sim,
        workers=args.workers,
        batch_size=args.impute_batch_size,
        threshold=args.threshold,
        seed=args.seed,
    )
    _save_model(model, args)


def sparsify(args):
    model = _load_model(args)
    out = args.save_path
    skip = args.skip_outputs
    if args.sparse_method == "pval":
        filt = model.sparse_by_p(p_adj_method=args.p_adj_method, cutoff=args.cutoff)
        if "corr" not in skip:
            filt["estimate"].to_csv(os.path.join(out, "df_corr.csv"))
        if "pvalues" not in skip:
            filt["p_value"].to_csv(os.path.join(out, "p_values.csv"))
            filt["q_value"].to_csv(os.path.join(out, "q_values.csv"))
        filt["sparse_estimate"].to_csv(os.path.join(out, "df_sparse_pval.csv"))
    else:
        rho_val = None if args.rho is None or args.rho < 0 else float(args.rho)
        restart_val = None if (args.sec_restart is None or args.sec_restart < 0) else int(args.sec_restart)
        filt = model.sparse_by_sec(
            rho=rho_val,
            epsilon=args.sec_epsilon,
            tol=args.sec_tol,
            max_iter=args.sec_max_iter,
            restart=restart_val,
            line_search_apg=not args.no_line_search_apg,
            delta=args.sec_delta,
            n_samples=None,
            c_delta=args.sec_c_delta,
            threshold=args.sec_threshold,
            c_grid=args.c_grid,
            n_splits=args.n_splits,
            seed=args.seed,
            workers=args.sec_workers,
            refine=(not args.no_refine),
            refine_points=args.refine_points,
        )
        if "corr" not in skip:
            filt["estimate"].to_csv(os.path.join(out, "df_corr.csv"))
        filt["sparse_estimate"].to_csv(os.path.join(out, "df_sparse_sec.csv"))
        if filt.get("best_rho") is not None:
            with open(os.path.join(out, "sec_selected.txt"), "w") as f:
                f.write(f"best_rho={filt['best_rho']}\n")
        if filt.get("scores_by_rho") is not None:
            filt["scores_by_rho"].to_csv(os.path.join(out, "sec_scores.csv"), index=False)

    if args.export_graphml:
        model.export_graphml(
            sparse_df=filt["sparse_estimate"],
            cutoffs=args.graphml_cutoffs,
            output_dir=out,
            file_prefix=args.graphml_prefix,
        )


def main(argv=None):
    args = parse_args(argv)
    {"initialize": initialize, "train": train, "correlate": correlate, "sparsify": sparsify}[args.stage](args)


if __name__ == "__main__":
    main()
