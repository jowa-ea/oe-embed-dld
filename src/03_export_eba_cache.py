"""Write the extracted embeddings as an EBA_detector embeddings cache (DuckDB).

EBA_detector (https://github.com/WorldCereal/EBA_detector) scores samples from
a DuckDB table `embeddings_cache` and never recomputes embeddings itself. Its
loader reads `sample_id, ewoc_code, model_hash, ref_id, h3_l3_cell, lat, lon`
plus every `embedding_*` column (dimension auto-detected), so OlmoEarth
vectors can be dropped in directly in place of its default Presto ones.

Steps
-----
1. Join the extracted embeddings (02) to the point list (01) on sample_id.
2. Pick the vector: --vector mean (default, 3x3 window mean) or center.
   Optionally truncate to the first 64 dims (--dims 64; Matryoshka).
3. Drop points with no valid embedding (nodata) and write the table.

Then run EBA_detector on it, e.g. with label_domain="ewoc_code" (raw codes;
GROW maize points are ewoc_code 1101060000, ref_id "GROW_<Source>"):

    from EBA_detector.anomaly import run_pipeline
    flagged, summary = run_pipeline(embeddings_db_path="data/eba/embeddings_cache.duckdb", ...)

Usage
-----
    python src/03_export_eba_cache.py --embeddings data/embeddings/kenya_points_olmoearth_v1_3_d128_w3.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POINTS = PROJECT_ROOT / "data" / "points" / "kenya_points.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "eba" / "embeddings_cache.duckdb"

logger = logging.getLogger("export_eba_cache")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings", type=Path, required=True, help="parquet written by 02_extract_embeddings.py")
    p.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--vector", choices=["mean", "center"], default="mean")
    p.add_argument("--dims", type=int, choices=[64, 128], help="truncate to the first N dims (default: all extracted)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    emb = pd.read_parquet(args.embeddings)
    src_cols = [c for c in emb.columns if c.startswith(f"emb_{args.vector}_")]
    dims = args.dims or len(src_cols)
    if dims > len(src_cols):
        raise ValueError(f"--dims {dims} but the file only has {len(src_cols)} dims")
    src_cols = src_cols[:dims]

    pts = pd.read_csv(args.points)
    df = pts.merge(emb[["sample_id"] + src_cols], on="sample_id", how="inner")
    vec = df[src_cols].to_numpy(np.float32)
    ok = ~np.isnan(vec).any(axis=1)
    logger.info("%d points joined; dropping %d without a valid %s embedding", len(df), int((~ok).sum()), args.vector)
    df = df[ok].reset_index(drop=True)

    model_hash = f"olmoearth_v1_3_d{dims}_{args.vector}_{args.embeddings.stem.rsplit('_', 1)[-1]}"
    out = pd.DataFrame({
        "sample_id": df["sample_id"],
        "model_hash": model_hash,
        "ref_id": df["ref_id"],
        "ewoc_code": df["ewoc_code"].astype("int64"),
        "h3_l3_cell": df["h3_l3_cell"],
        "lat": df["lat"],
        "lon": df["lon"],
        "dataset": df["dataset"],
        "year": df["year"],
    })
    out = pd.concat([out, pd.DataFrame(df[src_cols].to_numpy(np.float32),
                                       columns=[f"embedding_{i}" for i in range(dims)])], axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(args.out)) as con:
        con.register("out_df", out)
        con.execute("CREATE OR REPLACE TABLE embeddings_cache AS SELECT * FROM out_df")
        n = con.execute("SELECT count(*) FROM embeddings_cache").fetchone()[0]
    logger.info("Wrote %s: table embeddings_cache, %d rows, %d dims, model_hash=%s; by dataset %s",
                args.out, n, dims, model_hash, out["dataset"].value_counts().to_dict())


if __name__ == "__main__":
    main()
