"""Build the list of Kenya reference points to fetch OlmoEarth embeddings for.

Run this LOCALLY (it reads the sibling GROW_AFRICA and ESAWC_RDM repos). Its
output, data/points/kenya_points.csv, is committed to this repo so the VM
only needs this repo - not the sibling ones.

Sources
-------
- GROW_AFRICA  data/training/grow_points_{long,short}_rains.csv
               One Acre Fund / Corbeels maize points. ewoc_code is set to
               1101060000 (WorldCereal pure maize), ref_id = "GROW_<Source>".
- ESAWC_RDM    data/processed/kenya_rdm_qa.csv
               In-Kenya, deduplicated ESA WorldCereal RDM samples (all classes,
               incl. the manually annotated non-cropland points), with their
               own sample_id / ewoc_code / ref_id / h3_l3_cell.

Embedding year
--------------
OlmoEarth v1.3 embeddings are annual (calendar year), 2017-2025 only.
  GROW   year of `time` (planting date, else harvest date)
  ESAWC  year of `valid_time`
Points outside 2017-2025 are dropped and counted in the log (notably all
2016 GROW points - there is no 2016 embedding).

Output columns
--------------
sample_id, dataset (GROW|ESAWC), ref_id, ewoc_code, lon, lat, year, h3_l3_cell

Run with the `geospatial` conda environment:
    conda run -n geospatial python src/01_build_point_list.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

YEAR_MIN, YEAR_MAX = 2017, 2025
GROW_EWOC = {"Maize": 1101060000}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIBLINGS = PROJECT_ROOT.parent
GROW_FILES = [SIBLINGS / "GROW_AFRICA" / "data" / "training" / f"grow_points_{s}.csv"
              for s in ("long_rains", "short_rains")]
ESAWC_FILE = SIBLINGS / "ESAWC_RDM" / "data" / "processed" / "kenya_rdm_qa.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "points" / "kenya_points.csv"

COLS = ["sample_id", "dataset", "ref_id", "ewoc_code", "lon", "lat", "year", "h3_l3_cell"]

logger = logging.getLogger("build_point_list")


def load_grow() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in GROW_FILES], ignore_index=True)
    unknown = set(df["crop-growpts"]) - set(GROW_EWOC)
    if unknown:
        raise ValueError(f"no ewoc_code mapping for GROW crops {unknown}")
    return pd.DataFrame({
        "sample_id": df["sample_id"],
        "dataset": "GROW",
        "ref_id": "GROW_" + df["Source"],
        "ewoc_code": df["crop-growpts"].map(GROW_EWOC).astype("int64"),
        "lon": df["longitude"],
        "lat": df["latitude"],
        "year": pd.to_datetime(df["time"]).dt.year,
        "h3_l3_cell": df["h3_l3_cell"],
    })


def load_esawc() -> pd.DataFrame:
    df = pd.read_csv(ESAWC_FILE, usecols=["sample_id", "ref_id", "ewoc_code", "lon", "lat", "valid_time", "h3_l3_cell"])
    df["year"] = pd.to_datetime(df["valid_time"], errors="coerce", utc=True).dt.year
    df["dataset"] = "ESAWC"
    df["ewoc_code"] = df["ewoc_code"].astype("int64")
    return df[COLS]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pts = pd.concat([load_grow(), load_esawc()], ignore_index=True)
    logger.info("Loaded %d points: %s", len(pts), pts["dataset"].value_counts().to_dict())

    if not pts["sample_id"].is_unique:
        raise ValueError("sample_id is not unique across GROW + ESAWC")

    in_range = pts["year"].between(YEAR_MIN, YEAR_MAX)
    dropped = pts[~in_range]
    logger.info("Dropping %d points outside %d-%d: %s", len(dropped), YEAR_MIN, YEAR_MAX,
                dropped.groupby(["dataset", dropped["year"].fillna(-1).astype(int)]).size().to_dict())
    pts = pts[in_range].copy()
    pts["year"] = pts["year"].astype(int)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pts[COLS].to_csv(OUTPUT_PATH, index=False)
    logger.info("Wrote %s (%d points) by dataset x year:\n%s", OUTPUT_PATH.relative_to(PROJECT_ROOT), len(pts),
                pts.groupby(["dataset", "year"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
