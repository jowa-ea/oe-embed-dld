# Kenya reference points -> OlmoEarth v1.3 embeddings

Fetches [OlmoEarth v1.3 annual embeddings](https://source.coop/olmoearth/olmoearth-v1-3-embeddings)
(Ai2; 10 m, 128-d, 2017-2025) at the Kenya GROW-Africa and ESA WorldCereal
RDM reference points, and writes them in the DuckDB format that
[EBA_detector](https://github.com/WorldCereal/EBA_detector) reads, for
embedding-based label cleaning.

The point list is already in the repo (`data/points/kenya_points.csv`), so
the VM needs **only this repo** - not GROW_AFRICA or ESAWC_RDM.

## Why a VM

The store can't be read below one zstd chunk (256 x 256 px x 64 bands, ~3.4
MB), so each 128-byte point vector costs a ~3.4 MB download per 64 bands.
For the 68,259 points that is **~53 GB at 128 dims or ~26 GB at 64 dims**
(measured from the shard indexes), touching 1,295 shards in UTM zones 36 and
37. The bucket is in **AWS us-west-2**; a VM there keeps that traffic
in-region and only the result (~70 MB) has to come back.

## On the VM

Tested with Python 3.13; needs only requests, numpy, pandas, pyarrow, pyproj
(+ duckdb for step 03). No zarr/obstore - the script reads the Zarr v3 shards
with plain HTTP range requests and pyarrow's zstd codec.

```bash
git clone <this repo> && cd OLMOEARTH_EMBEDDINGS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. smoke test (20 points, ~70 MB)
python src/02_extract_embeddings.py --limit 20

# 2. full run (resumable - just rerun if interrupted)
python src/02_extract_embeddings.py                 # 128 dims, ~53 GB transfer
#   or: python src/02_extract_embeddings.py --dims 64   # ~26 GB

# 3. EBA_detector input
python src/03_export_eba_cache.py --embeddings data/embeddings/kenya_points_olmoearth_v1_3_d128_w3.parquet
```

Bring back `data/embeddings/kenya_points_olmoearth_v1_3_d128_w3.parquet` and/or
`data/eba/embeddings_cache.duckdb` (scp, or `git add -f` - both are
gitignored by default). The `parts_*` directories are only resume state.

Options for `02`: `--workers` (default 32; it is network-bound, so a small
2-4 vCPU instance is fine), `--endpoint s3|sourcecoop` (`s3` = direct bucket,
default; `sourcecoop` = CDN proxy, better outside AWS), `--window 1|3|5`.
Progress lines report GB transferred and MB/s.

## Pipeline

| Script | Where | Input | Output |
|---|---|---|---|
| `01_build_point_list.py` | **local** (needs sibling repos) | `../GROW_AFRICA/data/training/grow_points_*.csv`, `../ESAWC_RDM/data/processed/kenya_rdm_qa.csv` | `data/points/kenya_points.csv` (committed) |
| `02_extract_embeddings.py` | VM | `data/points/kenya_points.csv` | `data/embeddings/kenya_points_olmoearth_v1_3_d<dims>_w<window>.parquet` |
| `03_export_eba_cache.py` | anywhere | output of 02 + point list | `data/eba/embeddings_cache.duckdb` |

### Point list (01)

| dataset | points | years | labels |
|---|---|---|---|
| GROW | 17,729 | 2017-2020 | all maize -> `ewoc_code` 1101060000, `ref_id` = `GROW_<Source>` |
| ESAWC | 50,530 | 2017-2023 | RDM `ewoc_code` / `ref_id` as-is (all classes) |

Embedding year = GROW planting (else harvest) year / ESAWC `valid_time` year.
Dropped: 3,498 GROW points from 2015-2016 (no embeddings before 2017) and 91
ESAWC points without a `valid_time`. `sample_id` for GROW is
`GROW_Point_<row in GROW-Africa_Point.xlsx>`; ESAWC keeps its RDM ids.

### Extraction (02)

- Each point is expanded to a 3x3 pixel window (~30 m) to absorb GPS /
  plot-centre error. Output has both `emb_center_*` (the point's pixel) and
  `emb_mean_*` (mean of valid window pixels), dequantized to float32, plus
  `n_valid` window pixels. NaN = nodata (no imagery that year).
- Dequantization follows the dataset README: `(|q|/127.5)**2 * sign(q)`, -128 = nodata.
- Verified locally: vectors match an independent single-pixel decode
  exactly, via both endpoints; norms 3.0-3.5 (README: ~3.0-3.8); 64-dim run
  == first 64 dims of the 128-dim run.

### EBA cache (03)

Table `embeddings_cache`: `sample_id, model_hash, ref_id, ewoc_code,
h3_l3_cell, lat, lon, dataset, year, embedding_0..D-1` - the columns
EBA_detector's loader selects (it auto-detects D). `--vector mean|center`,
`--dims 64` to truncate.

## Caveats for the label cleaning

- GROW points are all maize. Score them **together with ESAWC** (same H3
  cells, multi-class) so EBA's kNN label-purity signal means something; on
  GROW alone it only finds "atypical maize".
- Embeddings are annual: they cannot check long- vs short-rains labels, and
  EBA cannot clean the GROW yield values.

## Licences

OlmoEarth embeddings: OlmoEarth Artifact License (see the dataset's
LICENSE.txt); cite "OlmoEarth v1.3 Embeddings", Allen Institute for AI, 2026.
GROW-Africa: CC-BY-4.0, Geyman et al. 2025, doi:10.5281/zenodo.14961637.
ESA WorldCereal RDM: per-collection licences (see ESAWC_RDM).
