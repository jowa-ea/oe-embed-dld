"""Extract OlmoEarth v1.3 embeddings at point locations (run on the VM).

Reads the public OlmoEarth v1.3 annual embeddings Zarr store
(https://source.coop/olmoearth/olmoearth-v1-3-embeddings) directly with HTTP
range requests - no zarr/obstore needed. For every point it fetches only the
compressed inner chunks that contain its pixels.

Store layout (verified against embeddings.zarr/utmNN/embeddings/zarr.json)
--------------------------------------------------------------------------
- one group per UTM zone (utm01..utm60), northern-hemisphere CRS EPSG:326NN
  (southern points have negative y), 10 m pixels,
  transform x = 10*col, y = 9502720 - 10*row
- array `embeddings` (time, band, y, x) int8, time = 2017..2025 (index year-2017)
- shards of (1, 128, 2048, 2048), key c/<t>/0/<shard_row>/<shard_col>,
  each holding 2 x 8 x 8 zstd-compressed inner chunks of (1, 64, 256, 256);
  the shard index (2*8*8 little-endian uint64 (offset, nbytes) pairs + 4-byte
  crc32c) is the last 2052 bytes of the shard. offset = nbytes = 2**64-1
  marks an empty inner chunk.
- -128 = nodata. Dequantize: v = (|q| / 127.5)**2 * sign(q)

Why so much data moves: a zstd chunk can't be range-read below chunk level,
so each pixel costs a whole ~3.4 MB (64-band) chunk. Run this in AWS
us-west-2 (where the bucket lives) so that traffic stays in-region.

Steps
-----
1. Expand each point to its --window x --window pixel neighbourhood
   (default 3x3 = ~30 m, to absorb GPS / plot-centre error).
2. Group pixels by (zone, year, shard). Each shard is one task: fetch its
   index, fetch the needed inner chunks (bands 0-63, plus 64-127 unless
   --dims 64), pull out the pixels, write a part file. Tasks run in a thread
   pool; finished parts are skipped on rerun, so the job is resumable.
3. Merge parts: per point, the centre-pixel vector and the mean of the valid
   (non-nodata) window pixels, dequantized to float32.

Outputs
-------
data/embeddings/parts_d<dims>_w<window>/utm<zone>_y<year>_r<row>_c<col>.parquet  per-shard raw int8 pixels (resume state)
data/embeddings/kenya_points_olmoearth_v1_3_d<dims>_w<window>.parquet   one row per point:
    sample_id, year, zone, row, col, n_window, n_valid,
    emb_center_0..D-1, emb_mean_0..D-1   (float32, NaN where nodata)
data/embeddings/extraction_log_d<dims>_w<window>.csv   per-shard bytes / status

Usage
-----
    python src/02_extract_embeddings.py --limit 50          # quick test
    python src/02_extract_embeddings.py                     # full run
    python src/02_extract_embeddings.py --dims 64           # bands 0-63 only (half the download)
"""

from __future__ import annotations

import argparse
import csv
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from pyproj import Transformer

ENDPOINTS = {
    # Direct S3 (path-style, anonymous). Fastest from AWS us-west-2.
    "s3": "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr",
    # Source Cooperative's CDN proxy. Use outside AWS if s3 is slow.
    "sourcecoop": "https://data.source.coop/olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr",
}
FIRST_YEAR, LAST_YEAR = 2017, 2025
Y_TOP = 9_502_720
PIXEL = 10
SHARD_PX = 2048
INNER_PX = 256
INNER_BANDS = 64
N_INNER = SHARD_PX // INNER_PX  # 8
INDEX_BYTES = 2 * N_INNER * N_INNER * 16 + 4  # 2052
EMPTY = np.uint64(2**64 - 1)
NODATA = -128
CHUNK_BYTES = INNER_BANDS * INNER_PX * INNER_PX

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POINTS = PROJECT_ROOT / "data" / "points" / "kenya_points.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "embeddings"
OUTPUT_STEM = "kenya_points_olmoearth_v1_3"

logger = logging.getLogger("extract_embeddings")
_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4))
        _local.s = s
    return _local.s


def get_range(url: str, rng: str, retries: int = 6) -> bytes | None:
    """GET a byte range; None on 404 (shard absent = all nodata)."""
    for attempt in range(retries):
        try:
            r = session().get(url, headers={"Range": f"bytes={rng}"}, timeout=120)
            if r.status_code == 404:
                return None
            if r.status_code in (200, 206):
                return r.content
            msg = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            msg = repr(e)
        wait = min(60, 2 ** attempt)
        logger.warning("%s [%s] failed (%s), retry in %ds", url, rng, msg, wait)
        time.sleep(wait)
    raise RuntimeError(f"giving up on {url} [{rng}]")


def expand_pixels(points: pd.DataFrame, window: int) -> pd.DataFrame:
    """One row per (point, window pixel) with zone / shard / inner-chunk / in-chunk indices."""
    points = points.copy()
    points["zone"] = ((points["lon"] + 180) // 6).astype(int) + 1
    rows, cols = np.empty(len(points), dtype=np.int64), np.empty(len(points), dtype=np.int64)
    for zone, idx in points.groupby("zone").groups.items():
        x, y = Transformer.from_crs(4326, 32600 + zone, always_xy=True).transform(
            points.loc[idx, "lon"].to_numpy(), points.loc[idx, "lat"].to_numpy())
        pos = points.index.get_indexer(idx)
        cols[pos] = np.floor(np.asarray(x) / PIXEL).astype(np.int64)
        rows[pos] = np.floor((Y_TOP - np.asarray(y)) / PIXEL).astype(np.int64)
    points["row"], points["col"] = rows, cols

    h = window // 2
    offs = [(dy, dx) for dy in range(-h, h + 1) for dx in range(-h, h + 1)]
    px = pd.concat([points[["sample_id", "zone", "year", "row", "col"]].assign(dy=dy, dx=dx) for dy, dx in offs],
                   ignore_index=True)
    px["prow"], px["pcol"] = px["row"] + px["dy"], px["col"] + px["dx"]
    px["sr"], px["sc"] = px["prow"] // SHARD_PX, px["pcol"] // SHARD_PX
    px["ir"], px["ic"] = (px["prow"] % SHARD_PX) // INNER_PX, (px["pcol"] % SHARD_PX) // INNER_PX
    px["jr"], px["jc"] = px["prow"] % INNER_PX, px["pcol"] % INNER_PX
    return points, px


def part_path(parts_dir: Path, zone: int, year: int, sr: int, sc: int) -> Path:
    return parts_dir / f"utm{zone:02d}_y{year}_r{sr}_c{sc}.parquet"


def process_shard(base: str, key: tuple, px: pd.DataFrame, n_band_chunks: int, out: Path) -> dict:
    zone, year, sr, sc = key
    url = f"{base}/utm{zone:02d}/embeddings/c/{year - FIRST_YEAR}/0/{sr}/{sc}"
    vals = np.full((len(px), n_band_chunks * INNER_BANDS), NODATA, dtype=np.int8)
    nbytes, status = 0, "ok"

    raw_index = get_range(url, f"-{INDEX_BYTES}")
    if raw_index is None:
        status = "shard_missing"
    else:
        if len(raw_index) != INDEX_BYTES:
            raise RuntimeError(f"{url}: index is {len(raw_index)} bytes, expected {INDEX_BYTES}")
        nbytes += len(raw_index)
        index = np.frombuffer(raw_index[:-4], dtype="<u8").reshape(2, N_INNER, N_INNER, 2)
        ir, ic, jr, jc = (px[c].to_numpy() for c in ("ir", "ic", "jr", "jc"))
        for (cr, cc) in sorted(set(zip(ir.tolist(), ic.tolist()))):
            sel = (ir == cr) & (ic == cc)
            for bc in range(n_band_chunks):
                off, n = index[bc, cr, cc]
                if off == EMPTY and n == EMPTY:
                    continue  # empty inner chunk -> stays nodata
                raw = get_range(url, f"{int(off)}-{int(off) + int(n) - 1}")
                nbytes += len(raw)
                chunk = np.frombuffer(
                    pa.decompress(raw, decompressed_size=CHUNK_BYTES, codec="zstd", asbytes=True), dtype=np.int8
                ).reshape(INNER_BANDS, INNER_PX, INNER_PX)
                vals[sel, bc * INNER_BANDS:(bc + 1) * INNER_BANDS] = chunk[:, jr[sel], jc[sel]].T

    table = pa.table({
        "sample_id": px["sample_id"].to_numpy(),
        "dy": px["dy"].to_numpy(np.int8),
        "dx": px["dx"].to_numpy(np.int8),
        "q": pa.FixedSizeListArray.from_arrays(pa.array(vals.ravel(), type=pa.int8()), vals.shape[1]),
    })
    tmp = out.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.replace(out)
    return {"shard": out.stem, "pixels": len(px), "bytes": nbytes, "status": status}


def dequantize(q: np.ndarray) -> np.ndarray:
    v = np.where(q == NODATA, np.nan, q).astype(np.float32)
    return (np.abs(v) / 127.5) ** 2.0 * np.sign(v)


def merge(points: pd.DataFrame, parts: list[Path], dims: int, out_path: Path) -> None:
    tables = [pq.read_table(p) for p in parts]
    t = pa.concat_tables(tables)
    q = np.asarray(t.column("q").combine_chunks().flatten()).reshape(len(t), -1)[:, :dims]
    sid = t.column("sample_id").to_numpy(zero_copy_only=False)
    is_center = (t.column("dy").to_numpy() == 0) & (t.column("dx").to_numpy() == 0)
    v = dequantize(q)
    valid = ~np.isnan(v).any(axis=1)

    order = pd.Index(points["sample_id"])
    pos = order.get_indexer(sid)
    n = len(points)
    sums = np.zeros((n, dims), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    totals = np.bincount(pos, minlength=n)
    np.add.at(sums, pos[valid], v[valid])
    np.add.at(counts, pos[valid], 1)
    with np.errstate(invalid="ignore"):
        mean = (sums / counts[:, None]).astype(np.float32)
    center = np.full((n, dims), np.nan, dtype=np.float32)
    center[pos[is_center]] = v[is_center]

    df = points[["sample_id", "year", "zone", "row", "col"]].reset_index(drop=True)
    df["n_window"], df["n_valid"] = totals, counts
    emb = pd.DataFrame(np.hstack([center, mean]),
                       columns=[f"emb_center_{i}" for i in range(dims)] + [f"emb_mean_{i}" for i in range(dims)])
    pd.concat([df, emb], axis=1).to_parquet(out_path, index=False)
    logger.info("Wrote %s: %d points, %d with a valid centre pixel, %d with >=1 valid window pixel",
                out_path, n, int((~np.isnan(center).any(axis=1)).sum()), int((counts > 0).sum()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dims", type=int, choices=[64, 128], default=128,
                   help="128 = full embedding; 64 = Matryoshka bands 0-63 only (half the download)")
    p.add_argument("--window", type=int, choices=[1, 3, 5], default=3, help="pixel neighbourhood size")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--endpoint", choices=list(ENDPOINTS), default="s3")
    p.add_argument("--limit", type=int, help="only the first N points (testing)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    points = pd.read_csv(args.points)
    if args.limit:
        points = points.head(args.limit)
    bad = ~points["year"].between(FIRST_YEAR, LAST_YEAR)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} points have years outside {FIRST_YEAR}-{LAST_YEAR}")
    points, px = expand_pixels(points.reset_index(drop=True), args.window)

    tag = f"d{args.dims}_w{args.window}" + (f"_limit{args.limit}" if args.limit else "")
    parts_dir = args.out_dir / f"parts_{tag}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    groups = px.groupby(["zone", "year", "sr", "sc"])
    todo = [(k, g) for k, g in groups if not part_path(parts_dir, *k).exists()]
    logger.info("%d points -> %d pixels in %d shards (%d already done); endpoint=%s dims=%d window=%d",
                len(points), len(px), groups.ngroups, groups.ngroups - len(todo), args.endpoint, args.dims, args.window)

    log_path = args.out_dir / f"extraction_log_{tag}.csv"
    new_log = not log_path.exists()
    n_band_chunks = args.dims // INNER_BANDS
    t0, done_bytes = time.time(), 0
    with log_path.open("a", newline="") as fh, ThreadPoolExecutor(args.workers) as ex:
        w = csv.DictWriter(fh, fieldnames=["shard", "pixels", "bytes", "status"])
        if new_log:
            w.writeheader()
        futs = [ex.submit(process_shard, ENDPOINTS[args.endpoint], k, g.reset_index(drop=True), n_band_chunks,
                          part_path(parts_dir, *k)) for k, g in todo]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            w.writerow(rec)
            fh.flush()
            done_bytes += rec["bytes"]
            if i % 25 == 0 or i == len(futs):
                el = time.time() - t0
                logger.info("%d/%d shards, %.2f GB in %.0fs (%.0f MB/s)", i, len(futs), done_bytes / 1e9, el,
                            done_bytes / 1e6 / max(el, 1e-9))

    parts = [part_path(parts_dir, *k) for k in groups.groups]
    merge(points, parts, args.dims, args.out_dir / f"{OUTPUT_STEM}_{tag}.parquet")


if __name__ == "__main__":
    main()
