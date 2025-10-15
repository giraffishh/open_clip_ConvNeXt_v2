#!/usr/bin/env python
"""
One-stop MS COCO Captions builder for OpenCLIP (2017 by default).

This script will:
  1) Download COCO train/val images and annotations (2017 or 2014)
  2) Verify MD5 checksums (when available)
  3) Extract archives idempotently (with per-zip markers)
  4) Build TSVs (filepath\ttitle) for train/val captions with absolute paths
  5) Print ready-to-run training commands (PowerShell + bash)

Outputs (for --year 2017):
  <root>/coco/images/train2017/
  <root>/coco/images/val2017/
  <root>/coco/annotations/captions_train2017.json
  <root>/coco/annotations/captions_val2017.json
  <root>/coco/tsv/train2017.tsv
  <root>/coco/tsv/val2017.tsv

Usage:
  python scripts/download_and_build_coco_captions.py --root /datasets --year 2017

Options:
  --skip-download            Only (re)build TSVs from existing files
  --force                    Redownload and re-extract (overwrite)
  --max-caps-per-image N     Emit up to N captions per image (default: 5)

Dependencies:
  pip install pandas requests tqdm
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

# lazy import for pandas to avoid editor lint errors when not installed
pd = None

try:
    import requests
except Exception:
    requests = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


COCO_URLS = {
    2017: {
        'train_images': (
            'http://images.cocodataset.org/zips/train2017.zip',
            'cced6f7f71b7629ddf16f17bbcfab6b2'
        ),
        'val_images': (
            'http://images.cocodataset.org/zips/val2017.zip',
            '442b8da7639aecaf257c1dceb8ba8c80'
        ),
        'annotations': (
            'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
            'f4bbac642086de4f52a3fdda2de5fa2c'
        )
    },
    2014: {
        'train_images': (
            'http://images.cocodataset.org/zips/train2014.zip',
            '0da8c0bd3d6becc4dcb32757491aca88'
        ),
        'val_images': (
            'http://images.cocodataset.org/zips/val2014.zip',
            'a3d79f5ed8d289b7a7554ce06a5782b3'
        ),
        'annotations': (
            'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
            '0d12afc5d6c8c27b19d6baf5d7f5a0e4'
        )
    }
}


def _require(dep, name):
    if dep is None:
        raise RuntimeError(f"{name} is required. Please install it via: pip install {name}")


def _md5(file_path: Path) -> str:
    h = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_md5: str = None, force: bool = False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if expected_md5 is None:
            print(f"[skip] exists: {dest}")
            return
        try:
            if _md5(dest) == expected_md5:
                print(f"[skip] exists and checksum ok: {dest}")
                return
            else:
                print(f"[warn] checksum mismatch, re-downloading: {dest}")
                dest.unlink(missing_ok=True)
        except Exception:
            print(f"[warn] failed to checksum, re-downloading: {dest}")
            dest.unlink(missing_ok=True)

    _require(requests, 'requests')
    _require(tqdm, 'tqdm')

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('Content-Length', 0))
        with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    if expected_md5:
        got = _md5(dest)
        if got != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {dest}: expected {expected_md5}, got {got}")


def extract_zip(zip_path: Path, out_dir: Path, marker_root: Path, force: bool = False):
    marker = marker_root / f".extract_done_{zip_path.stem}"
    if marker.exists() and not force:
        print(f"[skip] extracted: {zip_path} -> {out_dir}")
        return
    print(f"[extract] {zip_path} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    marker.touch()


def build_tsv(ann_path: Path, img_dir: Path, out_tsv: Path, max_caps_per_image: int = 5, absolute_paths: bool = True):
    global pd
    if pd is None:
        try:
            import pandas as pd  # type: ignore
        except Exception:
            pd = None
    _require(pd, 'pandas')
    with open(ann_path, 'r', encoding='utf-8') as f:
        ann = json.load(f)

    id_to_file = {img['id']: img['file_name'] for img in ann.get('images', [])}
    caps_per_image = {}
    for c in ann.get('annotations', []):
        img_id = c['image_id']
        caption = (c.get('caption') or '').strip()
        if not caption:
            continue
        caps_per_image.setdefault(img_id, []).append(caption)

    rows = []
    img_root = img_dir.resolve() if absolute_paths else img_dir
    for img_id, caps in caps_per_image.items():
        fn = id_to_file.get(img_id)
        if not fn:
            continue
        fpath = (img_root / fn)
        # use POSIX-style paths for broader compatibility
        fpath_str = fpath.as_posix() if hasattr(fpath, 'as_posix') else str(fpath)
        for cap in caps[: max_caps_per_image]:
            rows.append({'filepath': fpath_str, 'title': cap})

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_tsv, sep='\t', index=False)
    print(f"[ok] TSV written: {out_tsv} (rows={len(rows)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True, help='Root folder to place COCO data (will create <root>/coco)')
    ap.add_argument('--year', type=int, default=2017, choices=[2014, 2017], help='COCO year (default: 2017)')
    ap.add_argument('--skip-download', action='store_true', help='Skip downloading; only build TSVs from existing files')
    ap.add_argument('--force', action='store_true', help='Force redownload and re-extract/rewrite')
    ap.add_argument('--max-caps-per-image', type=int, default=5, help='Max captions per image to emit into TSVs')
    args = ap.parse_args()

    urls = COCO_URLS[args.year]
    root = args.root
    coco_root = root / 'coco'
    images_dir = coco_root / 'images'
    ann_dir = coco_root / 'annotations'
    tsv_dir = coco_root / 'tsv'

    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Archives
    train_zip = coco_root / f"train{args.year}.zip"
    val_zip = coco_root / f"val{args.year}.zip"
    ann_zip = coco_root / f"annotations_trainval{args.year}.zip"

    if not args.skip_download:
        download(urls['train_images'][0], train_zip, urls['train_images'][1], force=args.force)
        download(urls['val_images'][0], val_zip, urls['val_images'][1], force=args.force)
        download(urls['annotations'][0], ann_zip, urls['annotations'][1], force=args.force)

        extract_zip(train_zip, images_dir, coco_root, force=args.force)
        extract_zip(val_zip, images_dir, coco_root, force=args.force)
        # extract annotations into coco_root so that <root>/coco/annotations/... exists
        extract_zip(ann_zip, coco_root, coco_root, force=args.force)

    # Build TSV
    train_img_dir = images_dir / f"train{args.year}"
    val_img_dir = images_dir / f"val{args.year}"
    train_ann = coco_root / f"annotations/captions_train{args.year}.json"
    val_ann = coco_root / f"annotations/captions_val{args.year}.json"

    if not train_img_dir.exists() or not val_img_dir.exists() or not train_ann.exists() or not val_ann.exists():
        raise FileNotFoundError(
            "Missing extracted files. Expected:\n"
            f"  train images: {train_img_dir}\n"
            f"  val images:   {val_img_dir}\n"
            f"  train ann:    {train_ann}\n"
            f"  val ann:      {val_ann}"
        )

    tsv_dir.mkdir(parents=True, exist_ok=True)
    train_tsv = tsv_dir / f"train{args.year}.tsv"
    val_tsv = tsv_dir / f"val{args.year}.tsv"
    build_tsv(train_ann, train_img_dir, train_tsv, args.max_caps_per_image)
    build_tsv(val_ann, val_img_dir, val_tsv, args.max_caps_per_image)

    # Print training command examples
    print("\nAll set. Example training commands:\n")
    # PowerShell
    ps_root = coco_root.as_posix()
    print("# PowerShell (Windows):")
    print(
        "torchrun --nproc_per_node=2 -m open_clip_train.main "
        f"--dataset-type csv --train-data {train_tsv.as_posix()} --val-data {val_tsv.as_posix()} "
        "--csv-separator \"`t\" --csv-img-key filepath --csv-caption-key title\n"
    )
    # bash
    print("# bash (Linux/macOS):")
    print(
        "torchrun --nproc_per_node=2 -m open_clip_train.main "
        f"--dataset-type csv --train-data {train_tsv.as_posix()} --val-data {val_tsv.as_posix()} "
        "--csv-separator $'\\t' --csv-img-key filepath --csv-caption-key title\n"
    )
    print("Done.")


if __name__ == '__main__':
    main()
