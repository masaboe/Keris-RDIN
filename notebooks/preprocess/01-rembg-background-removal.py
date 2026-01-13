#!/usr/bin/env python3
"""
Background removal utility using rembg.

- Reads all .png/.jpg/.jpeg from an input folder (optionally recursive)
- Removes background via rembg
- Saves PNG outputs (with alpha channel) to an output folder
- Optional preview visualization (original vs. cutout)

Example:
  python scripts/remove_background.py --input data/images --output outputs/no_background --show

Notes:
- This script does NOT install packages at runtime. Use requirements.txt for reproducibility.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from rembg import remove
from PIL import Image, ImageOps

# Matplotlib is optional (only required if --show is used)
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class Config:
    input_dir: Path
    output_dir: Path
    recursive: bool = False
    overwrite: bool = False
    show: bool = False
    save: bool = True


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("remove_background")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def iter_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for p in input_dir.glob(pattern):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def safe_output_path(output_dir: Path, input_path: Path) -> Path:
    # Always save outputs as PNG to preserve alpha
    return output_dir / f"{input_path.stem}.png"


def remove_bg_pil(img: Image.Image) -> Image.Image:
    """
    Run rembg on a PIL image and return an RGBA image.
    rembg.remove() can accept PIL Image directly.
    """
    out = remove(img)
    if isinstance(out, Image.Image):
        return out.convert("RGBA")
    # Fallback: if rembg returns bytes for some reason
    from io import BytesIO

    return Image.open(BytesIO(out)).convert("RGBA")


def visualize(original: Image.Image, cutout: Image.Image, title: str = "") -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available. Install it or run without --show.")

    plt.figure(figsize=(10, 5))
    plt.suptitle(title, fontsize=12)

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Background Removed")
    plt.imshow(cutout)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def process_one(
    img_path: Path,
    cfg: Config,
    logger: logging.Logger,
) -> Optional[Path]:
    try:
        # Ensure correct orientation for JPEGs with EXIF rotation
        original = Image.open(img_path)
        original = ImageOps.exif_transpose(original).convert("RGBA")

        out_path = safe_output_path(cfg.output_dir, img_path)

        if out_path.exists() and not cfg.overwrite and cfg.save:
            logger.info(f"Skip (exists): {out_path.name}")
            return out_path

        cutout = remove_bg_pil(original)

        if cfg.save:
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            cutout.save(out_path, format="PNG")
            logger.info(f"Saved: {out_path}")

        if cfg.show:
            visualize(original, cutout, title=img_path.name)

        return out_path

    except Exception as e:
        logger.error(f"Failed: {img_path.name} | {e}")
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remove background from images using rembg.")
    p.add_argument("--input", required=True, help="Input folder containing images.")
    p.add_argument("--output", default="outputs/no_background", help="Output folder for PNG results.")
    p.add_argument("--recursive", action="store_true", help="Recursively search for images.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--no-save", action="store_true", help="Do not save outputs (use with --show).")
    p.add_argument("--show", action="store_true", help="Show original vs output (requires matplotlib).")
    p.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)

    cfg = Config(
        input_dir=Path(args.input).expanduser().resolve(),
        output_dir=Path(args.output).expanduser().resolve(),
        recursive=bool(args.recursive),
        overwrite=bool(args.overwrite),
        show=bool(args.show),
        save=not bool(args.no_save),
    )

    if not cfg.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {cfg.input_dir}")

    images = list(iter_images(cfg.input_dir, cfg.recursive))
    if not images:
        logger.warning(f"No images found in: {cfg.input_dir}")
        return

    logger.info(f"Found {len(images)} image(s). Processing...")
    ok = 0
    for img_path in images:
        if process_one(img_path, cfg, logger) is not None:
            ok += 1

    logger.info(f"Done. Success: {ok}/{len(images)}")


if __name__ == "__main__":
    main()
