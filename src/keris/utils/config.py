from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def deep_set(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def parse_value(raw: str) -> Any:
    """
    Parse CLI override values safely.
    Examples:
      --set seed=42 -> int
      --set use_gpu=true -> bool
      --set foo=null -> None
      --set lr=5e-5 -> float
      --set tags='["a","b"]' -> list (YAML/JSON compatible)
    """
    # YAML is a convenient parser for scalars + lists/dicts
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set override: {item}. Use key=value.")
        k, v = item.split("=", 1)
        keys = k.strip().split(".")
        deep_set(cfg, keys, parse_value(v.strip()))
    return cfg


def build_argparser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. --set seed=42 --set data.input_dir=/path",
    )
    ap.add_argument("--out_dir", default="", help="Optional override for output directory.")
    return ap


def load_config_with_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides or [])
    if args.out_dir:
        cfg.setdefault("run", {})
        cfg["run"]["out_dir"] = args.out_dir
    return cfg
