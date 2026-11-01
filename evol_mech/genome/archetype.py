"""Archetype files: a hand-designed starting body saved by tools/linkage_editor.html.

Format:
    {"name": "...", "tags": {"crank_l": "leg input crank L", ...},
     "phys": {"crank_len": 0.035, ...}}        # physical units, any subset of PARAMS

The GA seeds its initial population around the archetype instead of the
built-in defaults when a config sets "archetype": "<path>".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .spec import PARAMS, INDEX, LO, HI, encode, decode, DEFAULT_NORM


def load_archetype(path: str | Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    phys = d.get("phys", {})
    unknown = sorted(set(phys) - set(INDEX))
    if unknown:
        raise ValueError(f"archetype {path}: unknown parameters {unknown}")
    for k, v in phys.items():
        i = INDEX[k]
        if not (LO[i] - 1e-9 <= v <= HI[i] + 1e-9):
            raise ValueError(f"archetype {path}: {k}={v} outside genome bounds [{LO[i]}, {HI[i]}]")
    d["phys"] = phys
    return d


def archetype_center(path: str | Path | None) -> np.ndarray:
    """Normalised genome to centre the initial population on."""
    if not path:
        return DEFAULT_NORM.copy()
    return encode(load_archetype(path)["phys"])


def default_archetype(name: str = "default") -> dict:
    return {"name": name, "tags": {}, "phys": decode(DEFAULT_NORM)}
