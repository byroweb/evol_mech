"""Body registry: which genome, operators and rollout a run uses.

A run config picks the body with `"body": "biped"` (default, the crank-driven
4-bar biped) or `"body": "skater"` (docs/skater_design.md). The GA, run store
and video renderer only ever go through this namespace, so the two bodies
share the evolution stack without sharing a genome.
"""
from __future__ import annotations

from types import SimpleNamespace

BODIES = ("biped", "skater")


def get_body(name: str = "biped") -> SimpleNamespace:
    if name == "biped":
        import importlib
        from .genome import spec, ops, archetype
        ro = importlib.import_module("evol_mech.sim.rollout")   # the package re-exports the function under the same name
        return SimpleNamespace(
            name="biped", PARAMS=spec.PARAMS, INDEX=spec.INDEX, N_PARAMS=spec.N_PARAMS,
            DEFAULT_NORM=spec.DEFAULT_NORM, decode=spec.decode, encode=spec.encode,
            random_genome=ops.random_genome, mutate=ops.mutate, crossover=ops.crossover,
            archetype_center=archetype.archetype_center, load_archetype=archetype.load_archetype,
            rollout=ro.rollout, evaluate=ro.evaluate)
    if name == "skater":
        from . import skater as sk
        return SimpleNamespace(
            name="skater", PARAMS=sk.PARAMS, INDEX=sk.INDEX, N_PARAMS=sk.N_PARAMS,
            DEFAULT_NORM=sk.DEFAULT_NORM, decode=sk.decode, encode=sk.encode,
            random_genome=sk.random_genome, mutate=sk.mutate, crossover=sk.crossover,
            archetype_center=sk.archetype_center, load_archetype=sk.load_archetype,
            rollout=sk.rollout, evaluate=sk.evaluate)
    raise ValueError(f"unknown body {name!r}; choose from {BODIES}")
