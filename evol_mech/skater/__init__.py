"""Skater mech (docs/skater_design.md): tread-footed, five-joint legs, fans
vectored in pitch and roll. Selected by `"body": "skater"` in a run config."""
from .spec import PARAMS, NAMES, INDEX, GROUPS, N_PARAMS, DEFAULT_NORM, decode, encode
from .genome import random_genome, mutate, crossover, load_archetype, archetype_center, default_archetype
from .structure import InvalidBody, structure_from_params, is_buildable, nominal_height
from .mjcf import build_mjcf, nominal_qpos, ACTUATOR_NAMES, SIM_DT
from .controller import SkaterController
from .rollout import rollout, evaluate, make_model, make_data

__all__ = ["PARAMS", "NAMES", "INDEX", "GROUPS", "N_PARAMS", "DEFAULT_NORM", "decode", "encode",
           "random_genome", "mutate", "crossover", "load_archetype", "archetype_center",
           "default_archetype", "InvalidBody", "structure_from_params", "is_buildable",
           "nominal_height", "build_mjcf", "nominal_qpos", "ACTUATOR_NAMES", "SIM_DT",
           "SkaterController", "rollout", "evaluate", "make_model", "make_data"]
