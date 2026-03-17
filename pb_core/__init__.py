"""Reusable PB core package for generic experiments."""

from .controller import (
    as_bt,
    strip_t,
    boxtimes_timewise,
    NominalPlant,
    WIntegralAugmenter,
    OperatorBase,
    GenericOperator,
    TimewiseMatVec,
    FactorizedOperator,
    PBState,
    PBController,
)
from .interfaces import (
    BatchData,
    ContextBuilder,
    DatasetProvider,
    LossFn,
    MetricsFn,
    NoiseModel,
    TruePlant,
)
from .factories import FactorizedBuildSpec, build_factorized_controller, infer_dims_from_probe
from .noise import DecayingGaussianNoise, ZeroNoise
from .registry import Registry
from .rollout import RolloutResult, rollout_pb
from .runner import PBExperimentRunner, RunnerConfig
from .validation import validate_component_compatibility

__all__ = [
    # controller core
    "as_bt",
    "WIntegralAugmenter",
    "strip_t",
    "boxtimes_timewise",
    "NominalPlant",
    "OperatorBase",
    "GenericOperator",
    "TimewiseMatVec",
    "FactorizedOperator",
    "PBState",
    "PBController",
    # interfaces
    "BatchData",
    "ContextBuilder",
    "DatasetProvider",
    "LossFn",
    "MetricsFn",
    "NoiseModel",
    "TruePlant",
    # factories
    "FactorizedBuildSpec",
    "build_factorized_controller",
    "infer_dims_from_probe",
    # noise
    "DecayingGaussianNoise",
    "ZeroNoise",
    # registry
    "Registry",
    # rollout
    "RolloutResult",
    "rollout_pb",
    # runner
    "PBExperimentRunner",
    "RunnerConfig",
    # validation
    "validate_component_compatibility",
]
