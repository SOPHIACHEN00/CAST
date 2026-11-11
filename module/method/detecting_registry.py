"""Registry for the detecting methods supported by cadence v21."""

from dataclasses import dataclass
from typing import Type


QUANTITATIVE = "quantitative"
QUALITATIVE = "qualitative"


@dataclass(frozen=True)
class DetectingMethodSpec:
    name: str
    detecting_type: str
    implementation: Type[object]


def resolve_detecting_method(
    method_name: str,
    requested_type: str | None = None,
) -> DetectingMethodSpec:
    """Resolve and validate one of the detecting methods supported by v21."""
    normalized_name = method_name.lower()

    if normalized_name == "shapley":
        from module.method.quantitative_detecting.shapley_value import ShapleyValue

        spec = DetectingMethodSpec(
            name="shapley",
            detecting_type=QUANTITATIVE,
            implementation=ShapleyValue,
        )
    elif normalized_name == "sampling_shapley":
        from module.method.quantitative_detecting.sampling_shapley import (
            SamplingShapleyValue,
        )

        spec = DetectingMethodSpec(
            name="sampling_shapley",
            detecting_type=QUANTITATIVE,
            implementation=SamplingShapleyValue,
        )
    elif normalized_name == "sampling_least_core":
        from module.method.quantitative_detecting.sampling_least_core import (
            SamplingLeastCore,
        )

        spec = DetectingMethodSpec(
            name="sampling_least_core",
            detecting_type=QUANTITATIVE,
            implementation=SamplingLeastCore,
        )
    elif normalized_name == "eds":
        from module.method.qualitative_detecting.eds import EDSCalculator

        spec = DetectingMethodSpec(
            name="eds",
            detecting_type=QUALITATIVE,
            implementation=EDSCalculator,
        )
    elif normalized_name == "std_dagmm":
        from module.method.qualitative_detecting.std_dagmm import STDDAGMMDetector

        spec = DetectingMethodSpec(
            name="std_dagmm",
            detecting_type=QUALITATIVE,
            implementation=STDDAGMMDetector,
        )
    elif normalized_name == "delta_dagmm":
        from module.method.qualitative_detecting.delta_dagmm import DeltaDAGMMDetector

        spec = DetectingMethodSpec(
            name="delta_dagmm",
            detecting_type=QUALITATIVE,
            implementation=DeltaDAGMMDetector,
        )
    elif normalized_name == "wef":
        from module.method.qualitative_detecting.wef import WEFDetector

        spec = DetectingMethodSpec(
            name="wef",
            detecting_type=QUALITATIVE,
            implementation=WEFDetector,
        )
    elif normalized_name == "s2_wef":
        from module.method.qualitative_detecting.s2_wef import S2WEFDetector

        spec = DetectingMethodSpec(
            name="s2_wef",
            detecting_type=QUALITATIVE,
            implementation=S2WEFDetector,
        )
    elif normalized_name == "pass":
        from module.method.qualitative_detecting.pass_detector import PASSDetector

        spec = DetectingMethodSpec(
            name="pass",
            detecting_type=QUALITATIVE,
            implementation=PASSDetector,
        )
    elif normalized_name == "log_based_free_rider_detection":
        from module.method.qualitative_detecting.log_based_free_rider_detection import (
            LogBasedFreeRiderDetector,
        )

        spec = DetectingMethodSpec(
            name="log_based_free_rider_detection",
            detecting_type=QUALITATIVE,
            implementation=LogBasedFreeRiderDetector,
        )
    else:
        raise ValueError(
            f"Unsupported detecting method '{method_name}'. "
            "Cadence v21 supports only: shapley, sampling_shapley, "
            "sampling_least_core, eds, std_dagmm, delta_dagmm, wef, s2_wef, "
            "pass, log_based_free_rider_detection."
        )

    if requested_type is not None and requested_type.lower() != spec.detecting_type:
        raise ValueError(
            f"Detecting method '{spec.name}' is {spec.detecting_type}, "
            f"not {requested_type.lower()}."
        )
    return spec
