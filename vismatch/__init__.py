"""Minimal vismatch package for dedode-lightglue only."""

from pathlib import Path
from types import ModuleType

from .utils import get_default_device  # noqa: F401
from .base_matcher import BaseMatcher  # noqa: F401

WEIGHTS_DIR = Path(__file__).parent.joinpath("model_weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

__version__ = "1.1.3"

available_models = ["dedode-lightglue"]


def get_version(pkg: ModuleType) -> tuple[int, int, int]:
    version_num = pkg.__version__.split("-")[0]
    major, minor, patch = [int(num) for num in version_num.split(".")]
    return major, minor, patch


def get_matcher(
    matcher_name: str | list[str] = "dedode-lightglue",
    device: str = "cpu",
    max_num_keypoints: int = 2048,  # kept for API compatibility
    *args,
    **kwargs,
) -> BaseMatcher:
    del max_num_keypoints  # dedode-lightglue wrapper does not use this argument
    if isinstance(matcher_name, list):
        raise RuntimeError("Only one matcher is supported in this minimal build: dedode-lightglue")

    if matcher_name != "dedode-lightglue":
        raise RuntimeError(
            f"Matcher '{matcher_name}' not supported in this minimal build. "
            "Available models: ['dedode-lightglue']"
        )

    from vismatch.im_models import kornia

    return kornia.DeDoDeLightGlue(device, *args, **kwargs)
