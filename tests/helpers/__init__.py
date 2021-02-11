import torch

from pytorch_lightning.utilities import (
    _APEX_AVAILABLE,
    _NATIVE_AMP_AVAILABLE,
    _TORCH_LOWER_EQUAL_1_4,
    _TORCH_QUANTIZE_AVAILABLE,
)
from tests.helpers.boring_model import BoringDataModule, BoringModel, RandomDataset  # noqa: F401
from tests.helpers.skipif import SkipIf  # noqa: F401

_SKIPIF_PT_LE_1_4 = SkipIf(condition=_TORCH_LOWER_EQUAL_1_4, reason="test pytorch > 1.4")
_SKIPIF_NO_GPU = SkipIf(condition=not torch.cuda.is_available(), reason="test requires any GPU machine")
_SKIPIF_NO_GPUS = SkipIf(condition=torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
_SKIPIF_NO_AMP = SkipIf(condition=not _NATIVE_AMP_AVAILABLE, reason="test requires native AMP")
_SKIPIF_NO_APEX = SkipIf(condition=not _APEX_AVAILABLE, reason="test requires APEX")
_SKIPIF_NO_PT_QUANT = SkipIf(
    condition=not _TORCH_QUANTIZE_AVAILABLE, reason="PyTorch quantization is needed for this test"
)