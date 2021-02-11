import os
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.legacy.native_amp import NativeAMPPlugin
from tests import _SKIPIF_NO_AMP, _SKIPIF_NO_GPUS
from tests.helpers.boring_model import BoringModel


@pytest.mark.skipif(**_SKIPIF_NO_AMP)
@mock.patch.dict(
    os.environ, {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0"
    }
)
@mock.patch('torch.cuda.device_count', return_value=2)
@pytest.mark.parametrize(
    ['ddp_backend', 'gpus', 'num_processes'],
    [('ddp_cpu', None, 2), ('ddp', 2, 0), ('ddp2', 2, 0), ('ddp_spawn', 2, 0)],
)
def test_amp_choice_default_ddp_cpu(tmpdir, ddp_backend, gpus, num_processes):

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.precision_connector.backend, NativeAMPPlugin)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        precision=16,
        amp_backend='native',
        gpus=gpus,
        num_processes=num_processes,
        accelerator=ddp_backend,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@pytest.mark.skipif(**_SKIPIF_NO_AMP)
@mock.patch.dict(
    os.environ, {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0"
    }
)
@mock.patch('torch.cuda.device_count', return_value=2)
@pytest.mark.parametrize(
    ['ddp_backend', 'gpus', 'num_processes'],
    [('ddp_cpu', None, 2), ('ddp', 2, 0), ('ddp2', 2, 0), ('ddp_spawn', 2, 0)],
)
def test_amp_choice_custom_ddp_cpu(tmpdir, ddp_backend, gpus, num_processes):

    class MyNativeAMP(NativeAMPPlugin):
        pass

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.precision_connector.backend, MyNativeAMP)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        precision=16,
        amp_backend='native',
        gpus=gpus,
        num_processes=num_processes,
        accelerator=ddp_backend,
        plugins=[MyNativeAMP()],
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


class GradientUnscaleBoringModel(BoringModel):

    def on_after_backward(self):
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
        if not (torch.isinf(norm) or torch.isnan(norm)):
            assert norm.item() < 15.


@pytest.mark.skipif(**(_SKIPIF_NO_GPUS + _SKIPIF_NO_AMP))
def test_amp_gradient_unscale(tmpdir):
    model = GradientUnscaleBoringModel()

    trainer = Trainer(
        max_epochs=2,
        default_root_dir=os.getcwd(),
        limit_train_batches=2,
        limit_test_batches=2,
        limit_val_batches=2,
        amp_backend='native',
        accelerator='ddp_spawn',
        gpus=2,
        precision=16,
        track_grad_norm=2,
        log_every_n_steps=1,
    )
    trainer.fit(model)


class UnscaleAccumulateGradBatchesBoringModel(BoringModel):

    def on_after_backward(self):
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
        if not (torch.isinf(norm) or torch.isnan(norm)):
            assert norm.item() < 15.


@pytest.mark.skipif(**(_SKIPIF_NO_GPUS + _SKIPIF_NO_AMP))
def test_amp_gradient_unscale_accumulate_grad_batches(tmpdir):
    model = UnscaleAccumulateGradBatchesBoringModel()

    trainer = Trainer(
        max_epochs=2,
        default_root_dir=os.getcwd(),
        limit_train_batches=2,
        limit_test_batches=2,
        limit_val_batches=2,
        amp_backend='native',
        accelerator='ddp_spawn',
        gpus=2,
        precision=16,
        track_grad_norm=2,
        log_every_n_steps=1,
        accumulate_grad_batches=2,
    )
    trainer.fit(model)
