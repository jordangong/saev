import hypothesis.strategies as st
import torch
from hypothesis import given, settings

from .. import config
from . import modeling


def test_factories():
    assert isinstance(modeling.get_activation(config.Relu()), torch.nn.ReLU)


@st.composite
def relu_cfgs(draw):
    d_vit = draw(st.sampled_from([32, 64, 128]))
    exp = draw(st.sampled_from([2, 4]))
    return config.Relu(d_vit=d_vit, exp_factor=exp)


@settings(deadline=None)
@given(cfg=relu_cfgs(), batch=st.integers(min_value=1, max_value=4))
def test_sae_shapes(cfg, batch):
    sae = modeling.SparseAutoencoder(cfg)
    x = torch.randn(batch, cfg.d_vit)
    x_hat, f = sae(x)
    assert x_hat.shape == (batch, cfg.d_vit)
    assert f.shape == (batch, cfg.d_sae)
