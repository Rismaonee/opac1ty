"""Pytest configuration for Opacc1ty tests."""

import pytest
import torch
import warnings


@pytest.fixture(autouse=True)
def suppress_torch_warnings():
    """Suppress known PyTorch warnings during tests."""
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")


@pytest.fixture
def random_weights():
    """Generate random weight matrices for testing."""
    def _make(out_features=256, in_features=256):
        return torch.randn(out_features, in_features)
    return _make


@pytest.fixture
def tiny_model_state_dict():
    """A minimal state dict simulating a tiny transformer."""
    return {
        "model.embed_tokens.weight": torch.randn(1000, 128),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(128, 128),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(128, 128),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(128, 128),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(256, 128),
        "model.layers.0.mlp.up_proj.weight": torch.randn(256, 128),
        "model.layers.0.mlp.down_proj.weight": torch.randn(128, 256),
        "model.layers.0.input_layernorm.weight": torch.randn(128),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(128),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(128, 128),
        "model.layers.1.self_attn.k_proj.weight": torch.randn(128, 128),
        "model.layers.1.self_attn.v_proj.weight": torch.randn(128, 128),
        "model.layers.1.self_attn.o_proj.weight": torch.randn(128, 128),
        "model.layers.1.mlp.gate_proj.weight": torch.randn(256, 128),
        "model.layers.1.mlp.up_proj.weight": torch.randn(256, 128),
        "model.layers.1.mlp.down_proj.weight": torch.randn(128, 256),
        "model.layers.1.input_layernorm.weight": torch.randn(128),
        "model.layers.1.post_attention_layernorm.weight": torch.randn(128),
        "model.norm.weight": torch.randn(128),
        "lm_head.weight": torch.randn(1000, 128),
    }
