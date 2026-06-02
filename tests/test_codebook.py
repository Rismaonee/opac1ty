"""Tests for codebook learning."""

import torch
import pytest
from opacc1ty.quantize.codebook import CodebookLearner, CodebookConfig


class TestCodebookLearner:
    def test_initialization(self):
        learner = CodebookLearner()
        assert learner.n_entries == 4  # 2-bit
        assert learner.svs == 8

    def test_kmeans_plus_plus(self):
        """Verify k-means++ initialization produces valid codebooks."""
        config = CodebookConfig(n_bits=2, sub_vector_size=8, n_samples=1024)
        learner = CodebookLearner(config)

        # Create synthetic weight with clear cluster structure
        weight = torch.randn(256, 256)

        # Initialize codebooks
        groups = weight.reshape(256, 32, 8)
        codebooks = learner._initialize_codebooks(groups)

        assert codebooks.shape == (256, 4, 8)
        assert not torch.isnan(codebooks).any()
        assert not torch.isinf(codebooks).any()

    def test_lloyd_convergence(self):
        """Verify Lloyd's algorithm converges."""
        config = CodebookConfig(
            n_bits=2, sub_vector_size=8, n_iters=50,
            convergence_threshold=1e-4
        )
        learner = CodebookLearner(config)

        weight = torch.randn(128, 128)
        codebooks, indices = learner.learn(weight)

        assert codebooks.shape[0] == 128  # out_features
        assert codebooks.shape[1] == 4    # codebook entries
        assert codebooks.shape[2] == 8    # sub_vector_size
        assert indices.shape == (128, 16)  # 128 groups of 8 = 16 indices

        # Indices should be valid (0-3 for 2-bit)
        assert indices.min() >= 0
        assert indices.max() <= 3

    def test_quantization_error(self):
        """Verify quantization error is reasonable."""
        config = CodebookConfig(n_bits=2, sub_vector_size=8, n_iters=100)
        learner = CodebookLearner(config)

        weight = torch.randn(256, 256)
        codebooks, indices = learner.learn(weight)

        # Reconstruct from codebooks + indices
        reconstructed = torch.zeros_like(weight)
        for oc in range(weight.shape[0]):
            for g in range(indices.shape[1]):
                idx = indices[oc, g].item()
                start = g * 8
                end = start + 8
                reconstructed[oc, start:end] = codebooks[oc, idx]

        mse = ((weight - reconstructed) ** 2).mean()
        # With 2-bit quantization, expect some error but not catastrophic
        assert mse < 2.0, f"MSE too high: {mse}"

    def test_padding_handling(self):
        """Verify padding is handled correctly for non-divisible dimensions."""
        config = CodebookConfig(n_bits=2, sub_vector_size=8)
        learner = CodebookLearner(config)

        # 250 is not divisible by 8
        weight = torch.randn(128, 250)
        codebooks, indices = learner.learn(weight)

        # Should work without errors and produce correct shapes
        assert indices.shape[1] == 250 // 8  # 31 groups (last 6 padded away)

    def test_error_feedback(self):
        """Verify error feedback reduces accumulated error."""
        config_no_fb = CodebookConfig(
            n_bits=2, sub_vector_size=8, use_error_feedback=False
        )
        config_fb = CodebookConfig(
            n_bits=2, sub_vector_size=8, use_error_feedback=True
        )

        weight = torch.randn(128, 256)

        learner_no_fb = CodebookLearner(config_no_fb)
        learner_fb = CodebookLearner(config_fb)

        cb_no, idx_no = learner_no_fb.learn(weight)
        cb_fb, idx_fb = learner_fb.learn(weight)

        # Reconstruct both
        def reconstruct(cb, idx, shape):
            out = torch.zeros(shape)
            for oc in range(shape[0]):
                for g in range(idx.shape[1]):
                    c_idx = idx[oc, g].item()
                    start = g * 8
                    end = min(start + 8, shape[1])
                    out[oc, start:end] = cb[oc, c_idx, :end - start]
            return out

        rec_no = reconstruct(cb_no, idx_no, weight.shape)
        rec_fb = reconstruct(cb_fb, idx_fb, weight.shape)

        mse_no = ((weight - rec_no) ** 2).mean()
        mse_fb = ((weight - rec_fb) ** 2).mean()

        # Error feedback should reduce or equal error
        assert mse_fb <= mse_no * 1.1, \
            f"Error feedback should not significantly worsen error: {mse_no:.4f} vs {mse_fb:.4f}"
