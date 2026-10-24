"""Tests for tumour/normal mixing and the paired comparison."""
import numpy as np

from src.mixtures import mix, mixtures, paired_test


def test_mixing_happens_in_linear_space():
    tumour = np.log2(np.array([[1023.0, 1.0]]) + 1)   # linear 1023 and 1
    normal = np.log2(np.array([[1.0, 1023.0]]) + 1)
    assert np.allclose(mix(tumour, normal, 0.0), tumour)
    assert np.allclose(mix(tumour, normal, 1.0), normal)
    half = mix(tumour, normal, 0.5)
    assert np.allclose(np.exp2(half) - 1, [[512.0, 512.0]])   # averaged as counts, not as logs


def test_mixtures_keep_shape_and_stay_between_the_two_profiles():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 12, size=(50, 8)).astype(np.float32)
    normals = rng.uniform(0, 12, size=(5, 8)).astype(np.float32)
    out = mixtures(X, normals, rng, 0.1, 0.6)
    assert out.shape == X.shape
    lo = np.minimum(X[:, None, :], normals[None]).min(axis=1)
    hi = np.maximum(X[:, None, :], normals[None]).max(axis=1)
    assert np.all(out >= lo - 1e-4) and np.all(out <= hi + 1e-4)
    assert not np.allclose(out, X)


def test_paired_test_counts_disagreements():
    allowed = [{"A"}, {"A"}, {"B"}, {"B"}, {"C", "D"}]
    first = np.array(["A", "B", "B", "A", "C"])
    second = np.array(["A", "A", "A", "B", "D"])
    out = paired_test(first, second, allowed)
    assert out["only_first_correct"] == 1 and out["only_second_correct"] == 2
    assert 0 < out["p_value"] <= 1
