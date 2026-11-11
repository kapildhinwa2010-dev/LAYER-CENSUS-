import numpy as np
import pytest

from layer_census.census import census, effective_rank, layer_stats


def test_identity_layer():
    e = layer_stats(np.eye(8))
    assert e["shape"] == [8, 8]
    assert e["params"] == 64
    assert e["frobenius_norm"] == pytest.approx(8.0)
    assert e["effective_rank"] == pytest.approx(8.0)
    assert e["top1_variance_share"] == pytest.approx(1.0 / 8.0)
    assert e["top10_variance_share"] == pytest.approx(1.0)


def test_rank_one_layer():
    w = np.outer(np.ones(4), np.ones(4)) * 3.0
    e = layer_stats(w)
    assert e["effective_rank"] == pytest.approx(1.0)
    assert e["top1_variance_share"] == pytest.approx(1.0)
    assert e["frobenius_norm"] == pytest.approx(12.0)
    assert e["max_abs_weight"] == pytest.approx(3.0)


def test_zero_matrix():
    assert effective_rank(np.zeros((4, 6))) == 0.0
    e = layer_stats(np.zeros((4, 6)))
    assert e["top1_variance_share"] == 0.0
    assert e["top10_variance_share"] == 0.0


def test_dense_random_layer():
    rng = np.random.default_rng(0)
    e = layer_stats(rng.normal(size=(16, 32)))
    assert e["effective_rank"] > 10.0
    assert e["top1_variance_share"] < 0.2


def test_census_report():
    weights = [np.eye(8), np.outer(np.ones(4), np.ones(4)), np.eye(8) * 0.5]
    r = census(weights, names=["a", "b", "c"])
    assert r["num_layers"] == 3
    assert r["total_params"] == 64 + 16 + 64
    assert r["lowest_rank_layer"] == 1
    assert r["largest_layer"] == 0
    assert [e["name"] for e in r["layers"]] == ["a", "b", "c"]
    assert r["mean_effective_rank"] == pytest.approx((8.0 + 1.0 + 8.0) / 3.0)


def test_errors():
    with pytest.raises(ValueError):
        census([])
    with pytest.raises(ValueError):
        census([np.eye(4)], names=["a", "b"])
    with pytest.raises(ValueError):
        layer_stats(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        effective_rank(np.zeros(4))