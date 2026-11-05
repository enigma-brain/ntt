"""CPU tests for RankMe (smooth effective rank) in litmodule.py.

``NuclrContrastiveLitModule._compute_rankme`` is a ``@staticmethod`` taking an
``(N, dim)`` tensor and returning ``exp(H)`` where ``H`` is the Shannon entropy
of the L1-normalized singular values. RankMe ranges in [1, min(N, dim)].

These characterize the metric on known-rank inputs; no LightningModule is
instantiated (the static method only uses its tensor arg).
"""

import types
from unittest import mock

import torch

from ntt.contrib.nuclr import litmodule
from ntt.contrib.nuclr.litmodule import NuclrContrastiveLitModule

rankme = NuclrContrastiveLitModule._compute_rankme


def test_rankme_is_static_and_callable() -> None:
    """Sanity: callable without an instance (it is a @staticmethod)."""
    x = torch.eye(4)
    assert isinstance(rankme(x), float)


def test_rankme_rank_one_matrix_is_one() -> None:
    """A rank-1 matrix (outer product; all rows collinear) has a single nonzero
    singular value -> entropy 0 -> RankMe == 1."""
    torch.manual_seed(0)
    u = torch.randn(16, 1)
    v = torch.randn(1, 8)
    x = u @ v  # (16, 8), rank 1
    assert rankme(x) == torch.tensor(1.0).item() or abs(rankme(x) - 1.0) < 1e-4


def test_rankme_orthonormal_block_equals_dimension() -> None:
    """Equal-energy orthogonal directions -> uniform singular-value distribution
    over ``d`` dimensions -> RankMe ~= d."""
    d = 5
    # Rows are an orthonormal basis scaled equally: all singular values equal.
    x = torch.eye(d) * 3.0
    assert abs(rankme(x) - d) < 1e-4

    # A taller equal-energy matrix: stack two scaled identities -> still d
    # distinct directions, each singular value equal in magnitude.
    x2 = torch.cat([torch.eye(d), torch.eye(d)], dim=0) * 2.0
    assert abs(rankme(x2) - d) < 1e-4


def test_rankme_uniform_singular_values_via_orthonormal_columns() -> None:
    """Use a random orthonormal matrix (Q from QR) scaled uniformly: all
    singular values are exactly equal, so RankMe == number of columns."""
    torch.manual_seed(1)
    d = 6
    a = torch.randn(10, d)
    q, _ = torch.linalg.qr(a)  # (10, d) with orthonormal columns
    assert abs(rankme(q * 5.0) - d) < 1e-4


def test_rankme_near_rank_one_much_smaller_than_full_rank() -> None:
    """Monotonic sanity: a near-rank-1 matrix yields RankMe close to 1 and far
    below that of a full-rank random matrix of the same shape."""
    torch.manual_seed(2)
    n, d = 64, 16

    # Near rank-1: dominant outer product plus tiny isotropic noise.
    u = torch.randn(n, 1)
    v = torch.randn(1, d)
    near_rank1 = u @ v + 1e-3 * torch.randn(n, d)

    full = torch.randn(n, d)

    r_low = rankme(near_rank1)
    r_high = rankme(full)

    assert r_low < 2.0  # essentially collapsed
    assert r_high > 5.0  # many active dimensions
    assert r_high > 3.0 * r_low


def test_rankme_bounded_by_min_dim() -> None:
    """RankMe never exceeds min(N, dim) for any input."""
    torch.manual_seed(3)
    x = torch.randn(7, 12)
    assert rankme(x) <= min(x.shape) + 1e-4


# --------------------------------------------------------------------------- #
# _gather_embeddings — pool per-rank shards before computing RankMe under DDP.
# RankMe is nonlinear in the singular values, so it must run on the pooled
# matrix, not be averaged across ranks. These exercise the helper's logic on
# CPU (the no-op path and a mocked 2-rank gather) without a real process group.
# --------------------------------------------------------------------------- #
def test_gather_embeddings_is_noop_without_distributed() -> None:
    """With no process group, the local embeddings ARE the full matrix."""
    x = torch.randn(5, 8)
    out = NuclrContrastiveLitModule._gather_embeddings(x)
    assert out is x or torch.equal(out, x)


def test_gather_embeddings_concatenates_rank_shards() -> None:
    """With world_size>1, shards from all ranks are concatenated into the
    pooled matrix (variable per-rank row counts allowed)."""
    local = torch.arange(6.0).reshape(3, 2)  # this rank: 3 rows
    other = torch.arange(100.0, 104.0).reshape(2, 2)  # other rank: 2 rows

    def fake_all_gather_object(object_list, obj):
        object_list[0] = obj  # this rank's shard (a CPU copy of `local`)
        object_list[1] = other

    fake_dist = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 2,
        all_gather_object=fake_all_gather_object,
    )
    with mock.patch.object(litmodule, "dist", fake_dist):
        out = NuclrContrastiveLitModule._gather_embeddings(local)

    assert out.shape == (5, 2)
    assert torch.equal(out, torch.cat([local, other], dim=0))
