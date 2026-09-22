import numpy as np
import pytest

from eval_framework.metrics.loglikelihood.bpb_variants_estimators import (
    COMMON_BINS,
    PrefixItem,
    bin_mass,
    bpb_at_nstar,
    continuation_rate,
    corpus_bpb,
    cumulative_cost,
    effective_length_bpb,
    entry_cost,
    ls_bpb,
    make_prefix_items,
    marginal_rate_curve,
    mean_of_ratios,
    ols_affine,
    ols_leverage_weights,
    pbpb,
    per_byte_cost,
    prefix_curve,
    prefix_decompose,
    prior_bpb,
    resolve_nstar,
    summarize_all,
)


def test_constant_length_all_equal():
    bits = np.array([10.0, 20.0, 30.0])
    n = np.array([5.0, 5.0, 5.0])
    expected = bits.mean() / 5.0
    assert mean_of_ratios(bits, n) == pytest.approx(expected)
    assert corpus_bpb(bits, n) == pytest.approx(expected)
    s = summarize_all(bits, n)
    assert s["bpb_at_nstar_ols"]["mean"]["value"] == pytest.approx(expected)
    assert s["ls_bpb_task_q"]["value"] == pytest.approx(expected)


def test_nstar_mean_equals_corpus():
    rng = np.random.default_rng(0)
    n = rng.uniform(4, 40, size=200)
    bits = 12.0 + 0.8 * n + rng.normal(0, 0.5, size=200)
    nstar = resolve_nstar(n, "mean")
    got = bpb_at_nstar(bits, n, nstar, "ols")["value"]
    assert got == pytest.approx(corpus_bpb(bits, n), rel=1e-10)


def test_nstar_inf_equals_beta():
    rng = np.random.default_rng(1)
    n = rng.uniform(4, 40, size=200)
    bits = 12.0 + 0.8 * n + rng.normal(0, 0.01, size=200)
    fit = ols_affine(n, bits)
    got = bpb_at_nstar(bits, n, float("inf"), "ols")["value"]
    assert got == pytest.approx(fit.slope, rel=1e-8)


def test_synthetic_alpha_beta_recovery():
    n = np.linspace(5, 50, 100)
    bits = 10.0 + 0.5 * n
    fit = ols_affine(n, bits)
    assert fit.intercept == pytest.approx(10.0, abs=1e-8)
    assert fit.slope == pytest.approx(0.5, abs=1e-8)


def test_ls_bpb_weights_nonnegative_sum_to_one():
    rng = np.random.default_rng(2)
    n = rng.uniform(2, 80, size=300)
    bits = 8 + 0.4 * n
    out = ls_bpb(bits, n)
    w = np.array(out["weights"])
    assert np.all(w >= -1e-12)
    assert w.sum() == pytest.approx(1.0)
    assert out["n_negative_weights"] == 0


def test_bin_mass_sums_to_one_and_matches_internal_histogram():
    rng = np.random.default_rng(7)
    n = rng.uniform(1, 900, size=400)
    mass = bin_mass(n, COMMON_BINS)
    assert mass.sum() == pytest.approx(1.0)
    assert np.all(mass >= 0)
    # q_mass matches bin_mass
    out = ls_bpb(np.ones_like(n), n, bins=COMMON_BINS)
    assert np.allclose(out["q_mass"], mass)


def test_q_mass_override_equals_matching_reference_sample():
    """Explicit q_mass matches bin_mass from a reference sample."""
    rng = np.random.default_rng(11)
    n = rng.uniform(1, 900, size=500)
    bits = 30 + 0.5 * n + rng.normal(0, 5, size=500)
    ref = rng.uniform(1, 900, size=800)
    from_sample = ls_bpb(bits, n, q_nbytes=ref, bins=COMMON_BINS)
    from_mass = ls_bpb(bits, n, bins=COMMON_BINS, q_mass=bin_mass(ref, COMMON_BINS))
    assert from_mass["value"] == pytest.approx(from_sample["value"])


def test_equal_task_weighting_is_insensitive_to_item_counts():
    """Equal task weighting does not grow with item count per task."""
    rng = np.random.default_rng(13)
    short = rng.uniform(1, 20, size=100)
    long_small = rng.uniform(400, 900, size=100)
    long_huge = rng.uniform(400, 900, size=5000)

    small_pool = np.mean([bin_mass(short, COMMON_BINS), bin_mass(long_small, COMMON_BINS)], axis=0)
    huge_pool = np.mean([bin_mass(short, COMMON_BINS), bin_mass(long_huge, COMMON_BINS)], axis=0)
    assert np.allclose(small_pool, huge_pool, atol=0.05)

    # Item-count pooling follows the oversampled task.
    count_pool = bin_mass(np.concatenate([short, long_huge]), COMMON_BINS)
    assert np.abs(count_pool - small_pool).max() > 0.3


def test_q_mass_rejects_bad_input():
    n = np.array([5.0, 50.0, 500.0])
    bits = np.array([10.0, 40.0, 300.0])
    with pytest.raises(ValueError):
        ls_bpb(bits, n, bins=COMMON_BINS, q_mass=np.ones(2))
    with pytest.raises(ValueError):
        ls_bpb(bits, n, bins=COMMON_BINS, q_mass=np.array([-1.0, 1.0, 1.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        ls_bpb(bits, n, bins=COMMON_BINS, q_mass=np.zeros(5))


def test_improving_item_cannot_increase_corpus_or_ls():
    bits = np.array([10.0, 20.0, 30.0, 40.0])
    n = np.array([4.0, 8.0, 12.0, 20.0])
    c0 = corpus_bpb(bits, n)
    ls0 = ls_bpb(bits, n)["value"]
    better = bits.copy()
    better[0] -= 5.0
    assert corpus_bpb(better, n) < c0
    assert ls_bpb(better, n)["value"] <= ls0 + 1e-12


def test_negative_leverage_outside_range():
    n = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    h = ols_leverage_weights(n, n_star=100.0)
    assert np.any(h < 0)


def test_effective_length_undefined_when_beta_nonpositive():
    n = np.array([4.0, 8.0, 16.0, 32.0])
    bits = np.array([40.0, 40.0, 39.0, 38.0])
    out = effective_length_bpb(bits, n)
    if out["beta"] <= 0:
        assert out["defined"] is False
        assert np.isnan(out["value"])


# Prefix BPB


def _affine_item(alpha: float, beta: float, length: int, offset: int = 0) -> PrefixItem:
    """Single-byte tokens with C(k) = alpha + beta*k (k >= 1)."""
    bits = np.full(length, beta, dtype=float)
    bits[0] = alpha + beta
    byte_lens = np.ones(length, dtype=float)
    return PrefixItem(bits, byte_lens, offset=offset)


def _proportional_item(beta: float, byte_lens: list[float]) -> PrefixItem:
    """alpha=0: token cost is beta * byte length, so C(k) = beta*k."""
    m = np.asarray(byte_lens, dtype=float)
    return PrefixItem(beta * m, m)


def test_cumulative_cost_linear_inside_token():
    # one token, 4 bytes, 8 bits -> 2 bits/byte, linear inside
    assert cumulative_cost([8.0], [4.0], 0) == pytest.approx(0.0)
    assert cumulative_cost([8.0], [4.0], 1) == pytest.approx(2.0)
    assert cumulative_cost([8.0], [4.0], 4) == pytest.approx(8.0)
    assert cumulative_cost([8.0], [4.0], 10) == pytest.approx(8.0)  # clamps at n


def test_cumulative_cost_zero_byte_token_included_at_boundary():
    # zero-byte split-UTF8 token: cost at the boundary
    bits = [3.0, 5.0, 7.0]
    byte_lens = [3.0, 0.0, 2.0]
    # at k=3 both the 3-byte and the 0-byte token are fully covered
    assert cumulative_cost(bits, byte_lens, 3) == pytest.approx(8.0)
    assert cumulative_cost(bits, byte_lens, 5) == pytest.approx(15.0)


def test_alpha_zero_gives_constant_pbpb_equal_beta():
    beta = 0.75
    items = [_proportional_item(beta, bl) for bl in ([2, 3, 5], [1, 4, 4, 2], [6, 1])]
    for k in (1, 2, 3, 5, 8):
        assert pbpb(items, k) == pytest.approx(beta, abs=1e-12)


def test_constant_length_pbpb_equals_corpus_bpb():
    rng = np.random.default_rng(3)
    length = 6
    items = []
    total_bits = []
    for _ in range(50):
        b = rng.uniform(0.5, 3.0, size=length)
        items.append(PrefixItem(b, np.ones(length)))
        total_bits.append(b.sum())
    bits_arr = np.array(total_bits)
    n_arr = np.full(len(items), float(length))
    assert pbpb(items, length) == pytest.approx(corpus_bpb(bits_arr, n_arr), rel=1e-12)


def test_synthetic_alpha_beta_recovery_prefix():
    alpha, beta = 12.0, 0.4
    items = [_affine_item(alpha, beta, length) for length in (20, 30, 45, 60)]
    # PBPB(K) = alpha/K + beta
    for k in (5, 10, 20):
        assert pbpb(items, k) == pytest.approx(alpha / k + beta, abs=1e-9)
    # entry cost at 1 byte is alpha+beta; continuation rate is beta
    assert entry_cost(items, 1) == pytest.approx(alpha + beta, abs=1e-9)
    assert continuation_rate(items, 1, 20) == pytest.approx(beta, abs=1e-9)


def test_decomposition_identity_is_exact():
    rng = np.random.default_rng(4)
    items = []
    for _ in range(40):
        length = int(rng.integers(15, 40))
        b = rng.uniform(0.2, 4.0, size=length)
        m = rng.integers(1, 4, size=length).astype(float)
        items.append(PrefixItem(b, m))
    k0, k = 4.0, 12.0
    d = prefix_decompose(items, k0, k)
    assert d["pbpb"] == pytest.approx(pbpb(items, k), rel=1e-12)
    assert (d["A"] + d["R"] * (k - k0)) / k == pytest.approx(d["pbpb"], rel=1e-12)


def test_pbpb_is_monotone_under_worsening():
    rng = np.random.default_rng(5)
    items = []
    for _ in range(30):
        length = int(rng.integers(10, 25))
        items.append(PrefixItem(rng.uniform(0.2, 3.0, size=length), np.ones(length)))
    base = pbpb(items, 8)
    worse = [PrefixItem(it.bits.copy(), it.byte_lens.copy()) for it in items]
    worse[0].bits[2] += 5.0  # a single token gets more surprising
    assert pbpb(worse, 8) >= base - 1e-12
    assert pbpb(worse, 8) > base


def test_coverage_drops_with_k_and_short_items_excluded():
    items = [
        PrefixItem(np.ones(2), np.ones(2)),  # 2 content bytes
        PrefixItem(np.ones(10), np.ones(10)),  # 10 content bytes
        PrefixItem(np.ones(30), np.ones(30)),  # 30
    ]
    curve = prefix_curve(items, [2, 10, 30])
    assert curve[0]["coverage"] == pytest.approx(1.0)
    assert curve[1]["coverage"] == pytest.approx(2 / 3)
    assert curve[2]["coverage"] == pytest.approx(1 / 3)
    assert curve[2]["n_items"] == 1


def test_leading_space_offset_excluded_from_denominator():
    # " A": 2 bytes total, offset=1 -> 1 content byte; PBPB(1) divides by 1.
    with_offset = make_prefix_items([([4.0], [2.0])], choices=[" A"])
    assert with_offset[0].offset == 1
    assert with_offset[0].content_bytes == 1
    # content byte 1 at absolute byte 2 -> 4.0 bits / 1 content byte
    assert pbpb(with_offset, 1) == pytest.approx(4.0)
    # without offset, same cost over 2 bytes
    no_offset = make_prefix_items([([4.0], [2.0])], choices=["XA"])
    assert no_offset[0].offset == 0
    assert pbpb(no_offset, 2) == pytest.approx(2.0)


def test_pbpb_nan_when_no_support():
    items = [PrefixItem(np.ones(3), np.ones(3))]
    assert np.isnan(pbpb(items, 10))
    assert np.isnan(entry_cost(items, 10))
    assert np.isnan(continuation_rate(items, 5, 10))


# Prior BPB


def test_per_byte_cost_conserves_bits_and_length():
    item = PrefixItem(np.array([3.0, 5.0, 7.0]), np.array([2.0, 1.0, 3.0]))  # 6 content bytes
    pbc = per_byte_cost(item)
    assert len(pbc) == 6
    assert pbc.sum() == pytest.approx(15.0)
    # linear attribution: first two bytes share the 3-bit token
    assert pbc[0] == pytest.approx(1.5)
    assert pbc[2] == pytest.approx(5.0)  # the 1-byte token


def test_per_byte_cost_offset_drops_leading_byte():
    item = PrefixItem(np.array([4.0]), np.array([2.0]), offset=1)  # " A": 1 content byte
    pbc = per_byte_cost(item)
    assert len(pbc) == 1
    assert pbc[0] == pytest.approx(2.0)  # 4 bits over 2 bytes, one content byte kept


def test_reference_self_prior_equals_corpus_bpb():
    """Reference prior with task lengths equals corpus BPB."""
    rng = np.random.default_rng(21)
    items, bits_total, nbytes = [], [], []
    for _ in range(60):
        length = int(rng.integers(3, 40))
        b = rng.uniform(0.2, 4.0, size=length)
        m = rng.integers(1, 4, size=length).astype(float)
        it = PrefixItem(b, m)
        items.append(it)
        bits_total.append(float(b.sum()))
        nbytes.append(float(it.content_bytes))
    bits_total = np.array(bits_total)
    nbytes = np.array(nbytes)
    got = prior_bpb(items, prior="reference", ref_lengths=nbytes)["value"]
    assert got == pytest.approx(corpus_bpb(bits_total, nbytes), rel=1e-9)


def test_flat_rate_gives_constant_for_any_prior():
    beta = 0.6
    items = [PrefixItem(beta * np.ones(n), np.ones(n)) for n in (5, 12, 30, 60)]
    for prior_kwargs in (
        {"prior": "geometric", "mu": 8.0},
        {"prior": "geometric", "mu": 40.0},
        {"prior": "reference", "ref_lengths": np.array([5.0, 12.0, 30.0, 60.0])},
    ):
        assert prior_bpb(items, **prior_kwargs)["value"] == pytest.approx(beta, abs=1e-9)


def test_affine_gives_alpha_over_mu_plus_beta():
    alpha, beta, mu = 10.0, 0.4, 20.0
    # R(1)=alpha+beta, R(k>=2)=beta => Prior BPB = alpha/mu + beta; items long vs mu.
    items = []
    for length in (200, 300, 400):
        bits = np.full(length, beta)
        bits[0] = alpha + beta
        items.append(PrefixItem(bits, np.ones(length)))
    got = prior_bpb(items, prior="geometric", mu=mu)["value"]
    assert got == pytest.approx(alpha / mu + beta, rel=1e-3)


def test_prior_bpb_uses_late_bytes_and_is_monotone():
    rng = np.random.default_rng(22)
    items = [PrefixItem(rng.uniform(0.3, 2.0, size=80), np.ones(80)) for _ in range(20)]
    base = prior_bpb(items, prior="geometric", mu=32.0)["value"]
    worse = [PrefixItem(it.bits.copy(), it.byte_lens.copy()) for it in items]
    worse[0].bits[70] += 20.0  # late byte gets more surprising
    after = prior_bpb(worse, prior="geometric", mu=32.0)["value"]
    assert after > base  # late bytes raise the score


def test_marginal_rate_curve_coverage_decreases():
    items = [PrefixItem(np.ones(n), np.ones(n)) for n in (2, 10, 30)]
    k, R, cov = marginal_rate_curve(items)
    assert len(k) == 30
    assert cov[0] == pytest.approx(1.0)
    assert cov[9] == pytest.approx(2 / 3)
    assert cov[29] == pytest.approx(1 / 3)
    assert R[0] == pytest.approx(1.0)
