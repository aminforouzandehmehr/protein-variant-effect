"""ESM scoring logic, tested without torch.

The indexing and scoring rules are the parts that can silently be wrong (an
off-by-one in the masked position produces plausible-looking nonsense), so they
live in pure functions that these tests can drive directly.
"""

import numpy as np
import pytest

from pve.esm_features import (
    MAX_TOKENS,
    EmbeddingCache,
    masked_marginal_scores,
    window_sequence,
)


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def test_short_sequences_pass_through_unchanged():
    seq = "MKVLA" * 10
    sub, centre = window_sequence(seq, 7)
    assert sub == seq and centre == 7


def test_window_keeps_the_centre_residue_identical():
    """Whatever the crop, position `centre` in the window must be the same residue."""
    rng = np.random.RandomState(0)
    seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=3000))
    for pos in (1, 500, 1500, 2999, 3000):
        sub, local = window_sequence(seq, pos)
        assert len(sub) == MAX_TOKENS
        assert 1 <= local <= MAX_TOKENS
        assert sub[local - 1] == seq[pos - 1]


def test_window_respects_the_length_limit_at_both_ends():
    seq = "A" * 2000
    assert len(window_sequence(seq, 1)[0]) == MAX_TOKENS
    assert len(window_sequence(seq, 2000)[0]) == MAX_TOKENS


def test_window_offset_is_recoverable_even_with_repeated_motifs():
    """`start = centre - local` must locate the crop exactly.

    The featurizer derives the crop offset arithmetically rather than with
    str.find, which would latch onto the first copy of a repeated motif and
    silently mis-map every position in the window.
    """
    motif = "MKVLAWTYQ"
    seq = motif * 300  # ~2700 residues, every window appears many times over
    for centre in (5, 900, 1500, 2400, 2700):
        sub, local = window_sequence(seq, centre)
        start = centre - local
        assert seq[start:start + len(sub)] == sub
        # every other mutated position in the same window maps correctly too
        for other in range(max(1, centre - 3), min(len(seq), centre + 4)):
            if 1 <= other - start <= len(sub):
                assert sub[other - start - 1] == seq[other - 1]


# --------------------------------------------------------------------------- #
# masked marginals
# --------------------------------------------------------------------------- #
def test_masked_marginal_is_log_odds_of_mutant_over_wildtype():
    logprobs = {10: {"A": np.log(0.1), "V": np.log(0.4)}}
    scores = masked_marginal_scores(logprobs, [[("A", 10, "V")]])
    assert scores[0] == pytest.approx(np.log(0.4) - np.log(0.1))


def test_a_favoured_mutation_outranks_a_disfavoured_one():
    """Higher score must mean 'more wild-type-like', matching DMS fitness sign."""
    logprobs = {5: {"G": np.log(0.5), "A": np.log(0.4), "W": np.log(0.001)}}
    scores = masked_marginal_scores(logprobs, [[("G", 5, "A")], [("G", 5, "W")]])
    assert scores[0] > scores[1]


def test_multiple_substitutions_sum():
    logprobs = {
        1: {"A": np.log(0.5), "C": np.log(0.25)},
        2: {"D": np.log(0.5), "E": np.log(0.125)},
    }
    scores = masked_marginal_scores(logprobs, [[("A", 1, "C"), ("D", 2, "E")]])
    assert scores[0] == pytest.approx(np.log(0.5) + np.log(0.25))


def test_missing_position_or_residue_yields_nan():
    logprobs = {1: {"A": -1.0, "C": -2.0}}
    scores = masked_marginal_scores(
        logprobs, [None, [("A", 99, "C")], [("A", 1, "Z")], [("A", 1, "C")]]
    )
    assert np.isnan(scores[:3]).all()
    assert np.isfinite(scores[3])


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def test_cache_round_trips_through_disk(tmp_path):
    path = tmp_path / "emb.npz"
    cache = EmbeddingCache(path)
    vec = np.arange(8, dtype=np.float32)
    cache.put("k", vec)
    cache.save()

    assert np.array_equal(EmbeddingCache(path).get("k"), vec)


def test_cache_survives_a_corrupt_file(tmp_path):
    """A broken cache must degrade to a cold start, not kill the run."""
    path = tmp_path / "emb.npz"
    path.write_bytes(b"not an npz")
    cache = EmbeddingCache(path)
    assert cache.get("k") is None


def test_cache_without_a_path_is_a_no_op(tmp_path):
    cache = EmbeddingCache(None)
    cache.put("k", np.zeros(3))
    cache.save()  # must not raise
    assert cache.get("k") is not None
