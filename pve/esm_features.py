"""ESM-2 features: position-aware embeddings and zero-shot masked marginals.

Only the functions that actually run the network import ``torch``/``esm``; the
indexing, windowing and scoring logic is plain NumPy so it can be unit-tested
without the dependency.

Why position-aware: for a single substitution in a length-``L`` protein, the
mean-pooled embeddings of the wild-type and mutant sequences differ by roughly
``1/L``. Mean pooling therefore averages away most of the signal the model is
being asked to find. Taking the representation *at the mutated position* -- and
its contrast against the wild-type at the same position -- keeps it.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: ESM-2 positional limit, minus the BOS/EOS tokens.
MAX_TOKENS = 1022

POOLING_MODES = ("both", "mut_pos", "delta", "mean")


# --------------------------------------------------------------------------- #
# Pure logic
# --------------------------------------------------------------------------- #
def window_sequence(seq: str, center: int, max_len: int = MAX_TOKENS) -> tuple[str, int]:
    """Crop ``seq`` to ``max_len`` residues around 1-based ``center``.

    Returns ``(subsequence, new_center)`` with ``new_center`` also 1-based.
    Sequences that already fit are returned unchanged. ESM-2 cannot attend past
    ~1022 tokens, and many ProteinGym targets are longer than that.
    """
    if len(seq) <= max_len:
        return seq, center
    half = max_len // 2
    start = max(0, min(center - half, len(seq) - max_len))
    return seq[start:start + max_len], center - start


def masked_marginal_scores(
    logprobs_by_pos: dict, parsed: list, missing: float = np.nan
) -> np.ndarray:
    """Combine per-position log-probabilities into per-variant zero-shot scores.

    ``logprobs_by_pos`` maps a 1-based position to a dict of ``{aa: log p(aa)}``
    for the *wild-type* sequence with that position masked. The score of a
    variant is the summed log-odds of mutant over wild-type residue:

        score = sum over substitutions of  log p(mut) - log p(wt)

    which is the standard ESM-1v / ESM-2 masked-marginal protocol (Meier et al.
    2021). Higher = more like the wild type = predicted more tolerated, so it
    lines up in sign with a DMS fitness score.
    """
    out = np.full(len(parsed), missing, dtype=np.float64)
    for i, subs in enumerate(parsed):
        if not subs:
            continue
        total, ok = 0.0, True
        for wt, pos, mut in subs:
            lp = logprobs_by_pos.get(pos)
            if lp is None or wt not in lp or mut not in lp:
                ok = False
                break
            total += lp[mut] - lp[wt]
        if ok:
            out[i] = total
    return out


def _cache_key(model_name: str, mode: str, seq: str, pos) -> str:
    digest = hashlib.sha1(seq.encode()).hexdigest()[:16]
    return f"{model_name}|{mode}|{digest}|{pos}"


class EmbeddingCache:
    """Tiny on-disk cache of pooled embedding vectors.

    Only the pooled vectors are stored (a few hundred floats per entry), never
    the full ``L x D`` representation, so the file stays small. Without this,
    every re-run re-embeds the whole assay.
    """

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self.store: dict = {}
        self._dirty = False
        if self.path and self.path.exists():
            try:
                with np.load(self.path, allow_pickle=False) as f:
                    self.store = {k: f[k] for k in f.files}
                log.info("ESM cache: %d entries loaded from %s", len(self.store), self.path)
            except Exception as exc:  # a corrupt cache must never break a run
                log.warning("Ignoring unreadable ESM cache %s (%s)", self.path, exc)

    def get(self, key: str):
        return self.store.get(key)

    def put(self, key: str, value: np.ndarray) -> None:
        self.store[key] = value
        self._dirty = True

    def save(self) -> None:
        if not (self.path and self._dirty):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **self.store)
        log.info("ESM cache: %d entries saved to %s", len(self.store), self.path)


# --------------------------------------------------------------------------- #
# Torch-backed runner
# --------------------------------------------------------------------------- #
class ESMRunner:
    """Thin wrapper over a fair-esm model: batched embeddings and masked logits."""

    def __init__(self, model_name: str = "esm2_t12_35M_UR50D", device: str = "auto"):
        try:
            import torch
            import esm
        except ImportError as exc:  # pragma: no cover - exercised only without torch
            raise ImportError(
                "ESM features need `pip install -r requirements-esm.txt` "
                "(fair-esm and torch)."
            ) from exc

        self.torch = torch
        if device == "auto":
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        if not hasattr(esm.pretrained, model_name):
            raise ValueError(f"Unknown ESM model '{model_name}'.")
        log.info("Loading %s on %s ...", model_name, device)
        model, alphabet = getattr(esm.pretrained, model_name)()
        self.model = model.to(device).eval()
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.layer = model.num_layers
        self.model_name = model_name

    # -- representations ---------------------------------------------------- #
    def representations(self, seqs: list, batch_size: int = 8):
        """Yield ``(index, L x D)`` residue representations, batched by length."""
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = [(str(i), seqs[i]) for i in idx]
            _, _, toks = self.batch_converter(batch)
            with self.torch.no_grad():
                out = self.model(toks.to(self.device), repr_layers=[self.layer])
            reps = out["representations"][self.layer]
            for row, i in enumerate(idx):
                yield i, reps[row, 1:len(seqs[i]) + 1].float().cpu().numpy()

    def masked_logprobs(self, seq: str, positions: list, batch_size: int = 8) -> dict:
        """Masked-marginal log-probabilities at each 1-based position of ``seq``.

        One forward pass per *position*, not per variant -- a DMS assay has
        thousands of variants over a few hundred positions, so this is the
        difference between minutes and hours.
        """
        from .aaindex import AMINO_ACIDS

        aa_tokens = {aa: self.alphabet.get_idx(aa) for aa in AMINO_ACIDS}
        out: dict = {}
        for start in range(0, len(positions), batch_size):
            chunk = positions[start:start + batch_size]
            windows = [window_sequence(seq, p) for p in chunk]
            batch = [(str(p), sub) for p, (sub, _) in zip(chunk, windows)]
            _, _, toks = self.batch_converter(batch)
            toks = toks.to(self.device)
            for row, (_, local) in enumerate(windows):
                toks[row, local] = self.alphabet.mask_idx  # +1 for BOS, -1 for 1-based
            with self.torch.no_grad():
                logits = self.model(toks)["logits"]
            logprobs = self.torch.log_softmax(logits, dim=-1)
            for row, (pos, (_, local)) in enumerate(zip(chunk, windows)):
                vec = logprobs[row, local].float().cpu().numpy()
                out[pos] = {aa: float(vec[t]) for aa, t in aa_tokens.items()}
            if start and start % (batch_size * 25) == 0:
                log.info("  masked %d/%d positions", start, len(positions))
        return out


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def featurize_esm(
    sequences,
    parsed,
    wt_sequence: str | None,
    mode: str = "both",
    model_name: str = "esm2_t12_35M_UR50D",
    device: str = "auto",
    batch_size: int = 8,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Build ESM features under the requested pooling mode.

    ``mut_pos`` takes the mutant-sequence representation at the mutated position,
    ``delta`` its contrast with the wild type at the same position, ``both``
    concatenates them, and ``mean`` reproduces the original whole-sequence mean
    pooling (kept so the two can be compared directly).
    """
    if mode not in POOLING_MODES:
        raise ValueError(f"Unknown --esm-pool '{mode}'; choose from {POOLING_MODES}.")

    seqs = [str(s) for s in sequences]
    needs_positions = mode in ("mut_pos", "delta", "both")
    needs_wt = mode in ("delta", "both")

    if needs_wt and wt_sequence is None:
        log.warning("No wild-type sequence available; falling back to --esm-pool mut_pos.")
        mode, needs_wt = "mut_pos", False

    keep = np.ones(len(seqs), dtype=bool)
    if needs_positions:
        keep = np.array([bool(p) for p in parsed], dtype=bool)
        if not keep.any():
            raise ValueError(
                f"--esm-pool {mode} needs parseable mutant strings; none were found."
            )
        if (~keep).any():
            log.info("ESM: %d/%d variants have a parseable mutant string.",
                     int(keep.sum()), len(seqs))

    cache = EmbeddingCache(cache_path)
    runner = None

    # Positions to pool over, per kept row (1-based, relative to the full sequence).
    rows = [i for i in range(len(seqs)) if keep[i]]
    pos_per_row = {
        i: ([p for _, p, _ in parsed[i]] if needs_positions else None) for i in rows
    }

    # -- wild-type reference vectors --------------------------------------- #
    wt_vectors: dict = {}
    if needs_wt:
        wanted = sorted({p for i in rows for p in pos_per_row[i]})
        missing = [p for p in wanted
                   if cache.get(_cache_key(model_name, "wt_pos", wt_sequence, p)) is None]
        if missing:
            runner = runner or ESMRunner(model_name, device)
            log.info("Embedding wild-type reference at %d position(s) ...", len(missing))
            # Group by window so each distinct crop is embedded once.
            by_window: dict = {}
            for p in missing:
                sub, local = window_sequence(wt_sequence, p)
                by_window.setdefault(sub, []).append((p, local))
            subs = list(by_window)
            for idx, rep in runner.representations(subs, batch_size):
                for p, local in by_window[subs[idx]]:
                    cache.put(_cache_key(model_name, "wt_pos", wt_sequence, p), rep[local - 1])
        for p in wanted:
            wt_vectors[p] = cache.get(_cache_key(model_name, "wt_pos", wt_sequence, p))

    # -- variant vectors ---------------------------------------------------- #
    pool_tag = "mean" if mode == "mean" else "mutpos"
    todo = []
    for i in rows:
        tag = "mean" if pool_tag == "mean" else ",".join(map(str, pos_per_row[i]))
        if cache.get(_cache_key(model_name, pool_tag, seqs[i], tag)) is None:
            todo.append(i)

    if todo:
        runner = runner or ESMRunner(model_name, device)
        log.info("Embedding %d variant sequence(s) (%d cached) ...",
                 len(todo), len(rows) - len(todo))
        if mode == "mean":
            crops = [seqs[i][:MAX_TOKENS] for i in todo]
            for idx, rep in runner.representations(crops, batch_size):
                i = todo[idx]
                cache.put(_cache_key(model_name, pool_tag, seqs[i], "mean"), rep.mean(0))
        else:
            crops, locals_ = [], []
            for i in todo:
                centre = int(np.mean(pos_per_row[i]))
                sub, local_centre = window_sequence(seqs[i], centre)
                # The crop starts at `centre - local_centre` (0-based). Deriving the
                # offset arithmetically rather than with str.find avoids latching on
                # to the wrong copy when a repeated motif appears twice.
                offset = centre - local_centre
                crops.append(sub)
                locals_.append([p - offset for p in pos_per_row[i]])
            for idx, rep in runner.representations(crops, batch_size):
                i = todo[idx]
                sel = [l - 1 for l in locals_[idx] if 1 <= l <= rep.shape[0]]
                vec = rep[sel].mean(0) if sel else np.zeros(rep.shape[1], dtype=np.float32)
                tag = ",".join(map(str, pos_per_row[i]))
                cache.put(_cache_key(model_name, pool_tag, seqs[i], tag), vec)

    cache.save()

    # -- assemble ----------------------------------------------------------- #
    feats = []
    for i in rows:
        tag = "mean" if pool_tag == "mean" else ",".join(map(str, pos_per_row[i]))
        vec = cache.get(_cache_key(model_name, pool_tag, seqs[i], tag))
        if mode in ("mut_pos", "mean"):
            feats.append(vec)
        else:
            ref = np.mean([wt_vectors[p] for p in pos_per_row[i]], axis=0)
            feats.append(vec - ref if mode == "delta" else np.concatenate([vec, vec - ref]))

    X = np.vstack(feats).astype(np.float32)
    D = X.shape[1]
    if mode == "both":
        names = ([f"esm_mut_{k}" for k in range(D // 2)]
                 + [f"esm_delta_{k}" for k in range(D - D // 2)])
    else:
        names = [f"esm_{mode}_{k}" for k in range(D)]
    log.info("ESM features (%s pooling, %s): %s", mode, model_name, X.shape)
    return X, keep, names


def esm_zeroshot(
    wt_sequence: str,
    parsed,
    model_name: str = "esm2_t12_35M_UR50D",
    device: str = "auto",
    batch_size: int = 8,
) -> np.ndarray:
    """Zero-shot masked-marginal score per variant. No training involved."""
    positions = sorted({p for subs in parsed if subs for _, p, _ in subs})
    if not positions:
        return np.full(len(parsed), np.nan)
    runner = ESMRunner(model_name, device)
    log.info("Zero-shot: masking %d unique position(s) ...", len(positions))
    logprobs = runner.masked_logprobs(wt_sequence, positions, batch_size)
    return masked_marginal_scores(logprobs, list(parsed))
