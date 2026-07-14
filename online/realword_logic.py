"""Real-word (valid word) spelling error detection.

This module plugs into the existing `service.py` workflow:

    rw_errors, rw_suggestions = realword.detect_realword_errors(...)

Upgrade highlights (no extra training):
  - Supports **two LM backends**:
      (A) HuggingFace masked-LM pipeline (slow but strong)
      (B) N-gram LM (nltk.lm or our fin_ngram.pkl pack) (fast)
  - Uses **financial IDF pack** (resources/models/fin_idf.pkl) to *protect rare
    domain terms* and reduce false positives.

The output format MUST stay compatible with the existing business layer:
  - errors: list[dict] with token span info
  - suggestions_map: dict[str(index)] -> list[ {token, score, ...} ]
"""

from __future__ import annotations

import math
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------
# Settings
# ---------------------------

@dataclass
class RealWordSettings:
    # MLM (if available)
    model_name: str = "distilbert-base-uncased"
    top_k: int = 15

    # How many token positions to consider at most (speed vs recall)
    max_positions: int = 8

    # Decision thresholds (backend-agnostic)
    # For MLM: "score" is probability for the best candidate at the masked position.
    # For N-gram backend we DO NOT interpret exp(logprob) as a probability.
    # Instead, we convert {original}+candidates' local log-scores to a softmax
    # probability distribution so the same thresholds remain meaningful.
    min_best_score: float = 0.45
    min_gain: float = 0.12
    min_ratio: float = 3.0
    min_freq_ratio: float = 1.8

    # Optional: bigram surprisal trigger (nltk.lm). Disabled by default for speed.
    bigram_suspicious_surprisal: Optional[float] = None

    # Token filters
    min_token_len: int = 3
    skip_proper_like: bool = True

    # Speed knobs for MLM scoring
    window_size: int = 64     # tokens on each side
    max_seq_len: int = 256
    batch_size: int = 8

    # --- NEW: IDF guardrails (reduce FP) ---
    enable_idf_guard: bool = True
    # Protect rare tokens (likely names/tickers/terms). Higher = stricter.
    # With N≈20k sentences, idf is roughly in [1, ~11].
    idf_protect_threshold: float = 7.0
    # Also protect tokens that look like acronyms/tickers even if IDF is unknown.
    protect_ticker_like: bool = True

    ngram_require_observed_bigram: bool = True
    ngram_rare_max_freq: int = 3
    ngram_rare_min_freq_ratio: float = 10.0


def _parse_settings(settings: Dict[str, Any]) -> RealWordSettings:
    cfg = RealWordSettings()
    if not isinstance(settings, dict):
        return cfg

    # allow nested overrides under "realword"
    src = settings.get("realword", settings) if isinstance(settings.get("realword", settings), dict) else settings
    for k in cfg.__dict__.keys():
        if k in src:
            try:
                setattr(cfg, k, type(getattr(cfg, k))(src[k]))
            except Exception:
                # keep default on type mismatch
                pass
    return cfg


# ---------------------------
# IDF pack (optional)
# ---------------------------

_IDF_PACK: Optional[Dict[str, Any]] = None

def _try_load_fin_idf_pack() -> Optional[Dict[str, Any]]:
    """Try to load resources/models/fin_idf.pkl relative to this file or CWD."""
    global _IDF_PACK
    if _IDF_PACK is not None:
        return _IDF_PACK

    cand_paths: List[str] = []
    # current working dir
    cand_paths.append(os.path.join("resources", "models", "fin_idf.pkl"))
    # relative to this module (online/)
    here = os.path.dirname(os.path.abspath(__file__))
    cand_paths.append(os.path.join(here, "..", "resources", "models", "fin_idf.pkl"))
    cand_paths.append(os.path.join(here, "..", "..", "resources", "models", "fin_idf.pkl"))

    for p in cand_paths:
        p2 = os.path.normpath(p)
        if os.path.isfile(p2):
            try:
                with open(p2, "rb") as f:
                    _IDF_PACK = pickle.load(f)
                return _IDF_PACK
            except Exception:
                _IDF_PACK = None
                return None

    _IDF_PACK = None
    return None


def _idf_of(word: str) -> Optional[float]:
    pack = _try_load_fin_idf_pack()
    if not pack:
        return None
    if isinstance(pack, dict):
        d = pack.get("idf", {})
        if isinstance(d, dict):
            return d.get(word.lower())
        return None
    d = getattr(pack, "idf", None)
    if isinstance(d, dict):
        return d.get(word.lower())
    return None


# ---------------------------
# Token utils
# ---------------------------

_PUNCT_RE = re.compile(r"^[\W_]+$")

def _is_punct(tok: str) -> bool:
    return bool(_PUNCT_RE.match(tok or ""))

def _looks_like_number(tok: str) -> bool:
    return bool(re.search(r"\d", tok or ""))

_TICKER_RE = re.compile(r"^[A-Z]{1,6}(\.[A-Z]{1,3})?$")

def _looks_like_ticker(tok: str) -> bool:
    if not tok:
        return False
    if _TICKER_RE.match(tok):
        return True
    # often tickers / acronyms are short ALLCAPS
    return tok.isupper() and 2 <= len(tok) <= 6


# ---------------------------
# Candidate generation
# ---------------------------

def _edits1(word: str) -> Set[str]:
    """Edits at distance 1."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = {L + R[1:] for L, R in splits if R}
    transposes = {L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1}
    replaces = {L + c + R[1:] for L, R in splits if R for c in letters}
    inserts = {L + c + R for L, R in splits for c in letters}
    return deletes | transposes | replaces | inserts

def _levenshtein(s1: str, s2: str, max_dist: Optional[int] = None) -> int:
    s1 = s1.lower()
    s2 = s2.lower()
    n, m = len(s1), len(s2)
    if s1 == s2:
        return 0
    if n == 0:
        return m
    if m == 0:
        return n
    if max_dist is not None and abs(n - m) > max_dist:
        return max_dist + 1
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        row_min = cur[0]
        c1 = s1[i - 1]
        for j in range(1, m + 1):
            cost = 0 if c1 == s2[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
            if cur[j] < row_min:
                row_min = cur[j]
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        prev, cur = cur, prev
    return prev[m]

def _bucket_candidate_pool(
    word: str,
    *,
    vocab: Set[str],
    cand_buckets: Optional[Dict[str, Any]],
    max_ed: int,
) -> Set[str]:
    w = (word or "").lower()
    if not w:
        return set()
    if not cand_buckets:
        return set(vocab)
    length_buckets = cand_buckets.get("length", {}) if isinstance(cand_buckets, dict) else {}
    first_char_buckets = cand_buckets.get("first_char", {}) if isinstance(cand_buckets, dict) else {}
    L = len(w)
    cand_len_set: Set[str] = set()
    if isinstance(length_buckets, dict):
        for d in range(-max_ed, max_ed + 1):
            ll = L + d
            if ll <= 0:
                continue
            vals = length_buckets.get(ll, [])
            if isinstance(vals, list):
                cand_len_set.update(vals)
    fc = w[0]
    cand_fc_set: Set[str] = set()
    if isinstance(first_char_buckets, dict):
        vals = first_char_buckets.get(fc, [])
        if isinstance(vals, list):
            cand_fc_set.update(vals)
    cand_set = cand_len_set & cand_fc_set if (cand_len_set and cand_fc_set) else (cand_len_set or cand_fc_set)
    if not cand_set:
        cand_set = cand_len_set or cand_fc_set
    cand_set = {c.lower() for c in cand_set if isinstance(c, str)}
    return cand_set & vocab


def _generate_candidates(
    tok: str,
    *,
    vocab_set_for_candidates: Set[str],
    vocab_set_for_membership: Set[str],
    cand_buckets: Optional[Dict[str, Any]] = None,
    finance_extra: Set[str],
    word_freq: Dict[str, int],
    max_cands: int = 20,
    max_ed: int = 2,
) -> List[str]:
    """Generate a small candidate set based on edit distance and vocab/frequency."""
    w = (tok or "").lower()
    if not w:
        return []

    # If token is domain extra (tickers/entities), don't touch.
    if w in (finance_extra or set()):
        return []

    vocab = set(vocab_set_for_membership or set()) | set(vocab_set_for_candidates or set())
    pool = _bucket_candidate_pool(w, vocab=vocab, cand_buckets=cand_buckets, max_ed=max_ed)
    if pool:
        if max_ed <= 1 and len(pool) > 2000:
            cands = set()
            for e in _edits1(w):
                if e in vocab:
                    cands.add(e)
        else:
            cands = {c for c in pool if _levenshtein(w, c, max_dist=max_ed) <= max_ed}
    else:
        cands = set()
        for e in _edits1(w):
            if e in vocab:
                cands.add(e)

    # include plural/singular simple heuristic (helps many dataset errors)
    if w.endswith("s") and len(w) > 3:
        base = w[:-1]
        if base in vocab:
            cands.add(base)
    else:
        s = w + "s"
        if s in vocab:
            cands.add(s)

    # Keep only top by frequency
    def freq_key(x: str) -> int:
        return int(word_freq.get(x, 0))

    ranked = sorted(cands, key=freq_key, reverse=True)
    ranked = ranked[:max_cands]

    # Ensure we don't suggest the same token
    ranked = [x for x in ranked if x != w]
    return ranked


# ---------------------------
# POS lexicon (soft filter)
# ---------------------------

def _pos_compatible(original_pos: Optional[str], cand: str, pos_lexicon: Dict[str, Dict[str, int]]) -> bool:
    if not original_pos:
        return True
    if not pos_lexicon:
        return True
    cand_l = cand.lower()
    tag_counts = pos_lexicon.get(cand_l)
    if not tag_counts:
        return True
    if original_pos in tag_counts:
        return True
    if str(original_pos) in {"NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}:
        # Do not block content words here; allow group check to proceed
        pass

    def pos_group(tag: str) -> str:
        if not tag:
            return "O"
        if tag.startswith("NN"):
            return "N"
        if tag.startswith("VB"):
            return "V"
        if tag.startswith("JJ"):
            return "J"
        if tag.startswith("RB"):
            return "R"
        return "O"

    og = pos_group(str(original_pos))
    if og == "O":
        return True
    
    # Allow Noun <-> Adjective compatibility (e.g. mail -> main)
    compatible_groups = {og}
    if og == "N":
        compatible_groups.add("J")
    elif og == "J":
        compatible_groups.add("N")

    for t in tag_counts.keys():
        if pos_group(str(t)) in compatible_groups:
            return True
    return False


# ---------------------------
# Bigram LM helpers (nltk.lm OR fin_ngram.pkl dict)
# ---------------------------

def _cheap_bigram_surprisal(lm_model: Any, prev_tok: str, tok: str) -> Optional[float]:
    """Return -log p(tok|prev) if lm_model supports it (nltk.lm)."""
    if lm_model is None:
        return None
    prev = (prev_tok or "").lower()
    w = (tok or "").lower()
    if not prev or not w:
        return None
    if isinstance(lm_model, dict) and ("bigram" in lm_model or "unigram" in lm_model):
        lp = _pack_bigram_logprob(lm_model, prev, w)
        return -float(lp)
    try:
        p = float(lm_model.score(w, [prev]))  # nltk.lm
        if p > 0:
            return -math.log(p)
    except Exception:
        pass
    return None


def _pack_bigram_logprob(
    pack: Dict[str, Any],
    prev: str,
    w: str,
    k: float = 0.1,
    gamma: float = 5.0,
) -> float:
    """Interpolated log P(w|prev) from fin_ngram.pkl pack."""
    bigram = pack.get("bigram", {})
    unigram = pack.get("unigram", {})
    vocab = pack.get("vocab", None)
    V = 1
    if isinstance(vocab, list):
        V = max(1, len(vocab))
    elif isinstance(vocab, dict):
        V = max(
            1,
            int(
                vocab.get("size", 0)
                or len(vocab.get("itos", []))
                or len(vocab.get("stoi", {}))
                or 1
            ),
        )

    prev = (prev or "").lower()
    w = (w or "").lower()
    if not prev or not w:
        # fall back to unigram
        c = float(unigram.get(w, 0.0))
        tot = float(pack.get("total_unigram", pack.get("total", 1.0)) or 1.0)
        return math.log((c + k) / (tot + k * V))

    c_pw = 0.0
    c_prev = 0.0

    if isinstance(bigram, dict) and bigram:
        sample_val = next(iter(bigram.values()))
        if isinstance(sample_val, dict):
            row = bigram.get(prev, {})
            if isinstance(row, dict):
                c_pw = float(row.get(w, 0.0))
                c_prev = float(sum(row.values())) if row else float(unigram.get(prev, 0.0))
        else:
            key_tab = f"{prev}\t{w}"
            key_sp = f"{prev} {w}"
            c_pw = float(bigram.get(key_tab, bigram.get(key_sp, 0.0)) or 0.0)
            c_prev = float(unigram.get(prev, 0.0))

    if c_prev <= 0:
        c_prev = float(unigram.get(prev, 0.0))

    tot = float(pack.get("total_unigram", pack.get("total", 1.0)) or 1.0)
    p_uni = (float(unigram.get(w, 0.0)) + k) / (tot + k * V)
    if c_pw <= 0 or c_prev <= 0:
        return math.log(p_uni if p_uni > 0 else 1e-12)

    p_bi = (c_pw + k) / (c_prev + k * V)
    alpha = float(c_prev) / float(c_prev + gamma) if (c_prev + gamma) > 0 else 0.0
    p = alpha * p_bi + (1.0 - alpha) * p_uni
    return math.log(p if p > 0 else 1e-12)


def _pack_bigram_count(pack: Dict[str, Any], prev: str, w: str) -> float:
    bigram = pack.get("bigram", {})
    prev = (prev or "").lower()
    w = (w or "").lower()
    if not prev or not w:
        return 0.0
    if not isinstance(bigram, dict) or not bigram:
        return 0.0
    sample_val = next(iter(bigram.values()))
    if isinstance(sample_val, dict):
        row = bigram.get(prev, {})
        if not isinstance(row, dict):
            return 0.0
        return float(row.get(w, 0.0) or 0.0)
    return float(bigram.get(f"{prev}\t{w}", bigram.get(f"{prev} {w}", 0.0)) or 0.0)


def _local_score_ngram(lm_model: Any, prev: str, w: str, nxt: str) -> float:
    """A local *bi-bigram* score: log P(w|prev) + log P(nxt|w). Higher is better.

    Supports:
      - nltk.lm: lm_model.score(word, context)
      - fin_ngram.pkl dict pack: counts-based
    """
    prev_l = (prev or "").lower()
    w_l = (w or "").lower()
    nxt_l = (nxt or "").lower()

    lp1 = 0.0
    lp2 = 0.0

    # nltk.lm
    if hasattr(lm_model, "score") and callable(getattr(lm_model, "score")) and not isinstance(lm_model, dict):
        try:
            p1 = float(lm_model.score(w_l, [prev_l])) if prev_l and w_l else 0.0
            p2 = float(lm_model.score(nxt_l, [w_l])) if w_l and nxt_l else 0.0
            if p1 > 0:
                lp1 = math.log(p1)
            else:
                lp1 = -50.0
            if nxt_l:
                lp2 = math.log(p2) if p2 > 0 else -50.0
            else:
                lp2 = 0.0
            return lp1 + lp2
        except Exception:
            pass

    # our pack dict
    if isinstance(lm_model, dict) and ("bigram" in lm_model or "unigram" in lm_model):
        lp1 = _pack_bigram_logprob(lm_model, prev_l, w_l)
        lp2 = _pack_bigram_logprob(lm_model, w_l, nxt_l) if nxt_l else 0.0
        return lp1 + lp2

    # no LM -> neutral
    return 0.0


# ---------------------------
# MLM helpers (transformers pipeline)
# ---------------------------

def _is_mlm_pipeline(obj: Any) -> bool:
    # Transformers pipeline is callable AND has .tokenizer and .model
    return callable(obj) and hasattr(obj, "tokenizer") and hasattr(obj, "model")


def _mlm_mask_token(tokenizer, word: str) -> Optional[int]:
    """Return token id if word maps to a single token in tokenizer, else None."""
    try:
        pieces = tokenizer.tokenize(" " + word)
        if len(pieces) != 1:
            return None
        return tokenizer.convert_tokens_to_ids(pieces[0])
    except Exception:
        return None


def _make_window(tokens: List[str], idx: int, window_size: int) -> Tuple[str, int, int]:
    """Return (window_text, window_start_idx, window_end_idx_exclusive)."""
    left = max(0, idx - window_size)
    right = min(len(tokens), idx + window_size + 1)
    window_tokens = tokens[left:right]
    return " ".join(window_tokens), left, right


# ---------------------------
# Public API
# ---------------------------

_COMMON_FUNC_WORDS = {
    "in", "on", "at", "to", "of", "for", "with", "by", "from", "about",
    "as", "into", "like", "through", "after", "over", "between", "out",
    "against", "during", "without", "before", "under", "around", "among",
    "the", "a", "an", "and", "or", "but", "if", "because", "while",
    "so", "than", "when", "where", "which", "who", "what", "how",
    "that", "this", "these", "those", "it", "he", "she", "they", "we", "i", "you",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "can", "could", "should", "would", "will", "may", "might", "must"
}

def detect_realword_errors(
    *,
    tokens: List[Dict[str, Any]],
    settings: Dict[str, Any],
    vocab_set_for_candidates: Set[str],
    vocab_set_for_membership: Set[str],
    finance_extra: Set[str],
    word_freq: Dict[str, int],
    pos_lexicon: Dict[str, Dict[str, int]],
    cand_buckets: Optional[Dict[str, Any]] = None,
    lm_model: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Detect real-word errors and suggest replacements.

    Returns:
      errors: list of dicts (token span + metadata)
      suggestions: dict keyed by token index (string) -> list of suggestions
    """
    cfg = _parse_settings(settings or {})
    src = settings.get("realword", settings) if isinstance(settings.get("realword", settings), dict) else settings
    backend = str((settings or {}).get("realword_backend", "ngram") or "").strip().lower()
    if backend in {"n-gram", "ngram", "bigram"}:
        backend = "ngram"
    elif backend in {"mlm", "bert"}:
        backend = "mlm"
    elif backend in {"hybrid", "mix", "mixed"}:
        backend = "hybrid"
    elif backend in {"auto", ""}:
        backend = "auto"

    lm_ngram = lm_model
    lm_mlm = lm_model
    if isinstance(lm_model, dict) and "ngram" in lm_model and "mlm" in lm_model:
        lm_ngram = lm_model.get("ngram")
        lm_mlm = lm_model.get("mlm")

    if backend == "ngram":
        if isinstance(src, dict):
            if "max_positions" not in src:
                cfg.max_positions = min(int(cfg.max_positions), 6)
            if "min_best_score" not in src:
                cfg.min_best_score = max(float(cfg.min_best_score), 0.60)
            if "min_gain" not in src:
                cfg.min_gain = max(float(cfg.min_gain), 0.22)
            if "min_ratio" not in src:
                cfg.min_ratio = max(float(cfg.min_ratio), 4.0)
            if "bigram_suspicious_surprisal" not in src:
                cfg.bigram_suspicious_surprisal = 6.0
            if "idf_protect_threshold" not in src:
                cfg.idf_protect_threshold = min(float(cfg.idf_protect_threshold), 6.5)
    elif backend == "hybrid":
        if isinstance(src, dict):
            if "max_positions" not in src:
                cfg.max_positions = min(int(cfg.max_positions), 4)
            if "window_size" not in src:
                cfg.window_size = min(int(cfg.window_size), 24)

    apply_idf_guard = bool(cfg.enable_idf_guard)
    idf_pack = _try_load_fin_idf_pack() if apply_idf_guard else None

    # tokens -> string list
    seq: List[str] = [str(t.get("token", "")) for t in tokens]
    if not seq:
        return [], {}

    cand_vocab = {w.lower() for w in (vocab_set_for_candidates or set())}

    # --- 1) choose positions (cheap filtering) ---
    suspicious_scores: List[Tuple[float, int]] = []

    for i, t in enumerate(tokens):
        tok = str(t.get("token", ""))
        if not tok:
            continue
        tok_l = tok.lower()
        original_pos = t.get("pos")
        if original_pos and (settings or {}).get("enable_pos_filter", True):
            pos_s = str(original_pos)
            is_content = (pos_s.startswith("NN") or pos_s.startswith("VB") or pos_s.startswith("JJ") or pos_s.startswith("RB"))
            
            # If NLTK says not content, check if it's a known content word in lexicon
            if not is_content:
                lex_entry = pos_lexicon.get(tok_l)
                if lex_entry:
                    for tag in lex_entry:
                        if tag.startswith("NN") or tag.startswith("VB") or tag.startswith("JJ") or tag.startswith("RB"):
                            is_content = True
                            break
            
            # If still not content, check if it's NOT a common function word
            # (catches rare words or mis-tagged content words like "trough" tagged as IN)
            if not is_content:
                if tok_l not in _COMMON_FUNC_WORDS and len(tok_l) > 2:
                    is_content = True

            if not is_content:
                continue

        if _is_punct(tok) or _looks_like_number(tok) or len(tok_l) < cfg.min_token_len:
            continue
        if tok_l in (finance_extra or set()):
            continue
        if cfg.protect_ticker_like and _looks_like_ticker(tok):
            continue
        if cfg.skip_proper_like and tok[:1].isupper() and i != 0:
            continue

        # must be a valid vocab word (real-word) in at least one vocab
        if tok_l not in (vocab_set_for_membership or set()) and tok_l not in cand_vocab:
            continue

        # IDF guard: protect rare terms (reduce FP in finance mode)
        if apply_idf_guard:
            idf = _idf_of(tok_l)
            if idf is not None and idf >= cfg.idf_protect_threshold:
                continue

        prev_tok = ""
        j = i - 1
        while j >= 0:
            pt = seq[j]
            if pt and not _is_punct(pt):
                prev_tok = pt
                break
            j -= 1
        if not prev_tok:
            continue
        surprisal = _cheap_bigram_surprisal(lm_ngram, prev_tok, tok)
        if cfg.bigram_suspicious_surprisal is not None and surprisal is not None:
            if float(surprisal) < float(cfg.bigram_suspicious_surprisal):
                continue
        freq = float(word_freq.get(tok_l, 0))
        score = (float(surprisal) if surprisal is not None else 0.0) + 0.25 * (1.0 / (freq + 1.0))
        suspicious_scores.append((score, i))

    suspicious_scores.sort(reverse=True)
    candidate_indices = [i for _, i in suspicious_scores[: cfg.max_positions]]
    if not candidate_indices:
        return [], {}

    # --- 2) score candidates ---
    errors: List[Dict[str, Any]] = []
    suggestions: Dict[str, Any] = {}

    # choose backend
    if backend == "auto":
        use_mlm = _is_mlm_pipeline(lm_mlm)
    else:
        use_mlm = backend in {"mlm", "hybrid"}
    use_mlm = bool(use_mlm) and _is_mlm_pipeline(lm_mlm)

    if use_mlm:
        # MLM backend (strong but heavier)
        tokenizer = lm_mlm.tokenizer
        mask_tok = getattr(tokenizer, "mask_token", "[MASK]")

        masked_sents: List[str] = []
        meta: List[Tuple[int, int, int, str, str, Optional[str]]] = []  # (idx, winL, winR, orig, orig_pos, prev_tok)
        # build batch inputs
        for idx in candidate_indices:
            tok = seq[idx]
            tok_l = tok.lower()
            original_pos = tokens[idx].get("pos")
            # candidates
            cands = _generate_candidates(
                tok,
                vocab_set_for_candidates=vocab_set_for_candidates,
                vocab_set_for_membership=vocab_set_for_membership,
                cand_buckets=cand_buckets,
                finance_extra=finance_extra,
                word_freq=word_freq,
                max_cands=max(12, cfg.top_k),
            )
            if not cands:
                continue

            window_text, L, R = _make_window(seq, idx, cfg.window_size)
            # replace only the target token inside the window with mask
            window_tokens = seq[L:R]
            window_tokens[idx - L] = mask_tok
            masked = " ".join(window_tokens)
            masked_sents.append(masked)
            meta.append((idx, L, R, tok_l, str(original_pos) if original_pos else "", "",))

        if not masked_sents:
            return [], {}

        top_k = max(5, int(cfg.top_k))
        # call pipeline in batches
        for b in range(0, len(masked_sents), cfg.batch_size):
            batch_sents = masked_sents[b:b+cfg.batch_size]
            batch_meta = meta[b:b+cfg.batch_size]
            try:
                out = lm_mlm(batch_sents, top_k=top_k)
            except TypeError:
                # some pipeline versions use topk
                out = lm_mlm(batch_sents, topk=top_k)

            # out can be list[list[dict]] for batch or list[dict] for single
            if isinstance(out, list) and out and isinstance(out[0], dict):
                out = [out]

            for preds, (idx, L, R, orig_l, orig_pos, _) in zip(out, batch_meta):
                tok = seq[idx]
                tok_l = tok.lower()

                cands = _generate_candidates(
                    tok,
                    vocab_set_for_candidates=vocab_set_for_candidates,
                    vocab_set_for_membership=vocab_set_for_membership,
                    cand_buckets=cand_buckets,
                    finance_extra=finance_extra,
                    word_freq=word_freq,
                    max_cands=max(12, cfg.top_k),
                )
                if not cands:
                    continue

                # filter MLM predictions by our candidate list (+ original word)
                cand_set = set(cands)
                best = None  # (score, token)
                orig_score = 0.0

                for p in preds:
                    token_str = str(p.get("token_str", "")).strip()
                    if not token_str:
                        continue
                    w = token_str.lower()
                    score = float(p.get("score", 0.0))
                    if w == tok_l:
                        orig_score = max(orig_score, score)
                    if w in cand_set:
                        # POS compatibility
                        if orig_pos and not _pos_compatible(orig_pos, w, pos_lexicon):
                            continue
                        if best is None or score > best[0]:
                            best = (score, w)

                if best is None:
                    continue

                best_score, best_tok = best
                gain = best_score - orig_score

                # IDF-aware extra guard: if original is rare, require larger gain
                if cfg.enable_idf_guard:
                    idf = _idf_of(tok_l)
                    if idf is not None and idf >= (cfg.idf_protect_threshold - 0.5):
                        if gain < (cfg.min_gain + 0.08):
                            continue

                if best_score >= cfg.min_best_score and gain >= cfg.min_gain:
                    err = {
                        "token_index": idx,
                        "start": int(tokens[idx].get("start", -1)),
                        "end": int(tokens[idx].get("end", -1)),
                        "wrong": tok,
                        "correct": best_tok,
                        "type": "real-word",
                        "meta": {"backend": "mlm", "best_score": best_score, "gain": gain},
                    }
                    errors.append(err)

                    # suggestions list: keep stable keys for frontend/business
                    sug_list = suggestions.setdefault(str(idx), [])
                    sug_list.append({"token": best_tok, "score": float(best_score), "source": "mlm"})

    else:
        # N-gram backend (fast). lm_model can be nltk.lm OR fin_ngram pack dict.
        score_cache: Dict[Tuple[str, str], float] = {}

        def score_bigram(next_w: str, prev_w: str) -> float:
            key = (prev_w.lower(), next_w.lower())
            if key in score_cache:
                return score_cache[key]
            try:
                p = float(lm_ngram.score(key[1], [key[0]]))  # nltk.lm
            except Exception:
                p = 0.0
            lp = math.log(p) if p > 0 else -50.0
            score_cache[key] = lp
            return lp

        for idx in candidate_indices:
            tok = seq[idx]
            tok_l = tok.lower()
            original_pos = tokens[idx].get("pos")

            # IDF guard already applied above, but keep a quick check
            if apply_idf_guard:
                idf = _idf_of(tok_l)
                if idf is not None and idf >= cfg.idf_protect_threshold:
                    continue

            # candidates
            cands = _generate_candidates(
                tok,
                vocab_set_for_candidates=vocab_set_for_candidates,
                vocab_set_for_membership=vocab_set_for_membership,
                cand_buckets=cand_buckets,
                finance_extra=finance_extra,
                word_freq=word_freq,
                max_cands=12,
                max_ed=1,
            )
            if not cands:
                continue

            # context (prev, next) skipping punct
            prev_tok = ""
            j = idx - 1
            while j >= 0:
                pt = seq[j]
                if pt and not _is_punct(pt):
                    prev_tok = pt
                    break
                j -= 1
            next_tok = ""
            k = idx + 1
            while k < len(seq):
                nt = seq[k]
                if nt and not _is_punct(nt):
                    next_tok = nt
                    break
                k += 1

            # Score original + candidates (logprob-ish)
            def local_score(prev_w: str, w_mid: str, next_w: str) -> float:
                if hasattr(lm_model, "score") and callable(getattr(lm_model, "score")) and not isinstance(lm_model, dict):
                    lp1 = score_bigram(w_mid, prev_w) if prev_w and w_mid else -50.0
                    lp2 = score_bigram(next_w, w_mid) if w_mid and next_w else 0.0
                    return lp1 + lp2
                return _local_score_ngram(lm_ngram, prev_w, w_mid, next_w)

            # Heuristic: Auxiliary verbs (is/was/have...) expect VBN/VBG, not VBZ/VBP
            aux_verbs = {"is", "are", "was", "were", "be", "been", "being", "have", "has", "had"}
            prev_is_aux = prev_tok.lower() in aux_verbs

            options = []
            
            # Score original
            orig_lp = local_score(prev_tok, tok_l, next_tok)
            if prev_is_aux and original_pos and str(original_pos) in {"VBZ", "VBP", "VB"}:
                # Penalize "was includes"
                orig_lp -= 2.0
            options.append((tok_l, orig_lp))

            for c in cands:
                # POS Check (already relaxed)
                if original_pos and not _pos_compatible(str(original_pos), c, pos_lexicon):
                    continue
                
                lp = local_score(prev_tok, c, next_tok)
                
                # Boost "was included"
                if prev_is_aux:
                    c_lex = pos_lexicon.get(c, {})
                    if any(t in {"VBN", "VBG"} for t in c_lex):
                        lp += 4.0
                
                options.append((c, lp))

            # pick best (excluding original) by log-score
            best_tok, best_lp = None, options[0][1]
            for w, lp in options[1:]:
                if lp > best_lp:
                    best_tok, best_lp = w, lp

            if not best_tok:
                continue

            # Softmax probabilities over the option set
            max_lp = max(lp for _, lp in options)
            exps = [math.exp(lp - max_lp) for _, lp in options]
            denom = sum(exps) if exps else 1.0
            probs = [e / denom for e in exps]
            orig_p = float(probs[0])
            # find prob of best_tok
            best_p = float(0.0)
            for (w, _), p in zip(options, probs):
                if w == best_tok:
                    best_p = float(p)
                    break
            gain = float(best_p - orig_p)
            ratio = float((best_p + 1e-9) / (orig_p + 1e-9))

            if (
                cfg.ngram_require_observed_bigram
                and isinstance(lm_ngram, dict)
                and ("bigram" in lm_ngram or "unigram" in lm_ngram)
            ):
                tok_freq = int(word_freq.get(tok_l, 0) or 0)
                best_freq = int(word_freq.get(best_tok, 0) or 0)
                rare_override = (
                    tok_freq > 0
                    and tok_freq <= int(cfg.ngram_rare_max_freq)
                    and best_freq >= float(cfg.ngram_rare_min_freq_ratio) * float(tok_freq)
                    and _levenshtein(tok_l, best_tok, max_dist=1) <= 1
                )
                observed = bool(_pack_bigram_count(lm_ngram, prev_tok, best_tok) > 0.0) or bool(
                    next_tok and _pack_bigram_count(lm_ngram, best_tok, next_tok) > 0.0
                )
                if not observed and not rare_override:
                    if gain < 0.45:
                        continue

            # If original token is rare (high idf), require larger gain.
            if apply_idf_guard:
                idf = _idf_of(tok_l)
                if idf is not None and idf >= (cfg.idf_protect_threshold - 0.5):
                    if gain < (cfg.min_gain + 0.08):
                        continue

            if cfg.min_freq_ratio and float(cfg.min_freq_ratio) > 1.0:
                tok_freq2 = int(word_freq.get(tok_l, 0) or 0)
                best_freq2 = int(word_freq.get(best_tok, 0) or 0)
                if tok_freq2 > 0 and best_freq2 > 0:
                    if float(best_freq2) < float(tok_freq2) * float(cfg.min_freq_ratio):
                        if gain < (cfg.min_gain + 0.15):
                            continue

            if best_p >= cfg.min_best_score and gain >= cfg.min_gain and ratio >= cfg.min_ratio:
                err = {
                    "token_index": idx,
                    "start": int(tokens[idx].get("start", -1)),
                    "end": int(tokens[idx].get("end", -1)),
                    "wrong": tok,
                    "correct": best_tok,
                    "type": "real-word",
                    "meta": {"backend": "ngram", "best_score": best_p, "gain": gain},
                }
                errors.append(err)
                sug_list = suggestions.setdefault(str(idx), [])
                sug_list.append({"token": best_tok, "score": float(best_p), "source": "ngram"})

    # sort errors by position for stable output
    errors.sort(key=lambda e: int(e.get("token_index", 0)))
    return errors, suggestions
