from __future__ import annotations

import json
import pickle
import time
import sys
from pathlib import Path
from typing import Dict, Any, Set

import nltk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ONLINE_DIR = Path(__file__).resolve().parent
if str(ONLINE_DIR) not in sys.path:
    sys.path.insert(0, str(ONLINE_DIR))

from preprocess.normalization import _normalize_unicode

try:
    from . import nonword_logic as nonword
    from . import realword_logic as realword
except Exception:
    import nonword_logic as nonword
    import realword_logic as realword


BASE_DIR = Path(__file__).resolve().parent
RES_DIR = BASE_DIR.parent / "resources"
MODELS_DIR = RES_DIR / "models"
DEFAULT_PACK_DIR = MODELS_DIR

DICT_VOCAB_PKL = "vocab_set.pkl"
DICT_FREQ_JSON = "word_freq.json"
DICT_SORTED_TXT = "sorted_vocab_list.txt"

LM_MODEL_PKL = "bigram_kn_model.pkl"
LM_VOCAB_PKL = "lm_vocab.pkl"
FIN_NGRAM_PKL = "fin_ngram.pkl"

CAND_BUCKETS_PKL = "candidate_buckets.pkl"

FINANCE_EXTRA_TXT = "finance_extra.txt"

SAVED_TEXTS: Dict[str, Dict[str, Any]] = {}

BASE_VOCAB_SET: Set[str] = set()
VOCAB_SET: Set[str] = set()
WORD_FREQ: Dict[str, int] = {}
SORTED_VOCAB = []

CAND_BUCKETS: Dict[str, Any] = {}
LM_MODEL = None
LM_VOCAB = None
FIN_NGRAM_PACK = None

FINANCE_EXTRA: Set[str] = set()
POS_LEXICON: Dict[str, Dict[str, int]] = {}


def _load_pickle(p: Path):
    with p.open("rb") as f:
        return pickle.load(f)


def _load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dictionary_pack(pack_dir: Path):
    vocab_set: Set[str] = set()
    word_freq: Dict[str, int] = {}
    sorted_vocab = []

    vocab_p = pack_dir / DICT_VOCAB_PKL
    if vocab_p.exists():
        vocab_set = _load_pickle(vocab_p)

    freq_p = pack_dir / DICT_FREQ_JSON
    if freq_p.exists():
        word_freq = _load_json(freq_p)

    sorted_p = pack_dir / DICT_SORTED_TXT
    if sorted_p.exists():
        sorted_vocab = [
            line.strip().split()[0]
            for line in sorted_p.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]

    if not sorted_vocab:
        if word_freq:
            sorted_vocab = [w for w, _ in sorted(word_freq.items(), key=lambda x: -x[1])]
        else:
            sorted_vocab = sorted(vocab_set)

    if not vocab_set and sorted_vocab:
        vocab_set = set(sorted_vocab)

    return vocab_set, word_freq, sorted_vocab


def load_candidate_pack(pack_dir: Path) -> Dict[str, Any]:
    p = pack_dir / CAND_BUCKETS_PKL
    if p.exists():
        return _load_pickle(p)
    return {}


def load_bigram_lm_pack(pack_dir: Path):
    model_p = pack_dir / LM_MODEL_PKL
    vocab_p = pack_dir / LM_VOCAB_PKL
    if not model_p.exists() or not vocab_p.exists():
        return None, None
    try:
        model = _load_pickle(model_p)
        vocab = _load_pickle(vocab_p)
    except Exception as e:
        print("Error loading LM pack:", e)
        return None, None
    return model, vocab


def load_pos_lexicon(pack_dir: Path) -> Dict[str, Dict[str, int]]:
    p = pack_dir / "pos_lexicon.json"
    if not p.exists():
        return {}
    return _load_json(p)


def init_packs(pack_dir: Path = DEFAULT_PACK_DIR):
    global BASE_VOCAB_SET, VOCAB_SET, WORD_FREQ, SORTED_VOCAB
    global CAND_BUCKETS, LM_MODEL, LM_VOCAB, FINANCE_EXTRA
    global POS_LEXICON, FIN_NGRAM_PACK

    pack_dir = pack_dir.resolve()

    POS_LEXICON = load_pos_lexicon(pack_dir)

    BASE_VOCAB_SET, WORD_FREQ, SORTED_VOCAB = load_dictionary_pack(pack_dir)
    VOCAB_SET = set(BASE_VOCAB_SET)

    CAND_BUCKETS = load_candidate_pack(pack_dir)
    LM_MODEL, LM_VOCAB = load_bigram_lm_pack(pack_dir)
    fin_ngram_p = pack_dir / FIN_NGRAM_PKL
    FIN_NGRAM_PACK = _load_pickle(fin_ngram_p) if fin_ngram_p.exists() else None

    if LM_VOCAB is not None:
        lm_words = {str(w).lower() for w in LM_VOCAB if isinstance(w, str)}
        VOCAB_SET |= lm_words

    fin_p = RES_DIR / "corpus" / FINANCE_EXTRA_TXT
    if fin_p.exists():
        FINANCE_EXTRA = {
            line.strip().lower()
            for line in fin_p.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        }
    else:
        FINANCE_EXTRA = set()

    print("=== DEBUG PACK DIR ===")
    print("PACK_DIR:", pack_dir)
    print("VOCAB_SIZE:", len(BASE_VOCAB_SET))
    print("FINANCE_VOCAB_SIZE:", len(FINANCE_EXTRA))
    print("HAS_LM:", LM_MODEL is not None)
    print("HAS_FIN_NGRAM:", FIN_NGRAM_PACK is not None)
    print("HAS_CAND:", bool(CAND_BUCKETS))


init_packs()


def build_runtime_vocab(settings: dict) -> Set[str]:
    use_finance = bool(settings.get("enable_finance_dictionary", False))
    vocab = set(VOCAB_SET)
    if use_finance and FINANCE_EXTRA:
        vocab |= set(FINANCE_EXTRA)
    return vocab


def ensure_pos_tagger():
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")


def attach_pos_tags(tokens):
    idxs = []
    words = []
    for i, t in enumerate(tokens):
        tok = t["token"]
        if nonword.should_skip_token(tok):
            continue
        idxs.append(i)
        words.append(tok)
    if not words:
        return
    tagged = nltk.pos_tag(words)
    for i, (_, pos) in zip(idxs, tagged):
        tokens[i]["pos"] = pos


def detect_and_suggest(text: str, settings: dict):
    settings = dict(settings or {})
    settings["enable_finance_dictionary"] = True

    vocab_set = build_runtime_vocab(settings)

    tokens = nonword.tokenize_with_spans(text)

    if settings.get("enable_pos_filter", True):
        ensure_pos_tagger()
        attach_pos_tags(tokens)

    t0 = time.perf_counter()
    errors, nonword_suggestions = nonword.detect_nonword_errors(
        tokens=tokens,
        settings=settings,
        vocab_set=vocab_set,
        cand_buckets=CAND_BUCKETS,
        word_freq=WORD_FREQ,
        pos_lexicon=POS_LEXICON,
    )
    t1 = time.perf_counter()
    print(f"Total Time of non-words: {t1 - t0:.6f} s")

    def make_error_id(er: Dict[str, Any]) -> str:
        return f'{int(er.get("start", -1))}:{int(er.get("end", -1))}'

    for er in errors:
        er["id"] = make_error_id(er)

    suggestions_map: Dict[str, Any] = {}
    for er in errors:
        w = er.get("word")
        if not w:
            continue
        ranked = nonword_suggestions.get(w)
        if ranked:
            suggestions_map[er["id"]] = ranked
            # Fallback: also store by word for robustness
            if w not in suggestions_map:
                suggestions_map[w] = ranked

    enable_real_word_check = settings.get("enable_real_word_check", False)
    if not enable_real_word_check:
        return {"errors": errors, "suggestions": suggestions_map, "text": text[:500]}

    t2 = time.perf_counter()
    use_finance = bool(settings.get("enable_finance_dictionary", False))
    realword_backend = str(settings.get("realword_backend", "ngram") or "").strip().lower()
    if realword_backend in {"n-gram", "ngram", "bigram"}:
        realword_backend = "ngram"
    elif realword_backend in {"mlm", "bert"}:
        realword_backend = "mlm"
    elif realword_backend in {"hybrid", "mix", "mixed"}:
        realword_backend = "hybrid"
    elif realword_backend in {"auto", ""}:
        realword_backend = "auto"

    if realword_backend == "mlm":
        lm_for_realword = LM_MODEL
    elif realword_backend == "ngram":
        lm_for_realword = FIN_NGRAM_PACK if FIN_NGRAM_PACK is not None else LM_MODEL
    else:
        if FIN_NGRAM_PACK is not None and LM_MODEL is not None:
            lm_for_realword = {"ngram": FIN_NGRAM_PACK, "mlm": LM_MODEL}
        else:
            lm_for_realword = FIN_NGRAM_PACK if FIN_NGRAM_PACK is not None else LM_MODEL
    rw_errors, rw_suggestions = realword.detect_realword_errors(
        tokens=tokens,
        settings=settings,
        vocab_set_for_candidates=vocab_set,
        vocab_set_for_membership=VOCAB_SET,
        finance_extra=FINANCE_EXTRA if use_finance else set(),
        word_freq=WORD_FREQ,
        pos_lexicon=POS_LEXICON,
        cand_buckets=CAND_BUCKETS,
        lm_model=lm_for_realword,
    )
    t3 = time.perf_counter()
    print(f"Total Time of real-words: {t3 - t2:.6f} s")

    for er in rw_errors:
        start = int(er.get("start", -1))
        end = int(er.get("end", -1))
        wrong = er.get("wrong", "")
        out = {
            "id": f"{start}:{end}",
            "word": wrong,
            "start": start,
            "end": end,
            "type": "real-word",
        }
        if "correct" in er:
            out["correct"] = er.get("correct")
        errors.append(out)

        tok_idx = er.get("token_index")
        if tok_idx is None:
            continue
        key = str(tok_idx)
        raw_sugs = rw_suggestions.get(key) or []
        casing = nonword.detect_casing(str(wrong))
        converted = []
        for s in raw_sugs:
            cand = s.get("candidate") or s.get("token") or s.get("suggestion")
            if not cand:
                continue
            dist = nonword.levenshtein(str(wrong), str(cand))
            rec = {
                "candidate": nonword.apply_casing(str(cand), casing),
                "edit_distance": dist,
            }
            if "score" in s:
                rec["score"] = s.get("score")
            if "source" in s:
                rec["source"] = s.get("source")
            converted.append(rec)
        if converted:
            suggestions_map[out["id"]] = converted
            # Fallback by word
            if wrong and wrong not in suggestions_map:
                suggestions_map[wrong] = converted

    errors.sort(key=lambda e: (int(e.get("start", -1)), int(e.get("end", -1))))

    return {"errors": errors, "suggestions": suggestions_map, "text": text[:500]}


if __name__ == "__main__":
    test = "According for Gran , the companys has no plans to move all production to Russia , although that is where the company is growing"
    settings = {
        "enable_auto_correction": False,
        "show_candidate_ranking": True,
        "enable_finance_dictionary": False,
        "show_confidence": False,
        "enable_real_word_check": True,
    }
    t0 = time.perf_counter()
    result = detect_and_suggest(text=test, settings=settings)
    t1 = time.perf_counter()
    print("Total Time:")
    print(f"{t1 - t0:.6f} s")
    print(result)
