from __future__ import annotations

import csv
import importlib.util
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\s]")


@dataclass
class InjectedError:
    token_index: int
    start: int
    end: int
    correct: str
    wrong: str
    error_type: str


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tokenize_with_spans(text: str) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []
    for m in WORD_RE.finditer(text):
        tokens.append({"token": m.group(0), "start": m.start(), "end": m.end()})
    return tokens


def is_alpha_token(tok: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+", tok))


def levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    a = a.lower()
    b = b.lower()
    n, m = len(a), len(b)
    if a == b:
        return 0
    if n == 0:
        return m
    if m == 0:
        return n
    if abs(n - m) > max_dist:
        return max_dist + 1
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ca = a[i - 1]
        for j in range(1, m + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def load_txt_dir_lines(corpus_dir: Path) -> List[str]:
    texts: List[str] = []
    if not corpus_dir.exists():
        return texts
    for fp in sorted(corpus_dir.glob("*.txt")):
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    return texts


def load_csv_texts(csv_path: Path, text_col: str = "text", csv_no_header: bool = False) -> List[str]:
    if not csv_path.exists():
        return []
    if csv_no_header:
        df = pd.read_csv(csv_path, encoding="latin-1", header=None, names=["label", "text"])
        text_col = "text"
    else:
        df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {csv_path}. Available: {list(df.columns)}")
    return df[text_col].dropna().astype(str).tolist()


def load_vocab_set(dict_dir: Path) -> set:
    vocab_pkl = dict_dir / "vocab_set.pkl"
    freq_json = dict_dir / "word_freq.json"
    if vocab_pkl.exists():
        import pickle
        with vocab_pkl.open("rb") as f:
            vocab = pickle.load(f)
        return set(vocab)
    if freq_json.exists():
        with freq_json.open("r", encoding="utf-8") as f:
            freq = json.load(f)
        return set(freq.keys())
    raise FileNotFoundError(f"Cannot find vocab_set.pkl or word_freq.json in {dict_dir}")


def build_vocab_buckets(vocab: Iterable[str]) -> Dict[str, Dict[int, List[str]]]:
    buckets: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for w in vocab:
        if not w:
            continue
        wl = str(w).lower()
        if len(wl) < 2:
            continue
        if not re.fullmatch(r"[a-z]+", wl):
            continue
        buckets[wl[0]][len(wl)].append(wl)
    for fc in list(buckets.keys()):
        for L in list(buckets[fc].keys()):
            buckets[fc][L].sort()
    return {fc: dict(inner) for fc, inner in buckets.items()}


def corrupt_word_small(word: str) -> str:
    if len(word) < 3:
        return word
    first = word[0]
    tail = word[1:]
    op = random.choice(["sub", "swap", "del", "ins"])
    if op == "sub":
        i = random.randrange(len(tail))
        choices = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != tail[i].lower()]
        return first + tail[:i] + random.choice(choices) + tail[i + 1:]
    if op == "swap" and len(tail) >= 2:
        i = random.randrange(len(tail) - 1)
        swapped = list(tail)
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        return first + "".join(swapped)
    if op == "del" and len(tail) >= 2:
        i = random.randrange(len(tail))
        return first + tail[:i] + tail[i + 1:]
    if op == "ins":
        i = random.randrange(len(tail) + 1)
        ins = random.choice("abcdefghijklmnopqrstuvwxyz")
        return first + tail[:i] + ins + tail[i:]
    return word


def pick_real_word_replacement(word: str, vocab: set, buckets: Dict[str, Dict[int, List[str]]], max_ed: int = 1) -> Optional[str]:
    w = word.lower()
    if not re.fullmatch(r"[a-z]+", w):
        return None
    if w not in vocab:
        return None
    fc = w[0]
    if fc not in buckets:
        return None
    cand_pool: List[str] = []
    for d in range(-max_ed, max_ed + 1):
        L = len(w) + d
        if L <= 1:
            continue
        cand_pool.extend(buckets[fc].get(L, []))
    random.shuffle(cand_pool)
    for cand in cand_pool[:800]:
        if cand == w:
            continue
        if levenshtein(w, cand, max_dist=max_ed) <= max_ed:
            return cand
    return None


def apply_replacements_and_track_spans(
    original: str,
    token_spans: List[Dict[str, Any]],
    replacements: Dict[int, str],
) -> Tuple[str, Dict[int, Tuple[int, int]]]:
    pieces: List[str] = []
    new_spans: Dict[int, Tuple[int, int]] = {}
    cursor = 0
    out_len = 0
    for i, t in enumerate(token_spans):
        s = int(t["start"])
        e = int(t["end"])
        gap = original[cursor:s]
        pieces.append(gap)
        out_len += len(gap)
        tok_out = replacements.get(i, original[s:e])
        if i in replacements:
            new_spans[i] = (out_len, out_len + len(tok_out))
        pieces.append(tok_out)
        out_len += len(tok_out)
        cursor = e
    pieces.append(original[cursor:])
    return "".join(pieces), new_spans


def generate_test_file(
    texts: List[str],
    vocab: set,
    out_dir: Path,
    error_mode: str,
    num_samples: int,
    min_errors: int,
    max_errors: int,
    seed: int,
) -> Path:
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = build_vocab_buckets(vocab)
    fname = f"test_{error_mode}_n{num_samples}_e{min_errors}-{max_errors}_seed{seed}.jsonl"
    out_path = out_dir / fname
    with out_path.open("w", encoding="utf-8") as f:
        written = 0
        for raw in texts:
            if written >= num_samples:
                break
            sent = str(raw).strip()
            if len(sent) < 20:
                continue
            toks = tokenize_with_spans(sent)
            alpha_idxs = [i for i, t in enumerate(toks) if is_alpha_token(t["token"]) and len(t["token"]) >= 4]
            if len(alpha_idxs) < 3:
                continue
            k = random.randint(min_errors, max_errors)
            chosen = random.sample(alpha_idxs, k=min(k, len(alpha_idxs)))
            replacements: Dict[int, str] = {}
            gt_errors: List[InjectedError] = []
            wrong_surface_set: set = set()

            for ti in chosen:
                orig_tok = toks[ti]["token"]
                orig_lower = orig_tok.lower()

                submode = error_mode
                if error_mode == "mixed":
                    submode = random.choice(["nonword", "realword"])

                if submode == "nonword":
                    new_tok = orig_tok
                    for _ in range(60):
                        cand = corrupt_word_small(orig_tok)
                        if cand and cand[0].lower() != orig_tok[0].lower():
                            continue
                        if cand.lower() == orig_lower:
                            continue
                        if cand.lower() in vocab:
                            continue
                        if cand in wrong_surface_set:
                            continue
                        new_tok = cand
                        break
                    if new_tok.lower() == orig_lower or new_tok.lower() in vocab:
                        continue
                    replacements[ti] = new_tok
                    wrong_surface_set.add(new_tok)
                    gt_errors.append(InjectedError(ti, -1, -1, orig_tok, new_tok, "non-word"))

                elif submode == "realword":
                    rep = pick_real_word_replacement(orig_tok, vocab, buckets, max_ed=1)
                    if rep is None:
                        continue
                    rep_out = rep.capitalize() if (orig_tok[0].isupper() and orig_tok[1:].islower()) else rep
                    if rep_out in wrong_surface_set:
                        continue
                    replacements[ti] = rep_out
                    wrong_surface_set.add(rep_out)
                    gt_errors.append(InjectedError(ti, -1, -1, orig_tok, rep_out, "real-word"))

                else:
                    raise ValueError("error_mode must be one of: nonword, realword, mixed")

            if not gt_errors:
                continue

            corrupted, new_spans = apply_replacements_and_track_spans(sent, toks, replacements)

            final_errors: List[Dict[str, Any]] = []
            for err in gt_errors:
                if err.token_index not in new_spans:
                    continue
                ns, ne = new_spans[err.token_index]
                final_errors.append(
                    {
                        "token_index": err.token_index,
                        "start": ns,
                        "end": ne,
                        "correct": err.correct,
                        "wrong": err.wrong,
                        "type": err.error_type,
                    }
                )

            if not final_errors:
                continue

            record = {"sentence": corrupted, "errors": final_errors, "meta": {"error_mode": error_mode, "num_errors": len(final_errors)}}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return out_path


def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def reciprocal_rank(candidates: List[Dict[str, Any]], correct: str) -> float:
    target = correct.lower()
    for i, it in enumerate(candidates, start=1):
        # Be tolerant to different candidate field names (nonword uses
        # `candidate`, realword may keep `suggestion`).
        cand = it.get("candidate", it.get("suggestion", ""))
        if str(cand).lower() == target:
            return 1.0 / i
    return 0.0


def hit_at_k(candidates: List[Dict[str, Any]], correct: str, k: int) -> int:
    target = correct.lower()
    for it in candidates[:k]:
        cand = it.get("candidate", it.get("suggestion", ""))
        if str(cand).lower() == target:
            return 1
    return 0


def _safe_slice(text: str, start: int, end: int) -> str:
    s = max(0, min(int(start), len(text)))
    e = max(0, min(int(end), len(text)))
    if e < s:
        s, e = e, s
    return text[s:e]


def _snippet(text: str, start: int, end: int, window: int = 55) -> str:
    s = max(0, int(start) - window)
    e = min(len(text), int(end) + window)
    prefix = "..." if s > 0 else ""
    suffix = "..." if e < len(text) else ""
    return prefix + text[s:e] + suffix


def evaluate_test_file(
    test_file: Path,
    service_py: Path,
    settings_base: Dict[str, Any],
    pos_on: bool,
    plots_out_dir: Path,
    csv_out_dir: Path,
    diagnostics_out_dir: Optional[Path] = None,
    max_examples: int = 200,
    progress_every: int = 250,
) -> Dict[str, Any]:
    service = load_module_from_path("service_module", service_py)
    if not hasattr(service, "detect_and_suggest"):
        raise AttributeError("service.py must define detect_and_suggest(text, settings)")
    detect_and_suggest = getattr(service, "detect_and_suggest")

    settings = dict(settings_base)
    settings["enable_pos_filter"] = bool(pos_on)

    tp = fp = fn = 0
    top1_hits = top3_hits = top5_hits = 0
    mrrs: List[float] = []
    latencies: List[float] = []
    gt_total = 0
    matched_total = 0

    fp_word_counter: Counter[str] = Counter()
    fn_word_counter: Counter[str] = Counter()
    fp_examples: List[Dict[str, Any]] = []
    fn_examples: List[Dict[str, Any]] = []

    if diagnostics_out_dir is None:
        diagnostics_out_dir = csv_out_dir

    diagnostics_out_dir.mkdir(parents=True, exist_ok=True)

    with test_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            print(line_no,line)
            if progress_every and line_no % progress_every == 0:
                print(f"[{test_file.name}] POS={'ON' if pos_on else 'OFF'} processed {line_no} lines...")

            if not line.strip():
                continue

            rec = json.loads(line)
            sent = rec["sentence"]
            gt = rec["errors"]

            gt_spans = {(int(e["start"]), int(e["end"])) for e in gt}
            gt_total += len(gt_spans)

            t0 = time.perf_counter()
            out = detect_and_suggest(sent, settings)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

            pred_errors = out.get("errors", [])
            pred_spans = {(int(e["start"]), int(e["end"])) for e in pred_errors}

            tp_here = len(gt_spans & pred_spans)
            fp_here = len(pred_spans - gt_spans)
            fn_here = len(gt_spans - pred_spans)
            tp += tp_here
            fp += fp_here
            fn += fn_here

            suggestions = out.get("suggestions", {})

            pred_span_to_word: Dict[Tuple[int, int], str] = {}
            pred_span_to_index: Dict[Tuple[int, int], int] = {}
            for pe in pred_errors:
                ps = int(pe["start"])
                pe_ = int(pe["end"])
                w = str(pe.get("word") or _safe_slice(sent, ps, pe_))
                pred_span_to_word[(ps, pe_)] = w
                # Real-word module returns token_index (int); non-word may not.
                if pe.get("token_index") is not None:
                    try:
                        pred_span_to_index[(ps, pe_)] = int(pe.get("token_index"))
                    except Exception:
                        pass

            fp_spans = pred_spans - gt_spans
            for (s, e) in fp_spans:
                w = _safe_slice(sent, s, e)
                wl = w.lower()
                fp_word_counter[wl] += 1
                if len(fp_examples) < max_examples:
                    fp_examples.append(
                        {
                            "line": line_no,
                            "span": [s, e],
                            "word": w,
                            "snippet": _snippet(sent, s, e),
                        }
                    )

            fn_spans = gt_spans - pred_spans
            if fn_spans:
                gt_map = {(int(e["start"]), int(e["end"])): e for e in gt}
                for (s, e) in fn_spans:
                    ge = gt_map.get((s, e), {})
                    wrong = str(ge.get("wrong") or _safe_slice(sent, s, e))
                    correct = str(ge.get("correct") or "")
                    fn_word_counter[wrong.lower()] += 1
                    if len(fn_examples) < max_examples:
                        fn_examples.append(
                            {
                                "line": line_no,
                                "span": [s, e],
                                "wrong": wrong,
                                "correct": correct,
                                "snippet": _snippet(sent, s, e),
                            }
                        )

            for e in gt:
                span = (int(e["start"]), int(e["end"]))
                if span not in pred_spans:
                    continue
                matched_total += 1
                correct = str(e["correct"])
                wrong_surface = _safe_slice(sent, span[0], span[1])
                wrong_word = pred_span_to_word.get(span, wrong_surface)
                error_id = f"{span[0]}:{span[1]}"
                idx_key = None
                if span in pred_span_to_index:
                    idx_key = str(pred_span_to_index[span])
                if error_id in suggestions:
                    cand_list = suggestions.get(error_id, [])
                elif idx_key is not None and idx_key in suggestions:
                    cand_list = suggestions.get(idx_key, [])
                else:
                    cand_list = suggestions.get(wrong_word, [])
                if isinstance(cand_list, list):
                    top1_hits += hit_at_k(cand_list, correct, 1)
                    top3_hits += hit_at_k(cand_list, correct, 3)
                    top5_hits += hit_at_k(cand_list, correct, 5)
                    mrrs.append(reciprocal_rank(cand_list, correct))
                else:
                    mrrs.append(0.0)

    p, r, f1 = prf1(tp, fp, fn)
    top1 = top1_hits / matched_total if matched_total > 0 else 0.0
    top3 = top3_hits / matched_total if matched_total > 0 else 0.0
    top5 = top5_hits / matched_total if matched_total > 0 else 0.0
    mrr = statistics.mean(mrrs) if mrrs else 0.0

    mean_latency = statistics.mean(latencies) if latencies else 0.0
    lat_sorted = sorted(latencies)
    p50 = lat_sorted[int(0.50 * (len(lat_sorted) - 1))] if lat_sorted else 0.0
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))] if lat_sorted else 0.0

    summary = {
        "test_file": str(test_file),
        "pos_on": pos_on,
        "detection": {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1},
        "correction": {"matched": matched_total, "top1": top1, "top3": top3, "top5": top5, "mrr": mrr},
        "latency_sec": {"mean": mean_latency, "p50": p50, "p95": p95, "n": len(latencies)},
        "gt_errors": gt_total,
        "diagnostics": {
            "fp_words_top": fp_word_counter.most_common(50),
            "fn_words_top": fn_word_counter.most_common(50),
        },
    }

    plots_out_dir.mkdir(parents=True, exist_ok=True)
    csv_out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_out_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = csv_out_dir / f"summary_{test_file.stem}_pos{int(pos_on)}.json"
    with summary_json_path.open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, ensure_ascii=False, indent=2)

    summary_csv_path = csv_out_dir / f"summary_{test_file.stem}_pos{int(pos_on)}.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["test_file", "pos_on", "precision", "recall", "f1", "top1", "top3", "top5", "mrr", "lat_mean", "lat_p50", "lat_p95", "n_sent", "gt_errors", "matched"])
        w.writerow([str(test_file), int(pos_on), p, r, f1, top1, top3, top5, mrr, mean_latency, p50, p95, len(latencies), gt_total, matched_total])

    diag_prefix = f"{test_file.stem}_pos{int(pos_on)}"
    fp_words_path = diagnostics_out_dir / f"fp_words_{diag_prefix}.csv"
    with fp_words_path.open("w", newline="", encoding="utf-8") as fpc:
        w = csv.writer(fpc)
        w.writerow(["word", "count"])
        for word, cnt in fp_word_counter.most_common():
            w.writerow([word, cnt])

    fn_words_path = diagnostics_out_dir / f"fn_words_{diag_prefix}.csv"
    with fn_words_path.open("w", newline="", encoding="utf-8") as fnc:
        w = csv.writer(fnc)
        w.writerow(["wrong_word", "count"])
        for word, cnt in fn_word_counter.most_common():
            w.writerow([word, cnt])

    fp_ex_path = diagnostics_out_dir / f"fp_examples_{diag_prefix}.jsonl"
    with fp_ex_path.open("w", encoding="utf-8") as fpex:
        for it in fp_examples:
            fpex.write(json.dumps(it, ensure_ascii=False) + "\n")

    fn_ex_path = diagnostics_out_dir / f"fn_examples_{diag_prefix}.jsonl"
    with fn_ex_path.open("w", encoding="utf-8") as fnex:
        for it in fn_examples:
            fnex.write(json.dumps(it, ensure_ascii=False) + "\n")

    plt.figure()
    plt.bar(["precision", "recall", "f1"], [p, r, f1])
    plt.ylim(0, 1)
    plt.title(f"Detection ({test_file.stem}) - POS={'ON' if pos_on else 'OFF'}")
    plt.tight_layout()
    plt.savefig(plots_out_dir / f"det_{test_file.stem}_pos{int(pos_on)}.png", dpi=200)
    plt.close()

    plt.figure()
    plt.bar(["top1", "top3", "top5", "mrr"], [top1, top3, top5, mrr])
    plt.ylim(0, 1)
    plt.title(f"Correction ({test_file.stem}) - POS={'ON' if pos_on else 'OFF'}")
    plt.tight_layout()
    plt.savefig(plots_out_dir / f"cor_{test_file.stem}_pos{int(pos_on)}.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(latencies, bins=30)
    plt.title(f"Latency ({test_file.stem}) - POS={'ON' if pos_on else 'OFF'}")
    plt.xlabel("seconds")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plots_out_dir / f"lat_{test_file.stem}_pos{int(pos_on)}.png", dpi=200)
    plt.close()

    plt.figure()
    plt.bar(["FP", "FN"], [fp, fn])
    plt.title(f"Diagnostics FP/FN counts ({test_file.stem}) - POS={'ON' if pos_on else 'OFF'}")
    plt.tight_layout()
    plt.savefig(plots_out_dir / f"diag_fpfn_{test_file.stem}_pos{int(pos_on)}.png", dpi=200)
    plt.close()

    return summary


def compare_pos_on_off(summary_off: Dict[str, Any], summary_on: Dict[str, Any], plots_out_dir: Path) -> None:
    plots_out_dir.mkdir(parents=True, exist_ok=True)

    det_names = ["precision", "recall", "f1"]
    det_off = [summary_off["detection"][k] for k in det_names]
    det_on = [summary_on["detection"][k] for k in det_names]
    x = list(range(len(det_names)))

    plt.figure()
    plt.bar([i - 0.2 for i in x], det_off, width=0.4, label="POS OFF")
    plt.bar([i + 0.2 for i in x], det_on, width=0.4, label="POS ON")
    plt.xticks(x, det_names)
    plt.ylim(0, 1)
    plt.title(f"Detection POS OFF vs ON ({Path(summary_off['test_file']).stem})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_out_dir / f"cmp_det_{Path(summary_off['test_file']).stem}.png", dpi=200)
    plt.close()

    cor_names = ["top1", "top3", "top5", "mrr"]
    cor_off = [summary_off["correction"][k] for k in cor_names]
    cor_on = [summary_on["correction"][k] for k in cor_names]
    x2 = list(range(len(cor_names)))

    plt.figure()
    plt.bar([i - 0.2 for i in x2], cor_off, width=0.4, label="POS OFF")
    plt.bar([i + 0.2 for i in x2], cor_on, width=0.4, label="POS ON")
    plt.xticks(x2, cor_names)
    plt.ylim(0, 1)
    plt.title(f"Correction POS OFF vs ON ({Path(summary_off['test_file']).stem})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_out_dir / f"cmp_cor_{Path(summary_off['test_file']).stem}.png", dpi=200)
    plt.close()

    lat_off = summary_off["latency_sec"]["mean"]
    lat_on = summary_on["latency_sec"]["mean"]

    plt.figure()
    plt.bar(["POS OFF", "POS ON"], [lat_off, lat_on])
    plt.title(f"Mean latency POS OFF vs ON ({Path(summary_off['test_file']).stem})")
    plt.ylabel("seconds")
    plt.tight_layout()
    plt.savefig(plots_out_dir / f"cmp_lat_{Path(summary_off['test_file']).stem}.png", dpi=200)
    plt.close()


def main(
    cmd: str,
    corpus_dir: str = "../resources/models",
    corpus_dir_normal: str = "../resources/corpus/normal",
    corpus_dir_financial: str = "../resources/corpus/financial",
    corpus_csv: str = "../resources/corpus/financial/all-data.csv",
    text_col: str = "text",
    csv_no_header: bool = True,
    dict_dir: str = "../resources/models",
    out_dir: str = "../resources/test",
    error_mode: str = "nonword",
    num_samples: int = 500,
    min_errors: int = 1,
    max_errors: int = 1,
    seed: int = 42,
    test_file: str = "",
    service_py: str = "",
    settings_json: Optional[str] = None,
    enable_finance_dictionary: bool = False,
    enable_real_word_check: bool = False,
    plots_out_dir: str = str(Path("resources") / "test" / "plots"),
    csv_out_dir: str = str(Path("resources") / "test" / "reports"),
    diagnostics_out_dir: str = str(Path("resources") / "test" / "diagnostics"),
) -> None:
    cmd = (cmd or "").strip().lower()
    if cmd not in {"generate", "evaluate"}:
        raise ValueError("cmd must be 'generate' or 'evaluate'")

    if cmd == "generate":
        texts: List[str] = []
        if corpus_dir:
            texts.extend(load_txt_dir_lines(Path(corpus_dir)))
        if corpus_dir_normal:
            texts.extend(load_txt_dir_lines(Path(corpus_dir_normal)))
        if corpus_dir_financial:
            texts.extend(load_txt_dir_lines(Path(corpus_dir_financial)))
        if corpus_csv:
            texts.extend(load_csv_texts(Path(corpus_csv), text_col=text_col, csv_no_header=csv_no_header))
        if not texts:
            raise ValueError("No texts loaded. Check your corpus paths.")
        vocab = load_vocab_set(Path(dict_dir))
        out_path = generate_test_file(
            texts=texts,
            vocab=vocab,
            out_dir=Path(out_dir),
            error_mode=error_mode,
            num_samples=num_samples,
            min_errors=min_errors,
            max_errors=max_errors,
            seed=seed,
        )
        print(f"Generated: {out_path}")
        return

    if cmd == "evaluate":
        if not test_file:
            raise ValueError("When cmd='evaluate', test_file must be provided.")
        if not service_py:
            raise ValueError("When cmd='evaluate', service_py must be provided.")

        settings_base: Dict[str, Any] = {
            "enable_auto_correction": False,
            "show_candidate_ranking": True,
            "enable_finance_dictionary": bool(enable_finance_dictionary),
            "show_confidence": False,
            "enable_real_word_check": bool(enable_real_word_check),
        }
        if settings_json:
            with Path(settings_json).open("r", encoding="utf-8") as f:
                settings_base.update(json.load(f))

        # ------------------------------
        # EVALUATE
        # ------------------------------
        # For nonword: POS filter is irrelevant. Run ONCE only (POS OFF),
        # keep outputs in the same directories (reports/diagnostics/plots).
        if error_mode.strip().lower() == "nonword":
            print("Testing NONWORD (single run, POS ignored -> OFF)")
            summary = evaluate_test_file(
                test_file=Path(test_file),
                service_py=Path(service_py),
                settings_base=settings_base,
                pos_on=False,  # single run
                plots_out_dir=Path(plots_out_dir),
                csv_out_dir=Path(csv_out_dir),
                diagnostics_out_dir=Path(diagnostics_out_dir),
            )

            print("NONWORD (POS ignored):", summary["detection"], summary["correction"], summary["latency_sec"])
            print(f"Diagnostics written to: {diagnostics_out_dir}")
            return

        # For realword / mixed: keep original behavior (POS OFF vs ON + comparisons)
        print("Testing POS off")
        summary_off = evaluate_test_file(
            test_file=Path(test_file),
            service_py=Path(service_py),
            settings_base=settings_base,
            pos_on=False,
            plots_out_dir=Path(plots_out_dir),
            csv_out_dir=Path(csv_out_dir),
            diagnostics_out_dir=Path(diagnostics_out_dir),
        )

        print("Testing POS on")
        summary_on = evaluate_test_file(
            test_file=Path(test_file),
            service_py=Path(service_py),
            settings_base=settings_base,
            pos_on=True,
            plots_out_dir=Path(plots_out_dir),
            csv_out_dir=Path(csv_out_dir),
            diagnostics_out_dir=Path(diagnostics_out_dir),
        )

        compare_pos_on_off(summary_off, summary_on, plots_out_dir=Path(plots_out_dir))

        print("POS OFF:", summary_off["detection"], summary_off["correction"], summary_off["latency_sec"])
        print("POS  ON:", summary_on["detection"], summary_on["correction"], summary_on["latency_sec"])
        print(f"Diagnostics written to: {diagnostics_out_dir}")
        return


if __name__ == "__main__":
    #cmd = "generate"
    cmd = "evaluate"
    project_root = Path(__file__).resolve().parents[1]
    corpus_dir = str(project_root / "resources" / "models")
    corpus_dir_normal = str(project_root / "resources" / "corpus" / "normal")
    corpus_dir_financial = str(project_root / "resources" / "corpus" / "financial")
    corpus_csv = str(project_root / "resources" / "corpus" / "financial" / "all-data.csv")
    text_col = "text"
    csv_no_header = True
    dict_dir = str(project_root / "resources" / "models")
    out_dir = str(project_root / "resources" / "test")
    error_mode = "nonword"
    #error_mode = "realword"
    num_samples = 5000
    min_errors = 1
    max_errors = 3
    seed = 42

    test_file = str(project_root / "resources" / "test" / f"test_{error_mode}_n{num_samples}_e{min_errors}-{max_errors}_seed{seed}.jsonl")
    service_py = str(Path(__file__).resolve().parent / "service.py")
    settings_json = None
    enable_finance_dictionary = True
    enable_real_word_check = False
    plots_out_dir = str(project_root / "resources" / "test" / "plots")
    csv_out_dir = str(project_root / "resources" / "test" / "reports")
    diagnostics_out_dir = str(project_root / "resources" / "test" / "diagnostics")

    main(
        cmd=cmd,
        corpus_dir=corpus_dir,
        corpus_dir_normal=corpus_dir_normal,
        corpus_dir_financial=corpus_dir_financial,
        corpus_csv=corpus_csv,
        text_col=text_col,
        csv_no_header=csv_no_header,
        dict_dir=dict_dir,
        out_dir=out_dir,
        error_mode=error_mode,
        num_samples=num_samples,
        min_errors=min_errors,
        max_errors=max_errors,
        seed=seed,
        test_file=test_file,
        service_py=service_py,
        settings_json=settings_json,
        enable_finance_dictionary=enable_finance_dictionary,
        enable_real_word_check=enable_real_word_check,
        plots_out_dir=plots_out_dir,
        csv_out_dir=csv_out_dir,
        diagnostics_out_dir=diagnostics_out_dir,
    )
