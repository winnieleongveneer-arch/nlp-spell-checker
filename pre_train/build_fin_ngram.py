#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_fin_ngram.py

Build a lightweight n-gram (default: bigram) statistics pack from the provided
financial corpus files (Financial PhraseBank style).

- Reads .txt lines like: "<sentence>@neutral" / "@positive" / "@negative"
- Reads all-data.csv with records like: label,"sentence"

Outputs a pickle containing:
{
  "version": 1,
  "order": 2,
  "tokenization": {...},
  "unigram": Counter-like dict token->count,
  "bigram": dict (prev, tok)->count   (if order>=2)
  "total_unigram": int,
  "vocab": set/list,
  "meta": {"num_sentences":..., "source_files":[...]}
}

This pack is designed to be used as a *trigger* / *filter* in real-word
spell correction to:
- pick top-K suspicious positions quickly
- reduce MLM calls -> lower p95 latency
- suppress FP by checking whether finance-LM prefers the original

No external dependencies; runs on Windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

LABEL_SUFFIX_RE = re.compile(r"@(?:positive|negative|neutral)\s*$", re.IGNORECASE)
WS_RE = re.compile(r"\s+")
# Very lightweight tokenization: words / numbers / punctuation
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+(?:\.[0-9]+)?|[%$€£¥]|[^\w\s]", re.UNICODE)


def _read_text_lines(path: Path) -> Iterable[str]:
    # PhraseBank txts are sometimes latin-1-ish; ignore undecodable bytes.
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line


def _iter_sentences_from_txt(path: Path) -> Iterable[str]:
    for line in _read_text_lines(path):
        # Strip label suffix if present.
        line = LABEL_SUFFIX_RE.sub("", line).strip()
        if line:
            yield line


def _iter_sentences_from_csv(path: Path) -> Iterable[str]:
    """
    Robust CSV reader for files that may use \\r as line separators.
    Expected format: label,"sentence"
    """
    # Read bytes and normalize newlines.
    data = path.read_bytes()
    # Normalize CR-only to LF to satisfy csv.reader on some files.
    text = data.decode("latin-1", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
    # Use csv.reader so commas inside quotes are handled.
    reader = csv.reader(text.splitlines())
    for row in reader:
        if not row:
            continue
        if len(row) == 1:
            # Sometimes malformed: try split once
            parts = row[0].split(",", 1)
            if len(parts) == 2:
                row = [parts[0], parts[1]]
        if len(row) < 2:
            continue
        label = row[0].strip()
        sent = row[1].strip().strip('"')
        if not sent:
            continue
        yield sent


def _iter_sentences_from_json(path: Path) -> Iterable[str]:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore") or "null")
    items = data if isinstance(data, list) else [data]
    for obj in items:
        if not isinstance(obj, dict):
            continue
        parts: List[str] = []
        for k in ("title", "description", "content", "text", "summary"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        if not parts:
            continue
        yield " ".join(parts)

def _iter_sentences_from_jsonl(path: Path) -> Iterable[str]:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        parts: List[str] = []
        for k in ("title", "description", "content", "text", "summary"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        if not parts:
            continue
        yield " ".join(parts)


def tokenize(
    s: str,
    *,
    lowercase: bool = True,
    split_hyphen: bool = True,
    normalize_numbers: bool = True,
    keep_case_for_tickers: bool = True,
) -> List[str]:
    """
    Tokenize a sentence into tokens suited for n-gram trigger usage.

    - Splits hyphenated compounds into separate tokens if split_hyphen=True.
    - Normalizes numbers to <NUM> if normalize_numbers=True.
    - Lowercases tokens unless token looks like a ticker/abbrev (ALLCAPS, len<=6)
      and keep_case_for_tickers=True.
    """
    s = s.strip()
    if not s:
        return []

    # Normalize common PhraseBank artifacts
    s = s.replace("@-@", "-")
    s = WS_RE.sub(" ", s)

    toks: List[str] = []
    for m in TOKEN_RE.finditer(s):
        tok = m.group(0)

        if split_hyphen and tok == "-":
            # keep '-' as token boundary marker; caller may ignore
            toks.append("-")
            continue

        if normalize_numbers and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", tok):
            toks.append("<NUM>")
            continue

        if lowercase:
            if keep_case_for_tickers and tok.isupper() and 1 <= len(tok) <= 6 and tok.isalpha():
                toks.append(tok)  # keep
            else:
                toks.append(tok.lower())
        else:
            toks.append(tok)

    if split_hyphen:
        # Turn "a - b" into ["a","b"] by removing '-' tokens.
        toks = [t for t in toks if t != "-"]

    return toks


def build_ngram_pack(
    sentences: Iterable[str],
    *,
    order: int = 2,
    lowercase: bool = True,
    split_hyphen: bool = True,
    normalize_numbers: bool = True,
    keep_case_for_tickers: bool = True,
    add_sentence_markers: bool = True,
) -> Dict:
    if order < 1 or order > 3:
        raise ValueError("order must be 1..3 (recommended 2 for trigger)")

    uni = Counter()
    bi = Counter()
    tri = Counter()

    num_sent = 0
    for s in sentences:
        toks = tokenize(
            s,
            lowercase=lowercase,
            split_hyphen=split_hyphen,
            normalize_numbers=normalize_numbers,
            keep_case_for_tickers=keep_case_for_tickers,
        )
        if not toks:
            continue
        num_sent += 1

        if add_sentence_markers:
            seq = ["<s>"] + toks + ["</s>"]
        else:
            seq = toks

        uni.update(seq)

        if order >= 2:
            bi.update(zip(seq[:-1], seq[1:]))

        if order >= 3:
            tri.update(zip(seq[:-2], seq[1:-1], seq[2:]))

    vocab = set(uni.keys())
    pack = {
        "version": 1,
        "order": order,
        "tokenization": {
            "lowercase": lowercase,
            "split_hyphen": split_hyphen,
            "normalize_numbers": normalize_numbers,
            "keep_case_for_tickers": keep_case_for_tickers,
            "add_sentence_markers": add_sentence_markers,
            "regex": TOKEN_RE.pattern,
        },
        "unigram": dict(uni),
        "total_unigram": int(sum(uni.values())),
        "vocab": sorted(vocab),
        "meta": {
            "num_sentences": num_sent,
        },
    }
    if order >= 2:
        # Store as dict with "prev\ttok" keys to be pickle/json friendly
        pack["bigram"] = {f"{a}\t{b}": int(c) for (a, b), c in bi.items()}
    if order >= 3:
        pack["trigram"] = {f"{a}\t{b}\t{c}": int(cnt) for (a, b, c), cnt in tri.items()}
    return pack


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build lightweight financial n-gram statistics pack (no kenlm required).")
    ap.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing corpus files (txt/csv). e.g. ../resources/corpus/financial",
    )
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output pickle path. e.g. ../resources/models/fin_ngram.pkl",
    )
    ap.add_argument(
        "--order",
        type=int,
        default=2,
        help="n-gram order (1..3). Default: 2 (bigram).",
    )
    ap.add_argument(
        "--use_files",
        type=str,
        default="",
        help="Optional comma-separated basenames to include. If empty, includes all *.txt and *.csv in input_dir.",
    )
    ap.add_argument(
        "--extra_json",
        type=str,
        default="",
        help="Optional JSON/JSONL path to include (e.g. resources/corpus/financial/polygon_news_sample.json).",
    )
    ap.add_argument("--no_lowercase", action="store_true", help="Disable lowercasing (not recommended).")
    ap.add_argument("--no_split_hyphen", action="store_true", help="Do not split hyphenated compounds.")
    ap.add_argument("--no_normalize_numbers", action="store_true", help="Do not normalize numbers to <NUM>.")
    ap.add_argument("--no_sentence_markers", action="store_true", help="Do not add <s>/</s> markers.")
    args = ap.parse_args(argv)

    in_dir = Path(args.input_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"input_dir not found or not a directory: {in_dir}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    allowed = None
    if args.use_files.strip():
        allowed = {x.strip() for x in args.use_files.split(",") if x.strip()}

    files: List[Path] = []
    for p in sorted(in_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".txt", ".csv", ".json"}:
            continue
        if allowed is not None and p.name not in allowed:
            continue
        files.append(p)

    if not files:
        raise SystemExit(f"No .txt/.csv files found in {in_dir} (use_files={args.use_files!r}).")

    sentences: List[str] = []
    used_files: List[str] = []
    for p in files:
        used_files.append(p.name)
        if p.suffix.lower() == ".txt":
            sentences.extend(list(_iter_sentences_from_txt(p)))
        elif p.suffix.lower() == ".csv":
            sentences.extend(list(_iter_sentences_from_csv(p)))
        else:
            sentences.extend(list(_iter_sentences_from_json(p)))

    extra_json = Path(args.extra_json).resolve() if args.extra_json.strip() else None
    if extra_json and extra_json.exists() and extra_json.is_file():
        used_files.append(str(extra_json))
        if extra_json.suffix.lower() == ".json":
            sentences.extend(list(_iter_sentences_from_json(extra_json)))
        elif extra_json.suffix.lower() == ".jsonl":
            sentences.extend(list(_iter_sentences_from_jsonl(extra_json)))

    pack = build_ngram_pack(
        sentences,
        order=args.order,
        lowercase=not args.no_lowercase,
        split_hyphen=not args.no_split_hyphen,
        normalize_numbers=not args.no_normalize_numbers,
        add_sentence_markers=not args.no_sentence_markers,
    )
    pack["meta"]["source_files"] = used_files

    with out_path.open("wb") as f:
        pickle.dump(pack, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("✅ Built financial n-gram pack")
    print(f"  input_dir: {in_dir}")
    print(f"  files: {', '.join(used_files)}")
    print(f"  sentences: {pack['meta']['num_sentences']}")
    print(f"  vocab_size: {len(pack['vocab'])}")
    print(f"  order: {pack['order']}")
    print(f"  output: {out_path}")


if __name__ == "__main__":
    main(None if len(sys.argv) > 1 else "--input_dir ../resources/corpus/financial --output ../resources/models/fin_ngram.pkl --order 2 --extra_json ../resources/corpus/financial/polygon_news_sample.json".split())
