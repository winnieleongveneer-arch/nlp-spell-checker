#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_fin_idf.py

Build an IDF pack (token -> idf) from the financial corpus under:
  ../resources/corpus/financial/

Outputs:
  ../resources/models/fin_idf.pkl

This is designed for your PartA project layout:
  PartA/
    pre_train/   <-- put this file here
    resources/
      corpus/financial/  (txt + csv)
      models/

Notes
- No external dependencies required (only stdlib).
- Tokenization mirrors your earlier n-gram builder: normalize case, split hyphen, normalize numbers.
- IDF is sentence-level: df(token)=#sentences containing token.
- idf(token)=log((N+1)/(df+1)) + 1  (smooth, always >0)
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


_NUM_RE = re.compile(r"^\d+([.,]\d+)*$")


def _normalize_token(
    tok: str,
    lowercase: bool = True,
    split_hyphen: bool = True,
    normalize_numbers: bool = True,
) -> List[str]:
    """Normalize and (optionally) split a token into one or more pieces."""
    tok = tok.strip()
    if not tok:
        return []
    if lowercase:
        tok = tok.lower()

    if normalize_numbers and _NUM_RE.match(tok):
        tok = "<num>"

    if split_hyphen and "-" in tok and tok != "-":
        parts = [p for p in tok.split("-") if p]
        return parts if parts else [tok]
    return [tok]


def _iter_sentences_from_txt(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def _iter_sentences_from_csv(path: Path) -> Iterable[str]:
    """
    Attempt to read sentences from a CSV file.

    Heuristics:
    - If there is a column named 'sentence' or 'text' (case-insensitive), use it.
    - Otherwise, use the longest string field in each row.
    """
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        rows = list(reader)

    if not rows:
        return

    header = rows[0]
    header_l = [h.strip().lower() for h in header]
    col_idx = None
    has_header = False
    for key in ("sentence", "text", "content"):
        if key in header_l:
            has_header = True
            col_idx = header_l.index(key)
            break

    start = 1 if has_header else 0
    for row in rows[start:]:
        if not row:
            continue
        sent = None
        if col_idx is not None and col_idx < len(row):
            candidate = (row[col_idx] or "").strip()
            if candidate:
                sent = candidate
        if sent is None:
            sent = max((c or "" for c in row), key=lambda s: len(s.strip()), default="").strip()
        if sent:
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


def _tokenize_sentence(
    s: str,
    lowercase: bool = True,
    split_hyphen: bool = True,
    normalize_numbers: bool = True,
) -> List[str]:
    raw = s.strip().split()
    out: List[str] = []
    for t in raw:
        out.extend(_normalize_token(t, lowercase=lowercase, split_hyphen=split_hyphen, normalize_numbers=normalize_numbers))
    return out


@dataclass
class IDFPack:
    idf: Dict[str, float]
    df: Dict[str, int]
    n_sentences: int
    files: List[str]
    options: Dict[str, object]


def build_idf_pack(
    input_dir: Path,
    use_files: Optional[List[str]] = None,
    extra_json: Optional[Path] = None,
    lowercase: bool = True,
    split_hyphen: bool = True,
    normalize_numbers: bool = True,
) -> IDFPack:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found or not a directory: {input_dir}")

    files: List[Path] = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        if use_files and p.name not in use_files:
            continue
        if p.suffix.lower() in (".txt", ".csv"):
            files.append(p)

    if not files:
        raise RuntimeError(f"No .txt/.csv files found in {input_dir} (use_files={use_files})")

    df: Dict[str, int] = {}
    n_sentences = 0

    def add_sentence(sent: str) -> None:
        nonlocal n_sentences
        toks = _tokenize_sentence(
            sent,
            lowercase=lowercase,
            split_hyphen=split_hyphen,
            normalize_numbers=normalize_numbers,
        )
        if not toks:
            return
        n_sentences += 1
        uniq: Set[str] = set(toks)
        for t in uniq:
            df[t] = df.get(t, 0) + 1

    for fp in files:
        sent_iter = _iter_sentences_from_txt(fp) if fp.suffix.lower() == ".txt" else _iter_sentences_from_csv(fp)

        for sent in sent_iter:
            add_sentence(sent)

    extra_used: Optional[str] = None
    if extra_json and extra_json.exists() and extra_json.is_file():
        extra_used = str(extra_json)
        suf = extra_json.suffix.lower()
        if suf == ".json":
            for sent in _iter_sentences_from_json(extra_json):
                add_sentence(sent)
        elif suf == ".jsonl":
            for sent in _iter_sentences_from_jsonl(extra_json):
                add_sentence(sent)

    N = n_sentences
    idf: Dict[str, float] = {t: (math.log((N + 1.0) / (d + 1.0)) + 1.0) for t, d in df.items()}

    return IDFPack(
        idf=idf,
        df=df,
        n_sentences=n_sentences,
        files=[p.name for p in files] + ([extra_used] if extra_used else []),
        options={
            "lowercase": lowercase,
            "split_hyphen": split_hyphen,
            "normalize_numbers": normalize_numbers,
            "idf_formula": "log((N+1)/(df+1)) + 1",
        },
    )


def save_pack(pack: IDFPack, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(pack, f, protocol=pickle.HIGHEST_PROTOCOL)


def main(
    input_dir: str,
    output: str,
    use_files: Optional[List[str]] = None,
    extra_json: str = "",
    lowercase: bool = True,
    split_hyphen: bool = True,
    normalize_numbers: bool = True,
) -> None:
    in_dir = Path(input_dir)
    out_path = Path(output)

    extra_p = Path(extra_json).resolve() if extra_json.strip() else None
    pack = build_idf_pack(
        input_dir=in_dir,
        use_files=use_files,
        extra_json=extra_p,
        lowercase=lowercase,
        split_hyphen=split_hyphen,
        normalize_numbers=normalize_numbers,
    )
    save_pack(pack, out_path)

    print("✅ Built financial IDF pack")
    print(f"  input_dir: {in_dir}")
    print(f"  files: {', '.join(pack.files)}")
    print(f"  sentences: {pack.n_sentences}")
    print(f"  vocab_size: {len(pack.idf)}")
    print(f"  output: {out_path}")


if __name__ == "__main__":
    # Hard-coded args to match your project layout (run directly without CLI args).
    # Expected location: PartA/pre_train/build_fin_idf.py
    here = Path(__file__).resolve()
    project_root = here.parent.parent  # .../PartA
    input_dir = project_root / "resources" / "corpus" / "financial"
    output = project_root / "resources" / "models" / "fin_idf.pkl"

    use_files = None  # or e.g. ["Sentences_AllAgree.txt", "all-data.csv"]

    main(
        input_dir=str(input_dir),
        output=str(output),
        use_files=use_files,
        extra_json=str(project_root / "resources" / "corpus" / "financial" / "polygon_news_sample.json"),
        lowercase=True,
        split_hyphen=True,
        normalize_numbers=True,
    )
