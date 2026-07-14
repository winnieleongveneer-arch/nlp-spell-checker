# Enable postponed evaluation of annotations (helps with forward references and import speed).
from __future__ import annotations
# Import argparse to support running this script from the command line with parameters.
import argparse
# Import json to save the POS lexicon and configuration as JSON files.
import json
# Import random to optionally sample a subset of sentences when the corpus is huge.
import random
# Import re to do lightweight sentence splitting when needed.
import re
# Import time to measure preprocessing and tagging time for logging.
import time
# Import Counter and defaultdict for counting POS tags per word efficiently.
from collections import Counter, defaultdict
# Import Path for robust file path operations across OSes.
from pathlib import Path
# Import typing primitives for clear function signatures and static checking.
from typing import Dict, List, Optional, Tuple, Any
# Import nltk for tokenization, sentence splitting, and POS tagging.
import nltk
# Import pandas to load CSV corpora when provided.
import pandas as pd
# Import your project’s Unicode normalization utility to make word forms consistent.
from preprocess.normalization import _normalize_unicode
# Import your project’s URL/email/phone removal utility to reduce noisy tokens.
from preprocess.urls_emails_phones import _remove_urls_emails_phones
# Import your project’s whitespace normalization utility to standardize spacing.
from preprocess.whitespace import _normalize_whitespace


# Define the maximum number of sentences to process to keep offline training tractable.
MAX_SENTS = 300_000
# Define a fixed random seed for reproducible sampling when truncating a large corpus.
RANDOM_STATE = 42
# Define how many POS tags to keep per word in the exported lexicon.
TOP_POS_PER_WORD = 2
# Define a minimum frequency threshold for (word, POS) pairs to be kept in the final lexicon.
MIN_POS_COUNT = 2


# Define a helper function that ensures required NLTK resources are available.
def ensure_nltk_resources() -> None:
    # Define the set of NLTK resources this script depends on.
    resources: List[Tuple[str, str]] = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    # Loop over required resources and download any missing ones.
    for path, name in resources:
        # Try to find the resource locally.
        try:
            nltk.data.find(path)
        # Download the resource if NLTK cannot find it.
        except LookupError:
            nltk.download(name)


# Call the NLTK resource check at import-time so the script can run immediately.
ensure_nltk_resources()


# Define a regex used for a very lightweight sentence split fallback.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Define a regex used to identify tokens that contain any digit (for skipping).
HAS_DIGIT_RE = re.compile(r"\d")
# Define a regex used to decide whether a token is alphabetic-only (for keeping).
ALPHA_RE = re.compile(r"^[A-Za-z]+$")


# Define a function to read a corpus directory containing .txt files line-by-line.
def load_txt_dir_lines(corpus_dir: Path) -> List[str]:
    # Create an empty list to store all text lines.
    texts: List[str] = []
    # Collect all .txt files in the given directory, sorted for stable ordering.
    txt_files: List[Path] = sorted(corpus_dir.glob("*.txt"))
    # Raise an error if no .txt files were found.
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {corpus_dir}")
    # Iterate over every .txt file.
    for fp in txt_files:
        # Open the file as UTF-8 with ignore errors to survive messy corpora.
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            # Iterate line-by-line to avoid loading huge files into memory.
            for line in f:
                # Strip leading/trailing whitespace including newline.
                line = line.strip()
                # Skip empty lines to avoid producing empty sentences.
                if not line:
                    continue
                # Add the non-empty line to the list of raw text samples.
                texts.append(line)
    # Return the list of raw text samples.
    return texts


# Define a function to read a corpus CSV and extract the target text column.
def load_csv_texts(csv_path: Path, text_col: str = "text", csv_no_header: bool = False) -> List[str]:
    # Load CSV without header if requested and assign default column names.
    if csv_no_header:
        df = pd.read_csv(csv_path, encoding="latin-1", header=None, names=["label", "text"])
        text_col = "text"
    # Otherwise load CSV normally with its header row.
    else:
        df = pd.read_csv(csv_path)
    # Validate that the requested text column exists.
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in CSV. Available: {list(df.columns)}")
    # Return non-null rows coerced to string.
    return df[text_col].dropna().astype(str).tolist()


# Define a function to clean raw text into a normalized form suitable for POS tagging.
def basic_clean_text(text: str) -> str:
    # Normalize Unicode to reduce visually identical but different codepoints.
    s: str = _normalize_unicode(text)
    # Replace newlines and tabs with spaces to preserve sentence continuity.
    s = s.replace("\n", " ").replace("\t", " ")
    # Remove URLs/emails/phones so they do not become misleading tokens.
    s = _remove_urls_emails_phones(
        s,
        remove_urls=True,
        remove_emails=True,
        remove_phones=True,
        url_placeholder=" ",
        email_placeholder=" ",
        phone_placeholder=" ",
    )
    # Normalize whitespace (collapse repeated spaces and strip).
    s = _normalize_whitespace(s)
    # Return the cleaned string.
    return s


# Define a function to split a text into sentences (preferring NLTK, fallback to regex).
def split_into_sentences(text: str) -> List[str]:
    # Try using NLTK sentence tokenizer for better quality.
    try:
        return nltk.sent_tokenize(text)
    # Fall back to a simple regex split if sentence tokenizer fails for any reason.
    except Exception:
        return [s for s in SENT_SPLIT_RE.split(text) if s.strip()]


# Define a function to tokenize a sentence into word tokens using NLTK.
def tokenize_sentence(sent: str) -> List[str]:
    # Use NLTK word_tokenize for standard English tokenization.
    return nltk.word_tokenize(sent)


# Define a function to decide whether a token is usable for building a POS lexicon.
def is_usable_word_token(tok: str) -> bool:
    # Reject tokens containing any digits to match your spell-check skipping rule.
    if HAS_DIGIT_RE.search(tok):
        return False
    # Reject non-alphabetic tokens (punctuation, symbols, etc.).
    if not ALPHA_RE.fullmatch(tok):
        return False
    # Accept the token as a usable word token.
    return True


# Define a function that updates the word->POS counts using POS-tagged tokens.
def update_pos_counts(
    tagged_tokens: List[Tuple[str, str]],
    pos_counts: Dict[str, Counter],
) -> None:
    # Iterate over each (token, pos_tag) pair returned by the tagger.
    for tok, pos in tagged_tokens:
        # Skip tokens that are not usable word tokens.
        if not is_usable_word_token(tok):
            continue
        # Normalize and lowercase the token to make the lexicon consistent with your packs.
        w: str = _normalize_unicode(tok).lower()
        # Skip empty results after normalization.
        if not w:
            continue
        # Increment the count for this (word, pos) pair.
        pos_counts[w][pos] += 1


# Define a function that converts raw counts into a compact lexicon for runtime use.
def finalize_pos_lexicon(
    pos_counts: Dict[str, Counter],
    top_pos_per_word: int = TOP_POS_PER_WORD,
    min_pos_count: int = MIN_POS_COUNT,
) -> Dict[str, Dict[str, int]]:
    # Create an output dictionary mapping word -> {pos: count}.
    lex: Dict[str, Dict[str, int]] = {}
    # Iterate over each word and its POS counter.
    for w, c in pos_counts.items():
        # Filter out rare POS labels under the minimum count threshold.
        filtered_items: List[Tuple[str, int]] = [(pos, cnt) for pos, cnt in c.items() if cnt >= min_pos_count]
        # Skip words that have no remaining POS entries after filtering.
        if not filtered_items:
            continue
        # Sort POS entries by descending count, then by POS label for stable output.
        filtered_items.sort(key=lambda x: (-x[1], x[0]))
        # Keep only top-k POS tags per word for compactness.
        top_items: List[Tuple[str, int]] = filtered_items[:top_pos_per_word]
        # Store into the output dictionary.
        lex[w] = {pos: cnt for pos, cnt in top_items}
    # Return the finalized lexicon.
    return lex


# Define a function that exports the lexicon and metadata into an output directory.
def export_pos_pack(
    out_dir: Path,
    pos_lexicon: Dict[str, Dict[str, int]],
    config: Dict[str, Any],
) -> None:
    # Ensure the output directory exists.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Write the POS lexicon JSON file.
    (out_dir / "pos_lexicon.json").write_text(json.dumps(pos_lexicon, ensure_ascii=False, indent=2), encoding="utf-8")
    # Write the config JSON file for reproducibility.
    (out_dir / "pos_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    # Create a small snapshot file for quick manual inspection.
    snapshot_lines: List[str] = []
    # Iterate through a sample of words to create a readable snapshot.
    for w in sorted(pos_lexicon.keys())[:200]:
        # Build a printable representation of POS entries for the snapshot.
        entries: str = ", ".join([f"{p}:{pos_lexicon[w][p]}" for p in sorted(pos_lexicon[w].keys())])
        # Append a snapshot line.
        snapshot_lines.append(f"{w}\t{entries}")
    # Write the snapshot file.
    (out_dir / "pos_lexicon_snapshot.txt").write_text("\n".join(snapshot_lines), encoding="utf-8")


# Define the main offline training routine that builds a POS lexicon from a corpus.
def build_pos_lexicon_pack(
    corpus_dir: Optional[str],
    corpus_csv: Optional[str],
    text_col: str,
    csv_no_header: bool,
    out_dir: str,
    max_sents: int,
    top_pos_per_word: int,
    min_pos_count: int,
) -> None:
    # Seed randomness for reproducible sampling.
    random.seed(RANDOM_STATE)
    # Create containers to hold all raw text samples.
    all_texts: List[str] = []
    # Load raw texts from a directory if provided.
    if corpus_dir:
        dir_path: Path = Path(corpus_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"corpus_dir not found: {dir_path}")
        all_texts.extend(load_txt_dir_lines(dir_path))
    # Load raw texts from a CSV if provided.
    if corpus_csv:
        csv_path: Path = Path(corpus_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"corpus_csv not found: {csv_path}")
        all_texts.extend(load_csv_texts(csv_path, text_col=text_col, csv_no_header=csv_no_header))
    # Raise an error if nothing was loaded.
    if not all_texts:
        raise ValueError("No corpus texts loaded. Provide --corpus_dir and/or --corpus_csv.")
    # Initialize a list to store sentence strings for tagging.
    sentences: List[str] = []
    # Start timing preprocessing.
    t0: float = time.perf_counter()
    # Convert raw texts into cleaned sentences.
    for raw in all_texts:
        cleaned: str = basic_clean_text(raw)
        if not cleaned:
            continue
        sentences.extend(split_into_sentences(cleaned))
    # Stop timing preprocessing.
    t1: float = time.perf_counter()
    # Filter out empty sentences.
    sentences = [s for s in sentences if s and s.strip()]
    # If the corpus is too large, sample to cap runtime.
    if len(sentences) > max_sents:
        sentences = random.sample(sentences, max_sents)
    # Initialize word->POS Counter mapping.
    pos_counts: Dict[str, Counter] = defaultdict(Counter)
    # Start timing POS tagging.
    t2: float = time.perf_counter()
    # Iterate through sentences and update POS counts.
    for i, sent in enumerate(sentences, start=1):
        tokens: List[str] = tokenize_sentence(sent)
        tagged: List[Tuple[str, str]] = nltk.pos_tag(tokens)
        update_pos_counts(tagged, pos_counts)
        if i % 5000 == 0:
            print(f"POS tagged {i}/{len(sentences)} sentences...")
    # Stop timing POS tagging.
    t3: float = time.perf_counter()
    # Finalize the compact lexicon representation.
    pos_lexicon: Dict[str, Dict[str, int]] = finalize_pos_lexicon(
        pos_counts=pos_counts,
        top_pos_per_word=top_pos_per_word,
        min_pos_count=min_pos_count,
    )
    # Build a config dictionary describing how this pack was generated.
    config: Dict[str, Any] = {
        "corpus_dir": corpus_dir,
        "corpus_csv": corpus_csv,
        "text_col": text_col,
        "csv_no_header": csv_no_header,
        "max_sents": max_sents,
        "top_pos_per_word": top_pos_per_word,
        "min_pos_count": min_pos_count,
        "random_state": RANDOM_STATE,
        "sentences_used": len(sentences),
        "unique_words": len(pos_lexicon),
        "timing_seconds": {
            "preprocess_sentences": round(t1 - t0, 6),
            "pos_tagging": round(t3 - t2, 6),
        },
        "notes": "POS lexicon pack for runtime POS-aware candidate filtering/ranking.",
    }
    # Export the pack to disk.
    export_pos_pack(out_dir=Path(out_dir), pos_lexicon=pos_lexicon, config=config)
    # Print a summary so you can confirm outputs quickly.
    print("=" * 60)
    print("POS Lexicon Pack built successfully!")
    print(f"Output dir         : {out_dir}")
    print(f"Sentences used     : {len(sentences)}")
    print(f"Unique words stored: {len(pos_lexicon)}")
    print("Files generated:")
    print(" - pos_lexicon.json")
    print(" - pos_config.json")
    print(" - pos_lexicon_snapshot.txt")


# Define a function that parses CLI arguments for running this script from terminal.
def parse_args() -> argparse.Namespace:
    # Create an argument parser with a short description.
    parser = argparse.ArgumentParser(description="Build POS Lexicon Pack for Spell Correction (Offline)")
    # Add an argument for a text corpus directory.
    parser.add_argument("--corpus_dir", type=str, default="../resources/corpus", help="Folder containing .txt files (optional).")
    # Add an argument for a CSV corpus file.
    parser.add_argument("--corpus_csv", type=str, default="../resources/corpus/financial/all-data.csv", help="CSV corpus file path (optional).")
    # Add an argument for the text column name in CSV.
    parser.add_argument("--text_col", type=str, default="text", help="Text column name for --corpus_csv.")
    # Add a flag indicating the CSV has no header.
    parser.add_argument("--csv_no_header", default=True, action="store_true", help="Set if CSV has no header row.")
    # Add an argument for the output directory.
    parser.add_argument("--out_dir", type=str, default="../resources/models", help="Output directory for POS pack.")
    # Add an argument controlling the max number of sentences to process.
    parser.add_argument("--max_sents", type=int, default=MAX_SENTS, help="Maximum sentences to process.")
    # Add an argument controlling how many POS tags to keep per word.
    parser.add_argument("--top_pos_per_word", type=int, default=TOP_POS_PER_WORD, help="Top POS tags kept per word.")
    # Add an argument controlling minimum count per (word, POS) entry.
    parser.add_argument("--min_pos_count", type=int, default=MIN_POS_COUNT, help="Minimum POS count to keep.")
    # Return parsed arguments.
    return parser.parse_args()


# Run the script when executed directly.
if __name__ == "__main__":
    # Parse command-line arguments.
    args = parse_args()
    # Call the pack builder using the parsed arguments.
    build_pos_lexicon_pack(
        corpus_dir=args.corpus_dir,
        corpus_csv=args.corpus_csv,
        text_col=args.text_col,
        csv_no_header=args.csv_no_header,
        out_dir=args.out_dir,
        max_sents=args.max_sents,
        top_pos_per_word=args.top_pos_per_word,
        min_pos_count=args.min_pos_count,
    )
