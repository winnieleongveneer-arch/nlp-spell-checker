# Enable postponed evaluation of type annotations (Python 3.7+) for forward references.
from __future__ import annotations

# Import argparse to support optional CLI argument parsing (even if unused in hard-coded mode).
import argparse
# Import json to write word frequency dictionaries as JSON.
import json
# Import pickle to serialize Python objects like sets to disk.
import pickle
# Import re to define and apply regex patterns for normalization and tokenization.
import re
# Import Counter to count token frequencies efficiently.
from collections import Counter
# Import Path to handle filesystem paths in an OS-independent way.
from pathlib import Path
# Import typing helpers for type hints and clearer function signatures.
from typing import Iterable, List, Optional
# Import time to measure per-text processing time.
import time

# Import pandas to read CSV corpora into dataframes.
import pandas as pd

# Import Unicode normalization helper from your preprocess module.
from preprocess.normalization import _normalize_unicode
# Import URL/email/phone removal helper from your preprocess module.
from preprocess.urls_emails_phones import _remove_urls_emails_phones
# Import lowercasing helper from your preprocess module (not used directly in current flow).
from preprocess.lowercasing import _lowercase
# Import digit/punctuation normalization helper from your preprocess module.
from preprocess.digits_punct import _coarse_digits_punct
# Import whitespace normalization helper from your preprocess module.
from preprocess.whitespace import _normalize_whitespace


# Define a function that loads all .txt files in a folder and returns non-empty lines as samples.
def read_txt_files(corpus_dir: Path) -> List[str]:
    # Initialize a list to collect one text sample per non-empty line.
    texts: List[str] = []
    # Collect all .txt files in sorted order for deterministic processing.
    txt_files = sorted(corpus_dir.glob("*.txt"))
    # Raise an error if there are no .txt files to read.
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {corpus_dir}. "
            "Please put your corpus text files into this folder, "
            "or provide a CSV corpus via --corpus_csv."
        )
    # Iterate through each .txt file.
    for fp in txt_files:
        # Open each file as UTF-8 and ignore decode errors to be robust to messy data.
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            # Iterate line-by-line so each line becomes one text sample.
            for line in f:
                # Remove surrounding whitespace and newline characters.
                line = line.strip()
                # Skip empty lines to avoid creating empty samples.
                if not line:
                    continue
                # Append the cleaned line to the list of samples.
                texts.append(line)
    # Return the collected line-based text samples.
    return texts


# Define a function that loads each .txt file as one sample (file-level samples).
def read_txt_files_back(corpus_dir: Path) -> List[str]:
    # Initialize a list to collect one text sample per file.
    texts: List[str] = []
    # Collect all .txt files in sorted order for deterministic processing.
    txt_files = sorted(corpus_dir.glob("*.txt"))
    # Raise an error if there are no .txt files to read.
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {corpus_dir}. "
            "Please put your corpus text files into this folder, "
            "or provide a CSV corpus via --corpus_csv."
        )
    # Iterate through each .txt file.
    for fp in txt_files:
        # Open the file as UTF-8 and ignore decode errors to be robust to messy data.
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            # Read the entire file content as one sample.
            texts.append(f.read())
    # Return the collected file-level text samples.
    return texts


# Define a function that reads a CSV file and returns a list of text rows from a chosen column.
def read_csv_texts(
    csv_path: Path,
    text_col: str = "text",
    csv_no_header: bool = False,
) -> List[str]:
    # Read CSV without a header by assigning default column names when requested.
    if csv_no_header:
        # Read using latin-1 to tolerate non-UTF8 bytes and name columns as label/text.
        df = pd.read_csv(csv_path, encoding="latin-1", header=None, names=["label", "text"])
        # Force the text column name to the auto-assigned "text".
        text_col = "text"
    # Otherwise read the CSV normally and use the existing header.
    else:
        df = pd.read_csv(csv_path)
    # Validate that the desired text column exists in the dataframe.
    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    # Return the text column as a list of strings with NaNs removed.
    return df[text_col].dropna().astype(str).tolist()


# Define a legacy normalizer kept as a backup implementation.
def basic_normalize_bak(text: str) -> str:
    # Import unicodedata locally so this function is self-contained.
    import unicodedata
    # Normalize Unicode into NFKC form to unify visually equivalent characters.
    text = unicodedata.normalize("NFKC", text)
    # Remove simple HTML tags by replacing with spaces.
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse multiple whitespace into a single space and trim ends.
    text = re.sub(r"\s+", " ", text).strip()
    # Return the normalized string.
    return text


# Define the main normalizer that uses shared preprocess utilities and optional casing behavior.
def basic_normalize(text: str, *, keep_case: bool = False) -> str:
    # Apply shared Unicode normalization and control-character cleanup.
    s = _normalize_unicode(text)
    # Lowercase text unless caller requests preserving original case.
    if not keep_case:
        s = s.lower()
    # Remove HTML tags by replacing with spaces.
    s = re.sub(r"<[^>]+>", " ", s)
    # Remove or placeholder URLs/emails/phones to reduce noisy tokens.
    s = _remove_urls_emails_phones(
        s,
        remove_urls=True,
        remove_emails=True,
        remove_phones=True,
        url_placeholder="<URL>",
        email_placeholder="<EMAIL>",
        phone_placeholder="<PHONE>",
    )
    # Remove any whitespace-delimited token that contains at least one digit.
    s = re.sub(r"\b\S*\d\S*\b", " ", s)
    # Apply coarse punctuation normalization and digit handling (digits are not kept here).
    s = _coarse_digits_punct(
        s,
        keep_digits=False,
        replace_digits_with=None,
    )
    # Normalize whitespace and trim ends.
    s = _normalize_whitespace(s)
    # Return the cleaned and normalized string.
    return s


# Compile a token regex that captures alphanumeric tokens with internal hyphens/apostrophes.
TOKEN_PATTERN = re.compile(
    r"""
    (?:[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*)
    """,
    re.VERBOSE,
)


# Define a tokenizer that extracts tokens using the compiled regex pattern.
def tokenize(text: str) -> List[str]:
    # Return all tokens matched by the token regex.
    return TOKEN_PATTERN.findall(text)


# Define a filter that removes abnormally long tokens or punctuation-only tokens.
def filter_tokens(tokens: Iterable[str], max_token_len: int) -> List[str]:
    # Initialize the output list of filtered tokens.
    out: List[str] = []
    # Iterate through each candidate token.
    for t in tokens:
        # Skip tokens longer than the maximum allowed length.
        if len(t) > max_token_len:
            continue
        # Skip tokens that consist only of hyphens/apostrophes.
        if all(ch in "-'" for ch in t):
            continue
        # Keep tokens that pass both filters.
        out.append(t)
    # Return the filtered token list.
    return out


# Build the dictionary pack: vocabulary set, word frequency tables, and sorted vocab files.
def build_dictionary_pack(
    corpus_dir: Optional[str],
    corpus_csv: Optional[str],
    text_col: str = "text",
    out_dir: Path = Path("./outputs/dictionary_pack"),
    min_freq: int = 1,
    max_token_len: int = 50,
    keep_case: bool = False,
    csv_no_header: bool = False,
) -> None:
    # Ensure the output directory exists.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Create a container for all loaded texts from all sources.
    all_texts: List[str] = []
    # Create a container for normal-domain texts (if used).
    normal_texts: List[str] = []
    # Create a container for financial-domain texts (if used).
    financial_texts: List[str] = []
    # Print a stage marker for loading txt data.
    print("# 1a) 读取 txt")
    # Load .txt corpora if a directory was provided.
    if corpus_dir:
        # Convert the directory string to a Path object.
        dir_path = Path(corpus_dir)
        # Validate that the directory exists.
        if not dir_path.exists():
            raise FileNotFoundError(f"corpus_dir not found: {dir_path}")
        # Read line-based samples from the financial subdirectory.
        financial_texts.extend(read_txt_files(Path(dir_path, "financial")))
        # Read line-based samples from the normal subdirectory.
        normal_texts.extend(read_txt_files(Path(dir_path, "normal")))
    # Combine txt-based texts into the unified list.
    all_texts = financial_texts + normal_texts
    # Print a stage marker for loading csv data.
    print("# 1b) 读取 csv")
    # Load CSV corpora if a file path was provided.
    if corpus_csv:
        # Convert the CSV path string to a Path object.
        csv_path = Path(corpus_csv)
        # Validate that the CSV file exists.
        if not csv_path.exists():
            raise FileNotFoundError(f"corpus_csv not found: {csv_path}")
        # Read text rows from the configured text column and append to the unified list.
        all_texts.extend(read_csv_texts(csv_path, text_col=text_col, csv_no_header=csv_no_header))
    # Raise an error if nothing was loaded from any source.
    if not all_texts:
        raise ValueError("No corpus texts loaded. Provide corpus_dir and/or corpus_csv.")
    # Print a stage marker for normalization, tokenization, filtering, and counting.
    print("# 2) 规范化 + 3/4/5) 分词 & 过滤 & 统计")
    # Initialize a Counter to accumulate token frequencies across the entire corpus.
    counter: Counter = Counter()
    # Print the total number of text samples that will be processed.
    print("文本数量：", len(all_texts))
    # Print the number of financial text samples loaded.
    print("金融文本数量：", len(financial_texts))
    # Print the number of normal text samples loaded.
    print("日常文本数量：", len(normal_texts))
    # Iterate through each raw text sample to normalize, tokenize, filter, and count tokens.
    for raw_text in all_texts:
        # Start a timer for the normalization step.
        t0 = time.perf_counter()
        # Print the raw text length for debugging/monitoring.
        print("当前文本长度：", len(raw_text))
        # Normalize the raw text using the shared pipeline.
        norm_text = basic_normalize(raw_text)
        # Stop the timer after normalization.
        t1 = time.perf_counter()
        # Tokenize the normalized text using the regex tokenizer.
        tokens = tokenize(norm_text)
        # Apply lightweight token filtering rules.
        tokens = filter_tokens(tokens, max_token_len=max_token_len)
        # Update the global token frequency counter with this sample's tokens.
        counter.update(tokens)
        # Stop the timer after tokenization/filtering/counting.
        t2 = time.perf_counter()
        # Print timing information for normalization and subsequent processing.
        print(f"预处理耗时{t1 - t0:.6f} s", f"处理耗时{t2 - t1:.6f} s")
    # Print a stage marker for minimum-frequency filtering.
    print("# 6) min_freq 过滤")
    # Apply minimum frequency filtering by rebuilding the Counter when min_freq > 1.
    if min_freq > 1:
        counter = Counter({w: c for w, c in counter.items() if c >= min_freq})
    # Create a vocabulary set for O(1) membership checks in non-word detection.
    vocab_set = set(counter.keys())
    # Print a stage marker for writing output files.
    print("# 输出文件")
    # Sort by descending frequency then lexicographically for stable output.
    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    # Print a stage marker before writing the pickled vocabulary set.
    print("# 1) vocab_set.pkl")
    # Serialize the vocabulary set to a pickle file.
    with (out_dir / "vocab_set.pkl").open("wb") as f:
        pickle.dump(vocab_set, f)
    # Print a stage marker before writing the JSON frequency map.
    print("# 2) word_freq.json")
    # Write token frequency counts to JSON for easy inspection and reuse.
    with (out_dir / "word_freq.json").open("w", encoding="utf-8") as f:
        json.dump(counter, f, ensure_ascii=False, indent=2)
    # Print a stage marker before writing the sorted vocab text file.
    print("# 3) sorted_vocab_list.txt")
    # Write the sorted vocabulary list as tab-separated word and count per line.
    with (out_dir / "sorted_vocab_list.txt").open("w", encoding="utf-8") as f:
        for w, c in sorted_items:
            f.write(f"{w}\t{c}\n")
    # Print a stage marker before writing the CSV frequency table.
    print("# 4) word_freq.csv")
    # Write the sorted frequency table in CSV format for EDA or screenshots.
    with (out_dir / "word_freq.csv").open("w", encoding="utf-8") as f:
        f.write("word,count\n")
        for w, c in sorted_items:
            f.write(f"{w},{c}\n")
    # Print a stage marker for final summary logging.
    print("# 日志")
    # Print a success message.
    print("Dictionary Pack built successfully!")
    # Print the corpus directory used.
    print(f"Corpus dir : {corpus_dir}")
    # Print the corpus CSV used.
    print(f"Corpus csv : {corpus_csv}")
    # Print the output directory used.
    print(f"Output dir : {out_dir}")
    # Print the total number of tokens counted after preprocessing and filtering.
    print(f"Total tokens after preprocessing: {sum(counter.values())}")
    # Print the vocabulary size after applying min_freq filtering.
    print(f"Vocab size (after min_freq={min_freq}): {len(vocab_set)}")
    # Print the list of files written to disk.
    print("Files generated:")
    # Print the pickled set artifact name.
    print(" - vocab_set.pkl")
    # Print the JSON frequency artifact name.
    print(" - word_freq.json")
    # Print the sorted vocab list artifact name.
    print(" - sorted_vocab_list.txt")
    # Print the CSV frequency artifact name.
    print(" - word_freq.csv")


# Run the script entrypoint when executed as a program.
if __name__ == "__main__":
    # Execute dictionary building using the current hard-coded paths and settings.
    build_dictionary_pack(
        corpus_dir=Path("../resources/corpus"),
        corpus_csv=Path("../resources/csv/all-data.csv"),
        text_col="text",
        out_dir=Path("../resources/models"),
        min_freq=1,
        max_token_len=50,
        keep_case=False,
        csv_no_header=True,
    )
