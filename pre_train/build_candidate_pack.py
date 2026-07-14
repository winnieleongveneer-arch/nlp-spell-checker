# Enable postponed evaluation of type annotations (Python 3.7+) for forward references.
from __future__ import annotations

# Import argparse to support optional CLI argument parsing (even if currently unused).
import argparse
# Import json to read/write JSON configuration and frequency files.
import json
# Import pickle to load/save Python objects (sets and bucket indices).
import pickle
# Import defaultdict to build bucket dictionaries without manual key initialization.
from collections import defaultdict
# Import Path to handle filesystem paths in an OS-independent way.
from pathlib import Path
# Import typing helpers to specify return types and accepted bucket structures.
from typing import Dict, List, Set, Union

# Import your Unicode normalization helper for idempotent normalization of vocabulary items.
from preprocess.normalization import _normalize_unicode


# Define a function that loads a vocabulary set from a Dictionary Pack directory using priority fallbacks.
def load_vocab_from_dict_pack(dict_dir: Path) -> Set[str]:
    # Build the expected path to the pickled vocabulary set.
    vocab_pkl = dict_dir / "vocab_set.pkl"
    # Build the expected path to the JSON word frequency dictionary.
    freq_json = dict_dir / "word_freq.json"
    # Build the expected path to the sorted vocabulary list text file.
    sorted_txt = dict_dir / "sorted_vocab_list.txt"
    # Prefer loading the pickled vocabulary set if it exists.
    if vocab_pkl.exists():
        # Open the pickle file in binary read mode.
        with vocab_pkl.open("rb") as f:
            # Deserialize the object from disk.
            vocab = pickle.load(f)
        # Convert to a set if the stored object is not already a set type.
        if not isinstance(vocab, set):
            vocab = set(vocab)
        # Return the resulting vocabulary set.
        return vocab
    # Fall back to loading from the JSON frequency file if it exists.
    if freq_json.exists():
        # Open the JSON file as UTF-8 text.
        with freq_json.open("r", encoding="utf-8") as f:
            # Parse the JSON object into a Python dictionary.
            freq = json.load(f)
        # Return the set of dictionary keys as the vocabulary.
        return set(freq.keys())
    # Fall back to loading from the sorted vocab list text file if it exists.
    if sorted_txt.exists():
        # Initialize an empty set to collect vocabulary items.
        vocab: Set[str] = set()
        # Open the text file as UTF-8.
        with sorted_txt.open("r", encoding="utf-8") as f:
            # Read the file line-by-line.
            for line in f:
                # Remove surrounding whitespace.
                line = line.strip()
                # Skip empty lines.
                if not line:
                    continue
                # Split the line by tab to separate word and count.
                parts = line.split("\t")
                # Extract the word field.
                word = parts[0].strip()
                # Add the word if it is non-empty.
                if word:
                    vocab.add(word)
        # Return the collected vocabulary set.
        return vocab
    # Raise an error if none of the expected dictionary-pack files were found.
    raise FileNotFoundError(
        f"Cannot find vocab_set.pkl / word_freq.json / sorted_vocab_list.txt in {dict_dir}"
    )


# Define a function that lightly cleans the vocabulary by normalizing Unicode and filtering abnormal words.
def clean_vocab(vocab: Set[str], max_word_len: int) -> Set[str]:
    # Initialize an output set for cleaned vocabulary items.
    out: Set[str] = set()
    # Iterate over every word in the input vocabulary.
    for w in vocab:
        # Skip falsy entries (e.g., empty strings or None-like values).
        if not w:
            continue
        # Normalize Unicode and trim whitespace to stabilize word forms.
        w_norm = _normalize_unicode(w).strip()
        # Skip words that become empty after normalization/stripping.
        if not w_norm:
            continue
        # Skip words longer than the configured maximum length threshold.
        if len(w_norm) > max_word_len:
            continue
        # Add the normalized word to the cleaned vocabulary.
        out.add(w_norm)
    # Return the cleaned vocabulary set.
    return out


# Define a function that buckets vocabulary by word length to accelerate candidate generation.
def build_length_buckets(vocab: Set[str]) -> Dict[int, List[str]]:
    # Create a defaultdict that maps length to a list of words.
    buckets = defaultdict(list)
    # Place each word into the bucket keyed by its length.
    for w in vocab:
        buckets[len(w)].append(w)
    # Sort each bucket for stable output and easier inspection.
    for k in buckets:
        buckets[k].sort()
    # Convert defaultdict to a plain dict and return it.
    return dict(buckets)


# Define a function that buckets vocabulary by first character to accelerate candidate generation.
def build_first_char_buckets(vocab: Set[str]) -> Dict[str, List[str]]:
    # Create a defaultdict that maps first character to a list of words.
    buckets = defaultdict(list)
    # Place each word into the bucket keyed by its lowercased first character.
    for w in vocab:
        first = w[0].lower() if w else ""
        if not first:
            continue
        buckets[first].append(w)
    # Sort each bucket for stable output and easier inspection.
    for k in buckets:
        buckets[k].sort()
    # Convert defaultdict to a plain dict and return it.
    return dict(buckets)


# Define a function that creates a two-level bucket index: first by length, then by first character.
def build_length_first_buckets(vocab: Set[str]) -> Dict[int, Dict[str, List[str]]]:
    # Create a nested defaultdict: outer key is length, inner key is first character.
    outer = defaultdict(lambda: defaultdict(list))
    # Insert each word into its (length, first_char) bucket.
    for w in vocab:
        L = len(w)
        first = w[0].lower() if w else ""
        if not first:
            continue
        outer[L][first].append(w)
    # Initialize a regular dictionary for the final, sorted output.
    result: Dict[int, Dict[str, List[str]]] = {}
    # Convert nested defaultdicts into plain dicts while sorting word lists.
    for L, inner in outer.items():
        inner_dict: Dict[str, List[str]] = {}
        for ch, words in inner.items():
            inner_dict[ch] = sorted(words)
        result[L] = inner_dict
    # Return the result sorted by word length key for stable output.
    return dict(sorted(result.items(), key=lambda x: x[0]))


# Define a function that writes the candidate bucket index plus metadata and a vocab snapshot to disk.
def export_candidate_pack(
    out_dir: Path,
    buckets: Union[Dict, List],
    config: dict,
    vocab_snapshot: List[str],
):
    # Ensure the output directory exists.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Open the bucket output file and serialize the bucket structure.
    with (out_dir / "candidate_buckets.pkl").open("wb") as f:
        pickle.dump(buckets, f)
    # Open the config output file and write metadata as pretty JSON.
    with (out_dir / "candidate_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    # Open the vocabulary snapshot file and write one word per line.
    with (out_dir / "candidate_vocab_snapshot.txt").open("w", encoding="utf-8") as f:
        for w in vocab_snapshot:
            f.write(w + "\n")


# Define the end-to-end pipeline: load vocab, clean it, build buckets, and export the candidate pack.
def main(dict_dir: str, out_dir: str, mode: str, max_word_len: int):
    # Convert the input dictionary directory string to a Path object.
    dict_path = Path(dict_dir)
    # Validate that the dictionary directory exists.
    if not dict_path.exists():
        raise FileNotFoundError(f"dict_dir not found: {dict_path}")
    # Load the vocabulary set from the dictionary pack using the defined priority rules.
    vocab = load_vocab_from_dict_pack(dict_path)
    # Clean the vocabulary by normalizing and removing abnormal words.
    vocab = clean_vocab(vocab, max_word_len=max_word_len)
    # Build buckets based on the selected mode.
    if mode == "length":
        buckets = build_length_buckets(vocab)
    elif mode == "first_char":
        buckets = build_first_char_buckets(vocab)
    elif mode == "both":
        buckets = {"length": build_length_buckets(vocab), "first_char": build_first_char_buckets(vocab)}
    elif mode == "length_first":
        buckets = build_length_first_buckets(vocab)
    else:
        raise ValueError("mode must be one of: length, first_char, both, length_first")
    # Create a deterministic snapshot of the vocabulary for auditing and screenshots.
    vocab_snapshot = sorted(vocab)
    # Build a configuration dictionary that records how the candidate pack was produced.
    config = {
        "mode": mode,
        "max_word_len": max_word_len,
        "dict_dir": dict_dir,
        "vocab_size": len(vocab),
        "outputs": ["candidate_buckets.pkl", "candidate_config.json", "candidate_vocab_snapshot.txt"],
        "notes": "Candidate Pack for MED candidate generation acceleration (A1).",
    }
    # Export the bucket index, config metadata, and snapshot to the output directory.
    export_candidate_pack(
        out_dir=Path(out_dir),
        buckets=buckets,
        config=config,
        vocab_snapshot=vocab_snapshot,
    )
    # Print a separator line for readable logs.
    print("=" * 60)
    # Print a success message.
    print("Candidate Pack built successfully!")
    # Print the dictionary pack directory used.
    print(f"Dictionary dir : {dict_dir}")
    # Print the output directory used.
    print(f"Output dir     : {out_dir}")
    # Print the chosen bucketing mode.
    print(f"Mode           : {mode}")
    # Print the final vocabulary size after cleaning.
    print(f"Vocab size     : {len(vocab)}")
    # Print the names of files generated by this script.
    print("Files generated:")
    print(" - candidate_buckets.pkl")
    print(" - candidate_config.json")
    print(" - candidate_vocab_snapshot.txt")


# Run the script entrypoint when executed as a program.
if __name__ == "__main__":
    # Set the dictionary pack directory path to use in hard-coded mode.
    dict_dir = "../resources/models"
    # Set the output directory path where candidate pack files will be written.
    out_dir = "../resources/models"
    # Choose the bucketing mode for candidate acceleration.
    mode = "both"
    # Set the maximum allowed word length for vocabulary cleaning.
    max_word_len = 50
    # Execute the candidate pack build pipeline.
    main(
        dict_dir=dict_dir,
        out_dir=out_dir,
        mode=mode,
        max_word_len=max_word_len,
    )
    # Import pickle under an alias for self-check loading of the generated buckets file.
    import pickle as _pkl
    # Import Path under an alias for self-check filesystem handling.
    from pathlib import Path as _Path
    # Create a Path pointing to the generated candidate buckets pickle file.
    p = _Path("../resources/models/candidate_buckets.pkl")
    # Open the pickle file and deserialize the buckets structure for inspection.
    with p.open("rb") as f:
        buckets = _pkl.load(f)
    # Print the runtime type of the loaded buckets object.
    print(type(buckets))
    # Print the top-level keys of the buckets object (varies by mode).
    print(buckets.keys())
    # Print example bucket keys for the length bucketing strategy.
    print("length buckets example keys:", list(buckets["length"].keys())[:10])
    # Print example bucket keys for the first-char bucketing strategy.
    print("first_char buckets example keys:", list(buckets["first_char"].keys())[:10])
    # Pick the smallest length bucket key to show a small sample list.
    some_len = sorted(buckets["length"].keys())[0]
    # Print the sample bucket length and the first few words inside it.
    print("sample length =", some_len, buckets["length"][some_len][:20])
