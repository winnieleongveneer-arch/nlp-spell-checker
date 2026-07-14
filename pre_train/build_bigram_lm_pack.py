from __future__ import annotations

# Import JSON utilities for saving configuration files.
import json
# Import pickle utilities for serializing the trained model and vocabulary.
import pickle
# Import random utilities for optional down-sampling of very large corpora.
import random
# Import regular expression utilities for text cleaning rules.
import re
# Import punctuation constants used in token filtering.
import string
# Import time utilities for lightweight timing/progress logging.
import time
# Import Counter for frequency statistics (top tokens summary).
from collections import Counter
# Import Path for robust filesystem path handling.
from pathlib import Path
# Import typing helpers for clearer function signatures.
from typing import List, Optional

# Import NLTK for tokenization, POS tagging, lemmatization, and language modeling.
import nltk
# Import pandas for reading CSV corpora.
import pandas as pd
# Import train_test_split for optional held-out evaluation split.
from sklearn.model_selection import train_test_split

# Import stopwords and WordNet tag constants used for lemmatization.
from nltk.corpus import stopwords, wordnet
# Import Kneser-Ney interpolated language model implementation.
from nltk.lm.models import KneserNeyInterpolated
# Import helpers to create padded n-gram streams for training.
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
# Import Vocabulary to handle <UNK> mapping via an OOV cutoff.
from nltk.lm import Vocabulary
# Import everygrams to generate fixed-length n-grams for perplexity evaluation.
from nltk.util import everygrams

# Import Unicode normalization preprocessor.
from preprocess.normalization import _normalize_unicode
# Import URL/email/phone removal preprocessor.
from preprocess.urls_emails_phones import _remove_urls_emails_phones
# Import digit/punctuation normalization preprocessor.
from preprocess.digits_punct import _coarse_digits_punct
# Import whitespace normalization preprocessor.
from preprocess.whitespace import _normalize_whitespace

# Set a fixed random seed for reproducible sampling and splitting.
RANDOM_STATE = 42
# Set a default held-out ratio for optional evaluation.
TEST_SIZE = 0.2
# Set the n-gram order (2 = bigram model).
ORDER = 2
# Set the vocabulary cutoff: tokens with count <= cutoff become <UNK>.
UNK_CUTOFF = 1
# Set the maximum number of sentences to keep for one corpus subset (sampling).
MAX_SENTS = 300_000


# Define a helper to download required NLTK assets if missing.
def ensure_nltk():
    # Define the resources needed by tokenization, stopwords, POS tagging, and WordNet.
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    # Iterate over each resource spec (lookup path and download package name).
    for path, name in resources:
        # Try to locate the resource in the local NLTK data directory.
        try:
            nltk.data.find(path)
        # If not found, download the missing resource.
        except LookupError:
            nltk.download(name)


# Ensure NLTK resources are available before any processing starts.
ensure_nltk()

# Build a stopword set for optional filtering.
STOP = set(stopwords.words("english"))
# Build a punctuation character set for filtering punctuation-only tokens.
PUNCT = set(string.punctuation)
# Build a combined set of stopwords and punctuation for convenience.
STOP_ALL = STOP.union(PUNCT)
# Create a lemmatizer instance for optional lemmatization.
lemmatizer = nltk.WordNetLemmatizer()


# Map POS tag prefixes to WordNet POS categories for POS-aware lemmatization.
def get_wordnet_pos_from_tag(tag: str):
    # Normalize the tag to its first character and uppercase it, defaulting to noun.
    tag = tag[0].upper() if tag else "N"
    # Define the mapping from Penn-like tag prefix to WordNet category.
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    # Return the mapped WordNet POS, defaulting to noun if unknown.
    return tag_dict.get(tag, wordnet.NOUN)


# Load corpus texts from a directory of .txt files, treating each non-empty line as one sample.
def load_txt_dir(corpus_dir: Path) -> List[str]:
    # Initialize a list to hold all extracted text samples.
    texts: List[str] = []
    # List all .txt files in the directory in a deterministic order.
    txt_files = sorted(corpus_dir.glob("*.txt"))
    # Fail early if no text files exist in the given directory.
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {corpus_dir}. "
            "Please put your corpus text files into this folder, "
            "or provide a CSV corpus via --corpus_csv."
        )
    # Iterate over each text file.
    for fp in txt_files:
        # Open the file with UTF-8 decoding and ignore decoding errors.
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            # Read the file line-by-line.
            for line in f:
                # Strip leading/trailing whitespace including newline characters.
                line = line.strip()
                # Skip empty lines.
                if not line:
                    continue
                # Append the cleaned line as one text sample.
                texts.append(line)
    # Return the list of text samples.
    return texts


# Load corpus texts from a directory of .txt files, treating each whole file as one sample.
def load_txt_dir_back(corpus_dir: Path) -> List[str]:
    # Initialize a list to hold file-level text samples.
    texts: List[str] = []
    # Iterate over each .txt file in deterministic order.
    for fp in sorted(corpus_dir.glob("*.txt")):
        # Open the file with UTF-8 decoding and ignore decoding errors.
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            # Read the entire file content and strip outer whitespace.
            t = f.read().strip()
            # Append the content if it is non-empty.
            if t:
                texts.append(t)
    # Return the list of file-level text samples.
    return texts


# Load corpus texts from a CSV file using a specified text column.
def load_csv_texts(csv_path: Path, text_col: str = "text", no_header: bool = False):
    # Read CSV with a fixed schema when there is no header row.
    if no_header:
        df = pd.read_csv(csv_path, encoding="latin-1", header=None, names=["label", "text"])
    # Otherwise, read the CSV normally and infer column names.
    else:
        df = pd.read_csv(csv_path)
    # Validate that the requested column exists.
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {list(df.columns)}")
    # Return the chosen column as a list of strings with missing values removed.
    return df[text_col].dropna().astype(str).tolist()


# Define a slang mapping used optionally for tweet-like corpora.
text_mapping = {
    "u": "you",
    "r": "are",
    "btw": "by the way",
    "fyi": "for your information",
    "l8r": "later",
    "w8": "wait",
    "k": "ok",
    "2": "to",
    "4": "for",
}

# Compile a URL regex for fast repeated replacement.
URL_RE = re.compile(r"https?://\S+|www\.\S+")
# Compile an @mention regex for tweet-like cleanup.
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
# Compile a #hashtag regex for tweet-like cleanup.
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
# Compile an "RT " prefix regex for tweet retweet markers.
RT_RE = re.compile(r"^RT\s+")


# Provide a legacy cleaning function kept for reference/compatibility.
def clean_text_back(text: str) -> str:
    # Replace newlines with spaces to keep a single-line representation.
    text = text.replace("\n", " ").replace("\t", " ")
    # Remove leading "RT " markers.
    text = RT_RE.sub("", text)
    # Remove URLs.
    text = URL_RE.sub("", text)
    # Remove @mentions.
    text = MENTION_RE.sub("", text)
    # Remove #hashtags.
    text = HASHTAG_RE.sub("", text)
    # Keep only letters, digits, whitespace, apostrophes, and hyphens.
    text = re.sub(r"[^A-Za-z0-9\s'\-]", " ", text)
    # Collapse repeated whitespace and trim.
    text = re.sub(r"\s+", " ", text).strip()
    # Return the cleaned string.
    return text


# Clean raw text using preprocess modules plus the original tweet-specific rules.
def clean_text(text: str) -> str:
    # Normalize Unicode and lowercase to standardize text.
    s = _normalize_unicode(text).lower()
    # Replace newlines and tabs with spaces to avoid breaking tokenization.
    s = s.replace("\n", " ").replace("\t", " ")
    # Remove leading retweet marker.
    s = RT_RE.sub("", s)
    # Remove @mentions.
    s = MENTION_RE.sub("", s)
    # Remove #hashtags.
    s = HASHTAG_RE.sub("", s)
    # Remove URLs, emails, and phone numbers using the shared preprocessor.
    s = _remove_urls_emails_phones(
        s,
        remove_urls=True,
        remove_emails=True,
        remove_phones=True,
        url_placeholder=None,
        email_placeholder=None,
        phone_placeholder=None,
    )
    # Remove any token that contains at least one digit.
    s = re.sub(r"\b\S*\d\S*\b", " ", s)
    # Normalize digits/punctuation in a light-weight way while keeping digits unchanged.
    s = _coarse_digits_punct(s, keep_digits=True, replace_digits_with=None)
    # Restrict characters to letters, digits, whitespace, apostrophes, and hyphens.
    s = re.sub(r"[^A-Za-z0-9\s'\-]", " ", s)
    # Normalize whitespace to single spaces and trim.
    s = _normalize_whitespace(s)
    # Return the cleaned text.
    return s


# Tokenize cleaned text using NLTK word_tokenize, with optional slang expansion.
def tokenize(text: str, apply_mapping: bool = False) -> List[str]:
    # Lowercase again to ensure case normalization even if upstream changes.
    text = text.lower()
    # Reduce character elongation (e.g., "sooo" -> "so") to stabilize vocabulary.
    text = re.sub(r"([a-z])\1{2,}", r"\1", text)
    # Split into word tokens using NLTK.
    tokens = nltk.word_tokenize(text)
    # Optionally apply slang expansion mapping token-by-token.
    if apply_mapping:
        tokens = [text_mapping.get(t, t) for t in tokens]
    # Return the list of tokens.
    return tokens


# Convert a list of raw texts into a list of tokenized sentences with optional filtering/lemmatization.
def preprocess_texts(
    texts: List[str],
    is_tweet: bool = False,
    remove_stopwords: bool = False,
    use_lemma: bool = False,
    use_pos_lemma: bool = False,
) -> List[List[str]]:
    # Store the total number of texts for progress reporting.
    length = len(texts)
    # Track processed count for progress reporting.
    count = 0
    # Accumulate processed token lists here.
    processed: List[List[str]] = []
    # Iterate through each raw text.
    for t in texts:
        # Increment the processed counter.
        count += 1
        # Start timing the cleaning+tokenization step.
        t0 = time.perf_counter()
        # Clean the raw text using the shared cleaning pipeline.
        t = clean_text(t)
        # Tokenize the cleaned text and optionally apply tweet slang mapping.
        toks = tokenize(t, apply_mapping=is_tweet)
        # End timing the cleaning+tokenization step.
        t1 = time.perf_counter()
        # Initialize a filtered token buffer for this sample.
        filtered: List[str] = []
        # Iterate over tokens to apply filtering rules.
        for w in toks:
            # Optionally remove stopwords and punctuation tokens using the combined set.
            if remove_stopwords and w in STOP_ALL:
                continue
            # Skip empty tokens or tokens made entirely of punctuation characters.
            if (not w) or all(ch in PUNCT for ch in w):
                continue
            # Keep the token if it passes filters.
            filtered.append(w)
        # Skip samples that became empty after filtering.
        if not filtered:
            continue
        # Optionally lemmatize tokens to normalize word forms.
        if use_lemma:
            # Use POS-aware lemmatization if requested (slower but more accurate).
            if use_pos_lemma:
                # Tag each token with POS using NLTK's tagger.
                tags = nltk.pos_tag(filtered)
                # Lemmatize with the WordNet POS mapped from each tag.
                lemmas = [lemmatizer.lemmatize(w, get_wordnet_pos_from_tag(tag)) for w, tag in tags]
            # Otherwise use default lemmatization without POS information.
            else:
                lemmas = [lemmatizer.lemmatize(w) for w in filtered]
            # Remove any empty lemma outputs.
            filtered = [w for w in lemmas if w]
        # Append the processed tokens for this sample.
        processed.append(filtered)
        # End timing the filtering+lemmatization step.
        t2 = time.perf_counter()
        # Print periodic progress updates for large corpora.
        if (count % 10_000 == 0) or (count == length):
            print(
                f"共{length}条，当前第{count}条，进度{count / length:.2%}，"
                f"长度:{len(t)}，预处理:{t1 - t0:.4f}s，过滤+lemma:{t2 - t1:.4f}s"
            )
    # Return the list of tokenized sentences.
    return processed


# Split sentences into train and test sets for optional evaluation.
def split_sents(sents, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    # Return all data as training when the dataset is too small to split meaningfully.
    if len(sents) < 5:
        return sents, []
    # Split into train and test partitions with a fixed random seed.
    return train_test_split(sents, test_size=test_size, random_state=random_state)


# Build an NLTK Vocabulary from training sentences with an <UNK> cutoff.
def build_vocab(train_sents, unk_cutoff=UNK_CUTOFF):
    # Flatten sentence tokens into a single list for vocabulary building.
    flat = [w for sent in train_sents for w in sent]
    # Create and return a vocabulary that maps rare tokens to <UNK>.
    return Vocabulary(flat, unk_cutoff=unk_cutoff)


# Create fixed-length padded n-grams for perplexity evaluation.
def make_test_ngrams(sents, n):
    # Initialize a list to accumulate n-grams from all sentences.
    grams = []
    # Iterate over each tokenized sentence.
    for s in sents:
        # Pad both ends of the sentence to allow boundary n-grams.
        padded = list(pad_both_ends(s, n))
        # Generate fixed-length n-grams and append them to the global list.
        grams.extend(list(everygrams(padded, min_len=n, max_len=n)))
    # Return the list of n-grams.
    return grams


# Train a Kneser-Ney interpolated n-gram language model and return it with its vocabulary.
def train_kn_model(train_sents, n):
    # Print a simple checkpoint marker.
    print("1")
    # Build vocabulary from training data to define <UNK> mapping.
    vocab = build_vocab(train_sents, unk_cutoff=UNK_CUTOFF)
    # Print a simple checkpoint marker.
    print("2")
    # Map training sentences through vocab.lookup so rare tokens become <UNK>.
    train_sents_mapped = [list(vocab.lookup(sent)) for sent in train_sents]
    # Print a simple checkpoint marker.
    print("3")
    # Create the padded n-gram training stream and corresponding vocabulary stream.
    train_data, _ = padded_everygram_pipeline(n, train_sents_mapped)
    # Print a simple checkpoint marker.
    print("4")
    # Instantiate a Kneser-Ney interpolated language model of order n.
    model = KneserNeyInterpolated(order=n)
    # Print a simple checkpoint marker.
    print("5")
    # Fit the model to the training n-grams using the provided vocabulary.
    model.fit(train_data, vocab)
    # Print a simple checkpoint marker.
    print("6")
    # Return both the fitted model and the vocabulary used for mapping.
    return model, vocab


# Evaluate held-out perplexity by training on train_sents and scoring test_sents.
def evaluate_ppl(train_sents, test_sents, n):
    # Return None when either split is empty (cannot evaluate).
    if (not train_sents) or (not test_sents):
        return None
    # Train the language model and obtain its vocabulary.
    model, vocab = train_kn_model(train_sents, n)
    # Print a simple checkpoint marker.
    print("7")
    # Map test sentences through the same vocabulary to ensure consistent <UNK> handling.
    test_sents_mapped = [list(vocab.lookup(sent)) for sent in test_sents]
    # Print a simple checkpoint marker.
    print("8")
    # Convert test sentences into fixed-length n-grams for perplexity calculation.
    test_grams = make_test_ngrams(test_sents_mapped, n)
    # Print a simple checkpoint marker.
    print("9")
    # Return perplexity over the held-out n-grams.
    return model.perplexity(test_grams)


# Export the trained model, vocabulary, config, and top token counts into an output directory.
def export_lm_pack(
    model: KneserNeyInterpolated,
    vocab: Vocabulary,
    out_dir: Path,
    config: dict,
    top_tokens: List[tuple],
):
    # Create the output directory if it does not exist.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Open the model output file and serialize the trained model.
    with (out_dir / "bigram_kn_model.pkl").open("wb") as f:
        pickle.dump(model, f)
    # Open the vocabulary output file and serialize the vocabulary.
    with (out_dir / "lm_vocab.pkl").open("wb") as f:
        pickle.dump(vocab, f)
    # Open the config output file and write the JSON metadata.
    with (out_dir / "lm_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    # Open the token summary output file and write token counts line-by-line.
    with (out_dir / "top_tokens.txt").open("w", encoding="utf-8") as f:
        for w, c in top_tokens:
            f.write(f"{w}\t{c}\n")


# Run the end-to-end pipeline: load corpora, preprocess, train bigram LM, and export artifacts.
def main(
    corpus_dir: Optional[str],
    corpus_csv: Optional[str],
    text_col: str,
    out_dir: str,
    remove_stopwords: bool,
    use_lemma: bool,
    use_pos_lemma: bool,
    eval_ppl_flag: bool,
):
    # Initialize combined corpus list.
    texts: List[str] = []
    # Initialize a list for non-financial (normal) text samples.
    normal_texts: List[str] = []
    # Initialize a list for financial text samples.
    financial_texts: List[str] = []
    # Load texts from a directory structure if provided.
    if corpus_dir:
        # Convert the directory string to a Path.
        dir_path = Path(corpus_dir)
        # Validate directory existence.
        if not dir_path.exists():
            raise FileNotFoundError(f"corpus_dir not found: {dir_path}")
        # Load financial texts from the "financial" subfolder.
        financial_texts.extend(load_txt_dir(Path(dir_path, "financial")))
        # Load normal texts from the "normal" subfolder.
        normal_texts.extend(load_txt_dir(Path(dir_path, "normal")))
    # Load texts from a CSV file if provided.
    if corpus_csv:
        # Convert the CSV path string to a Path.
        csv_path = Path(corpus_csv)
        # Validate file existence.
        if not csv_path.exists():
            raise FileNotFoundError(f"corpus_csv not found: {csv_path}")
        # Load texts from the specified column, using a no-header schema if required.
        financial_texts.extend(load_csv_texts(csv_path, text_col=text_col, no_header=True))
    # Fail early if no texts were loaded at all.
    if not financial_texts:
        raise ValueError("No corpus texts loaded. Provide --corpus_dir and/or --corpus_csv.")
    # Print the number of financial texts loaded.
    print(f"length of texts: {len(financial_texts)}")
    # Down-sample normal texts if they exceed the maximum sentence budget.
    if len(normal_texts) > MAX_SENTS:
        print(f"共 {len(normal_texts)} 条普通文本，随机采样 {MAX_SENTS} 条用于训练 LM。")
        normal_texts = random.sample(normal_texts, MAX_SENTS)
    # Merge both sources into one training list.
    texts = normal_texts + financial_texts
    # Print the merged corpus size.
    print(f"new length of texts: {len(texts)}")
    # Print the per-source counts for transparency.
    print(f"其中普通文本{len(normal_texts)}条，金融文本{len(financial_texts)}条")
    # Print a stage marker for preprocessing.
    print("-------- Preprocess --------")
    # Convert raw texts into tokenized sentences using the configured preprocessing options.
    sents = preprocess_texts(
        texts,
        is_tweet=False,
        remove_stopwords=remove_stopwords,
        use_lemma=use_lemma,
        use_pos_lemma=use_pos_lemma,
    )
    # Print how many sentences remain after preprocessing.
    print(f"Sentences after preprocessing: {len(sents)}")
    # Print a stage marker for final full-data training.
    print("-------- Final training on ALL sentences --------")
    # Train the bigram Kneser-Ney model on all processed sentences.
    model, vocab = train_kn_model(sents, ORDER)
    # Print a stage marker for token statistics.
    print("-------- Top tokens summary --------")
    # Flatten tokens across all sentences for frequency counting.
    flat = [w for sent in sents for w in sent]
    # Compute the top 50 most common tokens.
    top_tokens = Counter(flat).most_common(50)
    # Print a stage marker for export.
    print("-------- Export Pack --------")
    # Build a configuration dictionary to record training and preprocessing choices.
    config = {
        "order": ORDER,
        "smoothing": "KneserNeyInterpolated",
        "unk_cutoff": UNK_CUTOFF,
        "remove_stopwords": remove_stopwords,
        "use_lemma": use_lemma,
        "use_pos_lemma": use_pos_lemma,
        "inputs": {"corpus_dir": corpus_dir, "corpus_csv": corpus_csv, "text_col": text_col},
        "notes": "Built for A1 spell correction bigram LM pack.",
    }
    # Write model artifacts and summaries to disk.
    export_lm_pack(model=model, vocab=vocab, out_dir=Path(out_dir), config=config, top_tokens=top_tokens)
    # Print a success separator line.
    print("=" * 50)
    # Print a success message.
    print("Bigram LM Pack built successfully!")
    # Print the output directory path.
    print(f"Output dir: {out_dir}")
    # Print the list of files expected to be generated.
    print("Files generated:")
    # Print the model artifact filename.
    print(" - bigram_kn_model.pkl")
    # Print the vocabulary artifact filename.
    print(" - lm_vocab.pkl")
    # Print the config artifact filename.
    print(" - lm_config.json")
    # Print the token summary filename.
    print(" - top_tokens.txt")


# Run the script entrypoint when executed as a program.
if __name__ == "__main__":
    # Execute the pipeline using the current hard-coded parameters.
    main(
        corpus_dir="../resources/corpus",
        corpus_csv="../resources/corpus/financial/all-data.csv",
        text_col="text",
        out_dir="../resources/models",
        remove_stopwords=False,
        use_lemma=False,
        use_pos_lemma=False,
        eval_ppl_flag=True,
    )
