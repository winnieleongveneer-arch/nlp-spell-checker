from preprocess.dataclass import *
from preprocess import init
from preprocess.sentence_splitter import _get_spacy
_resources = init.get_resources()
# ------------------------------------------------------------
# Step 7: Tokenizer: spaCy / NLTK / whitespace fallback
# ------------------------------------------------------------

# ------------------------------------------------------------
# Placeholder protection for tokenization
# ------------------------------------------------------------
_PLACEHOLDER = re.compile(r"<(?:URL|EMAIL|PHONE|NUM|PCT|CURNUM|BPS)>")

def _protect_placeholders(text: str) -> str:
    # Replace placeholders by a private-use separator to prevent tokenizer splits
    return (
        text.replace("<URL>",    "\uE000")
            .replace("<EMAIL>",  "\uE001")
            .replace("<PHONE>",  "\uE002")
            .replace("<NUM>",    "\uE003")
            .replace("<PCT>",    "\uE004")
            .replace("<CURNUM>", "\uE005")
            .replace("<BPS>",    "\uE006")
    )


def _restore_placeholders_token(tok: str) -> str:
    # Map private-use codepoints back to their textual placeholders
    return (
        tok.replace("\uE000", "<URL>")
           .replace("\uE001", "<EMAIL>")
           .replace("\uE002", "<PHONE>")
           .replace("\uE003", "<NUM>")
           .replace("\uE004", "<PCT>")
           .replace("\uE005", "<CURNUM>")
           .replace("\uE006", "<BPS>")
    )


# ------------------------------------------------------------
# Tokenizer: spaCy / NLTK / whitespace fallback
# ------------------------------------------------------------
def _ensure_nltk_word_tokenize():
    # Import inside function to avoid hard dependency at module import time
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
    except Exception as e:
        raise RuntimeError("NLTK not installed. Please `pip install nltk`.") from e


def _tokenize(
    sentences: List[str],
    *,
    backend: Backend,
    language: str = "english",
    spacy_model: str = "en_core_web_sm",
) -> List[List[str]]:
    # If using spaCy backend for tokenization
    if backend == Backend.SPACY:
        nlp = _get_spacy(spacy_model)
        out: List[List[str]] = []
        for sent in sentences:
            # Protect placeholders before tokenization
            protected = _protect_placeholders(sent)
            # Tokenize via spaCy
            doc = nlp(protected)
            # Extract tokens as strings
            toks = [t.text for t in doc]
            # Restore placeholders and strip empties
            toks = [_restore_placeholders_token(t).strip() for t in toks if t.strip()]
            # Append sentence-level list
            out.append(toks)
        # Return list of token lists
        return out
    # If using NLTK backend for tokenization
    if backend == Backend.NLTK:
        _ensure_nltk_word_tokenize()
        from nltk.tokenize import word_tokenize
        out: List[List[str]] = []
        for sent in sentences:
            # Protect placeholders
            protected = _protect_placeholders(sent)
            # Tokenize with NLTK
            toks = word_tokenize(protected, language=language)
            # Restore placeholders and strip blanks
            toks = [_restore_placeholders_token(t).strip() for t in toks if t.strip()]
            # Append
            out.append(toks)
        # Return
        return out
    # As a last resort, split on whitespace
    out: List[List[str]] = []
    for sent in sentences:
        protected = _protect_placeholders(sent)
        toks = [t for t in protected.split() if t]
        toks = [_restore_placeholders_token(t) for t in toks]
        out.append(toks)
    return out
