from preprocess.dataclass import *
from preprocess import init

_resources = init.get_resources()
# ------------------------------------------------------------
# Step 6: Sentence splitter: spaCy / NLTK / regex fallback
# ------------------------------------------------------------
def _get_spacy(model: str):
    # Import inside function to avoid hard dependency at module import time
    try:
        import spacy
    except Exception as e:
        raise RuntimeError("spaCy not installed. Please `pip install spacy`.") from e
    try:
        return spacy.load(model)
    except Exception:
        # Fall back to a blank English model when the small model is unavailable
        nlp = spacy.blank("en")
        # Ensure we have a sentencizer if no parser/senter is present
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp


def _ensure_nltk_punkt(language: str = "english"):
    # Import inside function to avoid hard dependency at module import time
    try:
        import nltk
        # Try to locate punkt; download if missing
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
    except Exception as e:
        raise RuntimeError("NLTK not installed. Please `pip install nltk`.") from e

# ---------- helpers for chunked, lightweight sentence splitting ----------

def _yield_chunks(text: str, max_chars: int = 200_000):
    """Yield ~max_chars sized chunks, trying to cut at a newline/space for cleaner boundaries."""
    n = len(text)
    i = 0
    while i < n:
        j = min(n, i + max_chars)
        # try to cut on a nicer boundary near the end of the window
        cut = text.rfind("\n", i, j)
        if cut == -1:
            cut = text.rfind(" ", i, j)
        if cut == -1 or cut <= i + max_chars // 2:
            cut = j
        yield text[i:cut]
        i = cut

def _get_spacy_sentencizer(lang_code: str = "en", max_length: int = 2_000_000):
    """A tiny spaCy pipeline with only 'sentencizer' for fast, low-memory sentence segmentation."""
    import spacy
    nlp = spacy.blank(lang_code)
    nlp.max_length = max_length
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


# ---------- optimized sentence segmentation (chunked + sentencizer) ----------

def _split_sentences(
    text: str,
    *,
    backend: Backend,
    language: str = "english",
    spacy_model: str = "en_core_web_sm",
) -> List[str]:
    """
    Sentence segmentation with three backends:
      - Backend.SPACY: use a lightweight blank('en') + sentencizer, and process the text in chunks
                       to avoid E088 (max_length) and reduce memory.
      - Backend.NLTK:  use Punkt tokenizer.
      - Fallback:       simple regex split.
    """
    # --- spaCy backend (chunked + light sentencizer) ---
    if backend == Backend.SPACY:
        # build a tiny pipeline that only detects sentence boundaries
        nlp = _get_spacy_sentencizer(lang_code="en", max_length=2_000_000)
        out: List[str] = []
        # iterate over chunks to keep memory usage bounded
        for chunk in _yield_chunks(text, max_chars=200_000):
            # make sure limit is always larger than current chunk
            if len(chunk) + 1 > nlp.max_length:
                nlp.max_length = len(chunk) + 1
            doc = nlp(chunk)
            out.extend([s.text.strip() for s in doc.sents if s.text.strip()])
        return out

    # --- NLTK backend ---
    if backend == Backend.NLTK:
        _ensure_nltk_punkt(language=language)
        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text, language=language) if s.strip()]

    # --- very small regex fallback ---
    fallback = re.compile(r'(?<!\b[A-Z])(?<=[.!?\u2026])["\')\]]*\s+')
    parts = fallback.split(text)
    return [p.strip() for p in parts if p.strip()]


