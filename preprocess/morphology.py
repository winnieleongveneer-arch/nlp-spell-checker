from preprocess.dataclass import *
from preprocess import init
from preprocess.sentence_splitter import _get_spacy
from preprocess.tokenizer import _PLACEHOLDER
_resources = init.get_resources()

# ------------------------------------------------------------
# Step 8: morphology (lemmatization or stemming)
#   Supports: spaCy + Lemmatization, spaCy + Stemming (simple rule),
#             NLTK + Lemmatization (WordNet), NLTK + Stemming (Snowball)
# ------------------------------------------------------------
def _apply_morphology(
    tokenized: List[List[str]],
    *,
    process_type: ProcessType,
    backend: Backend,
    language: str = "english",
    spacy_model: str = "en_core_web_sm",
) -> List[List[str]]:
    # If we are using spaCy backend
    if backend == Backend.SPACY:
        nlp = _get_spacy(spacy_model)
        # If lemmatization is requested
        if process_type == ProcessType.LEMMATIZATION:
            out: List[List[str]] = []
            for sent in tokenized:
                # Skip placeholders from being altered semantically
                doc = nlp(" ".join(sent))
                # Map each token to its lemma if not a placeholder
                lemmas: List[str] = []
                for t in doc:
                    raw = t.text
                    if _PLACEHOLDER.fullmatch(raw):
                        lemmas.append(raw)
                    else:
                        lemmas.append(t.lemma_ if t.lemma_ else raw)
                out.append(lemmas)
            return out
        # If stemming is requested with spaCy (no built-in stemmer; do a light rule)
        if process_type == ProcessType.STEMMING:
            # Very light suffix stripper as a placeholder (use NLTK Snowball for real stemming)
            def light_stem(tok: str) -> str:
                if _PLACEHOLDER.fullmatch(tok):
                    return tok
                # Simple rule-based (for demonstration)
                for suf in ("ing", "ed", "ly", "es", "s"):
                    if tok.lower().endswith(suf) and len(tok) > len(suf) + 2:
                        return tok[: -len(suf)]
                return tok
            return [[light_stem(t) for t in sent] for sent in tokenized]
        # Default: return unchanged if something else
        return tokenized

    # If we are using NLTK backend
    if backend == Backend.NLTK:
        try:
            import nltk
            # Ensure WordNet and Punkt are available
            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                nltk.download("wordnet")
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")
            # Lemmatization path
            if process_type == ProcessType.LEMMATIZATION:
                from nltk.stem import WordNetLemmatizer
                lemm = WordNetLemmatizer()
                out: List[List[str]] = []
                for sent in tokenized:
                    row = []
                    for t in sent:
                        if _PLACEHOLDER.fullmatch(t):
                            row.append(t)
                        else:
                            row.append(lemm.lemmatize(t))
                    out.append(row)
                return out
            # Stemming path (Snowball)
            if process_type == ProcessType.STEMMING:
                from nltk.stem.snowball import SnowballStemmer
                stemmer = SnowballStemmer("english")
                return [[t if _PLACEHOLDER.fullmatch(t) else stemmer.stem(t) for t in sent]
                        for sent in tokenized]
        except Exception as e:
            raise RuntimeError("NLTK is required for the selected morphology.") from e

    # If neither condition matched, return tokens unchanged
    return tokenized
