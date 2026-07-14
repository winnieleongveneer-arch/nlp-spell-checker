from preprocess.init import *
from preprocess.digits_punct import _coarse_digits_punct
from preprocess.lowercasing import _lowercase
from preprocess.morphology import _apply_morphology
from preprocess.normalization import _normalize_unicode
from preprocess.oov_mapping import _apply_oov_mapping
from preprocess.sentence_splitter import _split_sentences
from preprocess.stopword import _remove_stopwords_tokens
from preprocess.token_filtering import _filter_tokens
from preprocess.tokenizer import _tokenize
from preprocess.urls_emails_phones import _remove_urls_emails_phones
from preprocess.whitespace import _normalize_whitespace
from preprocess.whitespace import *
# ------------------------------------------------------------
# main orchestration function
# ------------------------------------------------------------
def preprocess_pipeline(
    text: str,
    *,
    lowercase: bool = True,
    keep_punct: bool = False,
    keep_digits: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_phones: bool = False,
    process_type: ProcessType = ProcessType.LEMMATIZATION,
    backend: Backend = Backend.SPACY,
    # Advanced knobs
    language: str = "english",
    spacy_model: str = "en_core_web_sm",
    return_by_sentence: bool = False,
    # Placeholders (None means "delete" for URLs/Emails/Phones)
    url_placeholder: Optional[str] = None,
    email_placeholder: Optional[str] = None,
    phone_placeholder: Optional[str] = None,
    replace_digits_with: Optional[str] = None,
    remove_stopwords: bool = False,
    stopwords: Optional[Set[str]] = None,
    apply_oov_mapping: bool = False,
    vocab_for_oov: Optional[Set[str]] = None,
    oov_token: str = "<unk>",
) -> Union[List[str], List[List[str]]]:
    # Ensure regexes/stopwords are loaded once (singleton)
    global _resources
    _resources = _init_resources()

    # Step 1: Unicode normalization / control characters cleanup
    s = _normalize_unicode(text)

    # Step 2: Remove or placeholder URLs/Emails/Phones, then re-normalize spaces
    s = _remove_urls_emails_phones(
        s,
        remove_urls=remove_urls,
        remove_emails=remove_emails,
        remove_phones=remove_phones,
        url_placeholder=url_placeholder,
        email_placeholder=email_placeholder,
        phone_placeholder=phone_placeholder,
    )

    # Step 3: Lowercasing with uppercase-protection
    s = _lowercase(s, lowercase=lowercase)

    # Step 4: Typed numeric placeholdering + light punctuation normalization
    s = _coarse_digits_punct(
        s,
        keep_digits=keep_digits,
        replace_digits_with=replace_digits_with,
    )

    # Step 5: Final whitespace normalization at string level
    s = _normalize_whitespace(s)

    # Step 6: Sentence segmentation (spaCy / NLTK / fallback)
    sentences = _split_sentences(
        s,
        backend=backend,
        language=language,
        spacy_model=spacy_model,
    )

    # Step 7: Tokenization (spaCy / NLTK / fallback)
    tokenized = _tokenize(
        sentences,
        backend=backend,
        language=language,
        spacy_model=spacy_model,
    )

    # Step 8: Morphology (lemmatization or stemming)
    tokenized = _apply_morphology(
        tokenized,
        process_type=process_type,
        backend=backend,
        language=language,
        spacy_model=spacy_model,
    )

    # Step 9: Fine-grained token filtering for punctuation/digits
    tokenized = _filter_tokens(
        tokenized,
        keep_punct=keep_punct,
        keep_digits=keep_digits,
        replace_digits_with=replace_digits_with,
    )

    # Step 10: Stopword removal (optional)
    tokenized = _remove_stopwords_tokens(
        tokenized,
        remove_stopwords=remove_stopwords,
        stopwords=stopwords,
    )

    # Step 11: OOV mapping (optional)
    tokenized = _apply_oov_mapping(
        tokenized,
        apply_oov_mapping=apply_oov_mapping,
        vocab_for_oov=vocab_for_oov,
        oov_token=oov_token,
    )

    # Step 12: Output shape (flatten or grouped by sentence)
    if return_by_sentence:
        return tokenized
    # Flatten sentence lists into a single token list
    return [t for sent in tokenized for t in sent]

def read_text_safely(path: str | Path) -> str:
    path = Path(path)
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            raise
    data = path.read_bytes()
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return data.decode("latin-1", errors="replace")