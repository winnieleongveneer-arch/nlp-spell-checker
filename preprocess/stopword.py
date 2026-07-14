from preprocess.dataclass import *
from preprocess import init
from preprocess.tokenizer import _PLACEHOLDER
_resources = init.get_resources()
# ------------------------------------------------------------
# Step 10: stopword removal (optional; usually False for LM/spell-check)
# ------------------------------------------------------------
def _remove_stopwords_tokens(
    tokenized: List[List[str]],
    *,
    remove_stopwords: bool,
    stopwords: Optional[Set[str]],
) -> List[List[str]]:
    # If removal is not requested, return unchanged
    if not remove_stopwords:
        return tokenized
    # Determine which stopword set to use (custom or default)
    sw = stopwords if stopwords is not None else _resources.STOPWORDS
    # Filter tokens that are not placeholders and appear in stopword set
    out: List[List[str]] = []
    for sent in tokenized:
        row = [t for t in sent if (_PLACEHOLDER.fullmatch(t) or t.lower() not in sw)]
        out.append(row)
    # Return stopword-filtered tokens
    return out

