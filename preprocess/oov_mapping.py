from preprocess.dataclass import *
from preprocess import init
from preprocess.tokenizer import _PLACEHOLDER
_resources = init.get_resources()
# ------------------------------------------------------------
# Step 11: OOV mapping (optional; typically for LM training)
# ------------------------------------------------------------
def _apply_oov_mapping(
    tokenized: List[List[str]],
    *,
    apply_oov_mapping: bool,
    vocab_for_oov: Optional[Set[str]],
    oov_token: str = "<unk>",
) -> List[List[str]]:
    # If OOV mapping is not requested or vocab is missing, return unchanged
    if not apply_oov_mapping or not vocab_for_oov:
        return tokenized
    # For each token, if not a placeholder and not in vocab, map to <unk>
    out: List[List[str]] = []
    for sent in tokenized:
        row = []
        for t in sent:
            if _PLACEHOLDER.fullmatch(t):
                row.append(t)
            else:
                row.append(t if t in vocab_for_oov else oov_token)
        out.append(row)
    # Return OOV-mapped tokens
    return out
