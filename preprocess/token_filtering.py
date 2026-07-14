from preprocess.dataclass import *
from preprocess import init
from preprocess.tokenizer import _PLACEHOLDER
_resources = init.get_resources()
# ------------------------------------------------------------
# Step 9: fine-grained token filtering (punct / digits)
# ------------------------------------------------------------
_PUNCT_ONLY = re.compile(r"^[^\w\s]+$")  # safe ASCII-class fallback

def _filter_tokens(
    tokenized: List[List[str]],
    *,
    keep_punct: bool,
    keep_digits: bool,
    replace_digits_with: Optional[str],
) -> List[List[str]]:
    # Prepare output container
    out: List[List[str]] = []
    # Iterate sentence by sentence
    for sent in tokenized:
        row: List[str] = []
        for tok in sent:
            # Keep placeholders always
            if _PLACEHOLDER.fullmatch(tok):
                row.append(tok)
                continue
            # Decide punctuation handling
            if not keep_punct and _PUNCT_ONLY.fullmatch(tok):
                # Drop tokens that are pure punctuation
                continue
            # Decide digit handling if not already replaced
            if not keep_digits and replace_digits_with is not None:
                if any(ch.isdigit() for ch in tok):
                    row.append(replace_digits_with)
                    continue
            # Otherwise keep token as-is
            row.append(tok)
        out.append(row)
    # Return filtered tokens
    return out

