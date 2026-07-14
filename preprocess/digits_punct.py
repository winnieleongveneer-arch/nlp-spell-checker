from preprocess.dataclass import *
from preprocess import init

_resources = init.get_resources()
# ------------------------------------------------------------
# Step 4: typed numeric placeholdering + light punctuation clean
# ------------------------------------------------------------
def _coarse_digits_punct(
    text: str,
    *,
    keep_digits: bool,
    replace_digits_with: Optional[str],
) -> str:
    # Start with input
    s = text
    # Normalize fancy quotes to straight
    s = _resources.FANCY_QUOTES.sub('"', s)
    s = _resources.FANCY_APOS.sub("'", s)
    # If you want to force-replace all numbers irrespective of type, do it now
    if replace_digits_with is not None:
        s = _resources.NUMBER_REGEX.sub(replace_digits_with, s)
    else:
        # Otherwise only replace specific numeric types when keep_digits is False
        if not keep_digits:
            s = _resources.CURRENCY_AMOUNT_REGEX.sub("<CURNUM>", s)
            s = _resources.PERCENT_REGEX.sub("<PCT>", s)
            s = _resources.PURE_NUMBER_REGEX.sub("<NUM>", s)
    # Compress repeated sentence-ending punctuation like "!!!"
    s = _resources.REPEAT_PUNCT.sub(r"\1", s)
    # Collapse long runs of separators "-_/=" into a single space
    s = _resources.SEP_RUNS.sub(" ", s)
    # Normalize whitespace
    #s = _normalize_whitespace(s)
    # Return processed string
    return s
