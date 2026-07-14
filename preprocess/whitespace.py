from preprocess.dataclass import *
from preprocess import init

_resources = init.get_resources()
# ------------------------------------------------------------
# Step 5: Utility: whitespace normalization (collapse runs to single)
# ------------------------------------------------------------
def _normalize_whitespace(text: str) -> str:
    # Replace any run of whitespace with a single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading and trailing whitespace
    return text.strip()