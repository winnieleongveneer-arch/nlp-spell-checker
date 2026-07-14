from preprocess.dataclass import *
from preprocess import init

_resources = init.get_resources()
# ------------------------------------------------------------
# Step 1: Unicode normalization and control-char cleanup
# ------------------------------------------------------------
def _normalize_unicode(text: str) -> str:
    # Normalize unicode to NFKC to fold compatibility characters
    s = unicodedata.normalize("NFKC", text)
    # Normalize newlines to "\n"
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Remove control chars using the loaded regex
    s = _resources.CONTROL_CHARS.sub("", s)
    # Replace any unicode separator categories (Z*) with a space
    buf = []
    for ch in s:
        if unicodedata.category(ch).startswith("Z"):
            buf.append(" ")
        else:
            buf.append(ch)
    # Join back into a string
    s = "".join(buf)
    # Collapse whitespace at the end
    #return _normalize_whitespace(s)
    return s
