from preprocess.dataclass import *
from preprocess import init
from preprocess.urls_emails_phones import _apply_protection,_restore_protection

_resources = init.get_resources()
# ------------------------------------------------------------
# Step 3: optional lowercasing with uppercase-protection
# ------------------------------------------------------------
def _lowercase(text: str, lowercase: bool) -> str:
    # If lowercasing is disabled, return original text
    if not lowercase:
        return text
    # Protect uppercase spans we want to preserve
    s, mapping = _apply_protection(text)
    # Lowercase (casefold could be used if you want more aggressive folding)
    s = s.lower()
    # Restore protected uppercase spans
    s = _restore_protection(s, mapping)
    # Return final string
    return s
