from preprocess.dataclass import *
from preprocess import init

_resources = init.get_resources()
# ------------------------------------------------------------
# Step 2: URL / Email / Phone removal or placeholdering
# ------------------------------------------------------------
def _remove_urls_emails_phones(
    text: str,
    *,
    remove_urls: bool,
    remove_emails: bool,
    remove_phones: bool,
    url_placeholder: Optional[str],
    email_placeholder: Optional[str],
    phone_placeholder: Optional[str],
) -> str:
    # Start with the raw text
    s = text
    # Remove or replace URLs
    if remove_urls:
        s = _resources.URL_REGEX.sub("", s)
    else:
        placeholder = "<URL>" if url_placeholder is None else url_placeholder
        s = _resources.URL_REGEX.sub(placeholder, s)
    # Remove or replace Emails
    if remove_emails:
        s = _resources.EMAIL_REGEX.sub("", s)
    else:
        placeholder = "<EMAIL>" if email_placeholder is None else email_placeholder
        s = _resources.EMAIL_REGEX.sub(placeholder, s)
    # Remove or replace Phones
    if remove_phones:
        s = _resources.PHONE_REGEX.sub("", s)
    else:
        placeholder = "<PHONE>" if phone_placeholder is None else phone_placeholder
        s = _resources.PHONE_REGEX.sub(placeholder, s)
    # Normalize whitespace after deletions/replacements
    #return _normalize_whitespace(s)
    return s


# ------------------------------------------------------------
# Protection utilities: preserve whitelisted uppercase spans
# ------------------------------------------------------------
def _apply_protection(text: str) -> Tuple[str, Dict[str, str]]:
    # Initialize mapping dict from placeholder -> original span
    mapping: Dict[str, str] = {}
    # Index counter for placeholders
    idx = 0

    # Replacement function called by regex .sub
    def repl(m: re.Match) -> str:
        nonlocal idx
        # Create a placeholder that won't be changed by .lower()
        placeholder = f"<<P{idx}>>"
        # Record the original span
        mapping[placeholder] = m.group(0)
        # Increment index
        idx += 1
        # Return placeholder to be inserted into text
        return placeholder

    # Replace all whitelisted spans with placeholders
    protected = _resources.UPPER_PROTECT.sub(repl, text)
    # Return protected text and the mapping
    return protected, mapping


def _restore_protection(text: str, mapping: Dict[str, str]) -> str:
    # If there is nothing to restore, return as-is
    if not mapping:
        return text
    # Sort keys by length desc to avoid nested/partial replacement issues
    keys = sorted(mapping.keys(), key=len, reverse=True)
    # Build a combined regex that matches any placeholder
    pattern = re.compile("|".join(map(re.escape, keys)))
    # Replace placeholders by original spans via a callback
    return pattern.sub(lambda m: mapping[m.group(0)], text)
