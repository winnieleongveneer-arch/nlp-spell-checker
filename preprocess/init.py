from preprocess.dataclass import *
# ------------------------------------------------------------
# Robust regex loader that tolerates BOM/leading whitespace
# and will retry compilation with VERBOSE if needed
# ------------------------------------------------------------
_resources: "ResourceHandles | None" = None

def _init_regex(name: str, flags: int = 0) -> re.Pattern:
    # Build the on-disk path for this regex
    regex_file = Path(REGEX_PATH) / f"{name}.txt"
    # Ensure file exists
    if not regex_file.exists():
        raise FileNotFoundError(f"Regex file not found: {regex_file}")
    # Read file content as UTF-8
    pattern_text = regex_file.read_text(encoding="utf-8")
    # Strip BOM and any leading whitespace so inline flags at column 1 work
    pattern_text = pattern_text.lstrip("\ufeff").lstrip()
    # First try: compile with given flags
    try:
        return re.compile(pattern_text, flags)
    except re.error:
        # Retry with VERBOSE added (useful when the file has line breaks)
        try:
            return re.compile(pattern_text, flags | re.VERBOSE)
        except re.error as e:
            # Give an actionable error message with file context
            raise re.error(
                f"Failed to compile regex '{name}' from {regex_file}. "
                f"Ensure inline flags like (?x)/(?i) appear at the very start. "
                f"Original error: {e}"
            )


# ------------------------------------------------------------
# Stopword loader (expects one token per line)
# ------------------------------------------------------------
def _init_stopwords() -> Set[str]:
    # Build on-disk path for stopwords file
    sw_file = Path(STOPWORDS_PATH) / "STOPWORDS.txt"
    # Ensure file exists
    if not sw_file.exists():
        raise FileNotFoundError(f"Stopwords file not found: {sw_file}")
    # Read and split lines, strip trailing whitespace, ignore empties/comments
    raw = sw_file.read_text(encoding="utf-8").splitlines()
    # Return as a set for O(1) membership checks
    return {ln.strip() for ln in raw if ln.strip() and not ln.strip().startswith("#")}


# ------------------------------------------------------------
# Singleton initializer for all resources (regex + stopwords)
# ------------------------------------------------------------
def _init_resources() -> ResourceHandles:
    # Use global cache
    global _resources
    # If already initialized, return cached resources
    if _resources is not None:
        return _resources

    # Precompute the common flags we want for file-based regex patterns
    common_flags = re.IGNORECASE | re.VERBOSE

    # Load regexes from files (each pattern may contain inline flags if needed)
    URL_REGEX               = _init_regex("URL_REGEX", common_flags)
    EMAIL_REGEX             = _init_regex("EMAIL_REGEX", common_flags)
    PHONE_REGEX             = _init_regex("PHONE_REGEX", common_flags)
    CONTROL_CHARS           = _init_regex("CONTROL_CHARS", common_flags)
    UPPER_PROTECT           = _init_regex("UPPER_PROTECT", common_flags)
    NUMBER_REGEX            = _init_regex("NUMBER_REGEX", common_flags)
    CURRENCY_AMOUNT_REGEX   = _init_regex("CURRENCY_AMOUNT_REGEX", common_flags)
    PERCENT_REGEX           = _init_regex("PERCENT_REGEX", common_flags)
    PURE_NUMBER_REGEX       = _init_regex("PURE_NUMBER_REGEX", common_flags)

    # Compile small inline regexes directly in code (no files needed here)
    REPEAT_PUNCT = re.compile(r"([!?.])\1{1,}")          # compress repeated . ! ?
    SEP_RUNS     = re.compile(r"([\-_/=]{2,})")          # collapse long separator runs
    FANCY_QUOTES = re.compile(r"[“”]")                   # curly quotes -> straight
    FANCY_APOS   = re.compile(r"[‘’]")                   # curly apostrophes -> straight

    # Load stopwords (as a set)
    STOPWORDS = _init_stopwords()

    # Create the ResourceHandles object and cache it
    _resources = ResourceHandles(
        URL_REGEX=URL_REGEX,
        EMAIL_REGEX=EMAIL_REGEX,
        PHONE_REGEX=PHONE_REGEX,
        CONTROL_CHARS=CONTROL_CHARS,
        UPPER_PROTECT=UPPER_PROTECT,
        NUMBER_REGEX=NUMBER_REGEX,
        CURRENCY_AMOUNT_REGEX=CURRENCY_AMOUNT_REGEX,
        PERCENT_REGEX=PERCENT_REGEX,
        PURE_NUMBER_REGEX=PURE_NUMBER_REGEX,
        REPEAT_PUNCT=REPEAT_PUNCT,
        SEP_RUNS=SEP_RUNS,
        FANCY_QUOTES=FANCY_QUOTES,
        FANCY_APOS=FANCY_APOS,
        STOPWORDS=STOPWORDS
    )
    # Return the singleton instance
    return _resources

def get_resources() -> ResourceHandles:
    return _init_resources()


# ★ 如果你仍然想保留 import * 的写法
__all__ = [
    "get_resources",
    "_init_resources",
    "_resources",
]