# preprocessing.py
# ------------------------------------------------------------
# End-to-end, teaching-style preprocessing pipeline
# - Robust regex loader (_init_regex) that tolerates BOM/whitespace
# - Regex files compiled with IGNORECASE|VERBOSE to avoid inline-flag errors
# - Steps 1–12 exposed via helper functions and a single `preprocess` entrypoint
# ------------------------------------------------------------

from __future__ import annotations  # allow forward type annotations
from dataclasses import dataclass    # lightweight containers for resources
from enum import Enum                # typed enums for backend/process_type
from pathlib import Path             # file path utilities
from typing import (                  # type hints
    List, Dict, Set, Tuple, Optional, Union
)
import re                            # regular expressions
import unicodedata                   # unicode normalization utilities
import html                          # optional safety for text
import sys                           # environment checks

# Optional libraries – imported lazily when actually used
# (do not import spaCy / NLTK at module import time to avoid env errors)

# -------------------------------
# Enums for configuration knobs
# -------------------------------
class ProcessType(str, Enum):
    LEMMATIZATION = "lemmatization"
    STEMMING      = "stemming"


class Backend(str, Enum):
    SPACY = "spacy"
    NLTK  = "nltk"


# -----------------------------------------
# Resource container (regexes, stopwords…)
# -----------------------------------------
@dataclass
class ResourceHandles:
    URL_REGEX: re.Pattern
    EMAIL_REGEX: re.Pattern
    PHONE_REGEX: re.Pattern
    CONTROL_CHARS: re.Pattern
    UPPER_PROTECT: re.Pattern
    NUMBER_REGEX: re.Pattern
    CURRENCY_AMOUNT_REGEX: re.Pattern
    PERCENT_REGEX: re.Pattern
    PURE_NUMBER_REGEX: re.Pattern
    REPEAT_PUNCT: re.Pattern
    SEP_RUNS: re.Pattern
    FANCY_QUOTES: re.Pattern
    FANCY_APOS: re.Pattern
    STOPWORDS: Set[str]

# -------------------------------
# Global configuration / caches
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_PATH = PROJECT_ROOT / "resources"
REGEX_PATH = RESOURCE_PATH / "regex"
STOPWORDS_PATH = RESOURCE_PATH / "stopwords"

_resources: Optional[ResourceHandles] = None  # singleton cache

