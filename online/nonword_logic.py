# Enable postponed evaluation of type annotations for forward references and faster imports.
from __future__ import annotations

# Import regular expressions for tokenization and pattern matching.
import re
# Import math for logarithms and exponentials used in scoring and softmax.
import math
# Import typing primitives for type annotations used across the module.
from typing import Dict, List, Tuple, Optional, Any, Set

# Import your Unicode normalization utility used to keep token forms consistent.
from preprocess.normalization import _normalize_unicode


# Compile a tokenizer regex that keeps alphanumeric terms together (e.g., 'VH1', 'Zap2it') and supports apostrophes.
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s]")


# Define a function that tokenizes text and returns tokens with character start/end offsets.
def tokenize_with_spans(text: str) -> List[Dict[str, Any]]:
    # Initialize an output list to hold token dictionaries.
    tokens: List[Dict[str, Any]] = []
    # Iterate over all regex matches and collect token and span boundaries.
    for m in WORD_RE.finditer(text):
        # Extract the matched token string.
        tok = m.group(0)
        # Append a record containing the token and its span offsets.
        tokens.append({"token": tok, "start": m.start(), "end": m.end()})
    # Return the full list of token records.
    return tokens


# Define a function that checks whether a token is strictly alphabetic.
def is_alpha_token(tok: str) -> bool:
    # Return True if the whole token matches letters only.
    return bool(re.fullmatch(r"[A-Za-z]+", tok))


# Define a constant label for fully uppercase words.
CASING_UPPER = "UPPER"
# Define a constant label for fully lowercase words.
CASING_LOWER = "LOWER"
# Define a constant label for title-case words (first letter uppercase, remainder lowercase).
CASING_TITLE = "TITLE"
# Define a constant label for mixed/other casing patterns.
CASING_OTHER = "OTHER"


# Define a function that detects the casing pattern of an input token.
def detect_casing(token: str) -> str:
    # Return the uppercase casing label if the token is all uppercase.
    if token.isupper():
        return CASING_UPPER
    # Return the lowercase casing label if the token is all lowercase.
    if token.islower():
        return CASING_LOWER
    # Return title casing label if it matches Title-case style and length is > 1.
    if len(token) > 1 and token[0].isupper() and token[1:].islower():
        return CASING_TITLE
    # Otherwise mark the token as other/mixed casing.
    return CASING_OTHER


# Define a function that applies a detected casing pattern to a candidate token.
def apply_casing(token: str, casing: str) -> str:
    # Convert candidate to uppercase if original casing was uppercase.
    if casing == CASING_UPPER:
        return token.upper()
    # Convert candidate to title-case if original casing was title-case.
    if casing == CASING_TITLE:
        return token.capitalize()
    # Convert candidate to lowercase if original casing was lowercase.
    if casing == CASING_LOWER:
        return token.lower()
    # For mixed casing, keep candidate in its current form to avoid awkward casing artifacts.
    return token


# Define a function that applies casing conversion to every candidate record in a list.
def apply_casing_to_list(candidates: List[Dict[str, Any]], casing: str) -> List[Dict[str, Any]]:
    # Iterate through candidate records and rewrite their "candidate" field.
    for it in candidates:
        # Read the candidate string from the dict and coerce to string.
        cand = str(it.get("candidate", ""))
        # Replace the candidate string with a casing-adjusted version.
        it["candidate"] = apply_casing(cand, casing)
    # Return the updated list (mutated in place).
    return candidates


# Define a function that decides whether a token should be excluded from spell checking.
def should_skip_token(tok: str) -> bool:
    # Skip tokens that contain any digit character.
    if any(ch.isdigit() for ch in tok):
        return True
    # Skip tokens that are not purely alphabetic.
    if not is_alpha_token(tok):
        return True
    # Otherwise do not skip the token.
    return False


# Define a function that computes standard Levenshtein edit distance with optional length-difference pruning.
def levenshtein(s1: str, s2: str, max_dist: Optional[int] = None) -> int:
    # Normalize both input strings to lowercase to compute case-insensitive edit distance.
    s1 = s1.lower()
    s2 = s2.lower()
    # Compute string lengths for DP matrix sizing and early exits.
    n, m = len(s1), len(s2)
    # Return 0 if the strings are identical after normalization.
    if s1 == s2:
        return 0
    # If s1 is empty, the distance is the length of s2.
    if n == 0:
        return m
    # If s2 is empty, the distance is the length of s1.
    if m == 0:
        return n
    # If max_dist is set and the length gap exceeds it, return a value guaranteed to be > max_dist.
    if max_dist is not None and abs(n - m) > max_dist:
        return max_dist + 1
    # Allocate a DP matrix where dp[i][j] is distance between s1[:i] and s2[:j].
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # Initialize the first column to represent deletions to match empty s2 prefix.
    for i in range(n + 1):
        dp[i][0] = i
    # Initialize the first row to represent insertions to match empty s1 prefix.
    for j in range(m + 1):
        dp[0][j] = j
    # Fill the DP matrix row by row.
    for i in range(1, n + 1):
        # Extract current character from s1 for substitution cost calculation.
        ci = s1[i - 1]
        # Fill the i-th row across all columns.
        for j in range(1, m + 1):
            # Extract current character from s2 for substitution cost calculation.
            cj = s2[j - 1]
            # Set substitution cost to 0 if characters match, otherwise 1.
            cost = 0 if ci == cj else 1
            # Compute best of delete, insert, and substitute operations.
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    # Return the final edit distance in the bottom-right DP cell.
    return dp[n][m]


# Map Penn Treebank POS tags to coarse groups for robust matching.
def pos_group(tag: Optional[str]) -> str:
    # Return the "other" group when tag is missing.
    if not tag:
        return "O"
    # Map nouns to the noun group.
    if tag.startswith("NN"):
        return "N"
    # Map verbs to the verb group.
    if tag.startswith("VB"):
        return "V"
    # Map adjectives to the adjective group.
    if tag.startswith("JJ"):
        return "J"
    # Map adverbs to the adverb group.
    if tag.startswith("RB"):
        return "R"
    # Default all other tags into the "other" group.
    return "O"


# Check whether candidate POS is compatible with target POS using coarse groups.
def pos_compatible(target_pos: Optional[str], cand_pos_keys: Set[str]) -> bool:
    # Convert target POS into a coarse group.
    tg = pos_group(target_pos)
    # Always accept when target group is unknown.
    if tg == "O":
        return True
    # Accept if any candidate POS maps to the same coarse group.
    for p in cand_pos_keys:
        if pos_group(p) == tg:
            return True
    # Otherwise reject.
    return False


# Define a function that builds a candidate pool using buckets and filters by edit distance.
def get_candidate_pool(
    word: str,
    vocab_set: Set[str],
    cand_buckets: Dict[str, Any],
    max_ed: int = 2,
) -> List[Tuple[str, int]]:
    # Normalize the input word for consistent candidate comparison.
    word_norm = _normalize_unicode(word).lower()
    # Compute the normalized word length for bucketing.
    L = len(word_norm)
    # Return empty if vocabulary is missing or word is empty.
    if not vocab_set or L == 0:
        return []
    # Initialize a set to store deduplicated candidate words.
    cand_set: Set[str] = set()
    # Use candidate buckets if available for faster narrowing of the search space.
    if cand_buckets:
        # Extract length buckets mapping length -> words from candidate pack.
        length_buckets: Dict[int, List[str]] = cand_buckets.get("length", {})
        # Extract first-character buckets mapping first char -> words from candidate pack.
        first_char_buckets: Dict[str, List[str]] = cand_buckets.get("first_char", {})
        # Initialize a set to collect candidates in the allowable length neighborhood.
        cand_len_set: Set[str] = set()
        # Iterate length offsets within the maximum edit distance band.
        for d in range(-max_ed, max_ed + 1):
            # Compute candidate word length bucket key.
            ll = L + d
            # Skip non-positive lengths.
            if ll <= 0:
                continue
            # Add all words from that length bucket.
            cand_len_set.update(length_buckets.get(ll, []))
        # Read the first character for first-char bucketing.
        fc = word_norm[0]
        # Collect candidates from the first-char bucket if present.
        cand_fc_set: Set[str] = set(first_char_buckets.get(fc, []))
        # Intersect length and first-char sets when both are present, otherwise use whichever exists.
        if cand_len_set and cand_fc_set:
            cand_set = cand_len_set & cand_fc_set
        else:
            cand_set = cand_len_set or cand_fc_set
        # Restrict candidates to the given vocabulary set.
        cand_set &= vocab_set
    else:
        # Fall back to scanning the entire vocabulary if no buckets are available.
        cand_set = set(vocab_set)
    # Initialize an output list of (candidate, distance) pairs.
    results: List[Tuple[str, int]] = []
    # Compute edit distance for each candidate and keep those within the threshold.
    for cand in cand_set:
        dist = levenshtein(word_norm, cand, max_dist=max_ed)
        if dist <= max_ed:
            results.append((cand, dist))
    # Return the filtered candidate list.
    return results


# Define a function that ranks candidates for a token position using edit distance and frequency scoring.
def rank_candidates_for_position(
    word: str,
    vocab_set: Set[str],
    cand_buckets: Dict[str, Any],
    word_freq: Dict[str, int],
    max_ed: int = 2,
    topk: int = 6,
    target_pos: Optional[str] = None,
    pos_lexicon: Optional[Dict[str, Dict[str, int]]] = None,
) -> List[Dict[str, Any]]:
    # Normalize the word before candidate generation.
    word_norm = _normalize_unicode(word).lower()
    # Build a candidate pool using buckets and edit-distance filtering.
    pool = get_candidate_pool(word_norm, vocab_set, cand_buckets, max_ed=max_ed)
    # Return empty if there are no candidates.
    if not pool:
        return []
    # Initialize a list to hold scored candidate dicts.
    scored: List[Dict[str, Any]] = []
    # Score each candidate using distance and word frequency.
    for cand, dist in pool:
        # Read frequency from the provided dictionary.
        freq = word_freq.get(cand, 1)
        # Combine distance penalty with frequency bonus.
        raw = -dist * 1.0 + math.log(freq + 1.0) * 0.7
        # Apply a POS compatibility bonus/penalty if POS info is available.
        if target_pos and pos_lexicon:
            cand_pos_dict = pos_lexicon.get(cand, {})
            cand_pos_keys = set(cand_pos_dict.keys())
            if cand_pos_keys:
                if pos_compatible(target_pos, cand_pos_keys):
                    raw += 0.6
                else:
                    raw -= 0.6
        # Store the candidate record.
        scored.append({"candidate": cand, "edit_distance": dist, "raw": raw})
    # Sort candidates by raw score descending.
    scored.sort(key=lambda x: x["raw"], reverse=True)
    # Return the top-k candidates.
    return scored[:topk]


# Define a function that converts raw scores into normalized softmax probabilities.
def add_softmax_confidence(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Return the input unchanged if the list is empty.
    if not items:
        return items
    # Extract raw scores from candidate records.
    raws = [it["raw"] for it in items]
    # Compute the maximum raw score for numerical stability.
    mx = max(raws)
    # Compute exponentiated shifted scores.
    exps = [math.exp(r - mx) for r in raws]
    # Compute denominator and guard against division by zero.
    denom = sum(exps) or 1.0
    # Assign normalized probability to each candidate record.
    for it, e in zip(items, exps):
        it["score"] = e / denom
    # Return the updated list with confidence scores.
    return items


# Run non-word detection on tokenized input and return (errors, suggestions_map).
def detect_nonword_errors(
    tokens: List[Dict[str, Any]],
    settings: dict,
    vocab_set: Set[str],
    cand_buckets: Dict[str, Any],
    word_freq: Dict[str, int],
    pos_lexicon: Dict[str, Dict[str, int]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    # Initialize an error list to store detected errors with spans and types.
    errors: List[Dict[str, Any]] = []
    # Initialize a mapping from original token string to its ranked suggestions list.
    suggestions_map: Dict[str, List[Dict[str, Any]]] = {}
    # Iterate through token records with their indices for later context lookup.
    for idx, t in enumerate(tokens):
        tok = t["token"]
        start = t["start"]
        end = t["end"]
        # Skip tokens that contain digits or are not pure letters.
        if should_skip_token(tok):
            continue
        # Detect casing pattern to preserve original casing in suggestions.
        casing = detect_casing(tok)
        # Normalize the token and lowercase it for dictionary lookup.
        lower = _normalize_unicode(tok).lower()
        # Detect non-word errors by checking if the normalized token is absent from the vocabulary set.
        if lower not in vocab_set:
            target_pos = tokens[idx].get("pos")
            ranked = rank_candidates_for_position(
                word=tok,
                vocab_set=vocab_set,
                cand_buckets=cand_buckets,
                word_freq=word_freq,
                max_ed=2,
                topk=6,
                target_pos=target_pos,
                pos_lexicon=pos_lexicon,
            )
            # Skip flagging this token as an error when no reasonable candidates are available.
            if not ranked:
                continue
            # Optionally compute softmax confidence scores for candidates.
            if settings.get("show_confidence", False):
                ranked = add_softmax_confidence(ranked)
            # Remove the internal raw score field before sending results to the frontend.
            for it in ranked:
                it.pop("raw", None)
            # Apply original casing pattern to each candidate so replacements look natural.
            ranked = apply_casing_to_list(ranked, casing)
            # Store ranked candidates under the original surface token.
            suggestions_map[tok] = ranked
            # Add this token to the error list as a non-word error with its character span.
            errors.append({"word": tok, "start": start, "end": end, "type": "non-word"})
    # Return non-word errors and suggestions.
    return errors, suggestions_map
