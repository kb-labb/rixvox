import re
import unicodedata

import numpy as np
from num2words import num2words

abbreviations = {"bl.a.": "bland annat", "kungl. maj:t": "kunglig majestät"}


def format_abbreviations():
    """
    Formats abbreviations for text normalization and substitution.
    """
    abbreviation_patterns = []
    for abbreviation, expansion in abbreviations.items():
        abbreviation_patterns.append(
            {
                "pattern": re.escape(abbreviation),
                "replacement": expansion,
                "transformation_type": "substitution",
            }
        )
    return abbreviation_patterns


def collect_regex_patterns(user_patterns=None):
    """
    Collects regex patterns for text normalization and substitution.

    Args:
        user_patterns (list of dict): User-supplied regex patterns with keys "pattern", "replacement", and "transformation_type".

    Returns:
        list of dict: Collected regex patterns with default and user-supplied patterns.
    """

    patterns = []

    # Include abbreviations
    patterns.extend(format_abbreviations())

    patterns.extend(
        [
            # Capture pattern groups of type '(digits) kap. (digits) §'. For example "4 kap. 7 §".
            # Replace the numbers with ordinals: "fjärde kapitlet sjunde paragrafen"
            {
                "pattern": r"(\d+) kap\. (\d+) \§",
                "replacement": lambda m: f"{num2words(int(m.group(1)),lang='sv',ordinal=True)} kapitlet {num2words(int(m.group(2)),lang='sv', ordinal=True)} paragrafen",
                "transformation_type": "substitution",
            },
            # Replace punctuations with whitespace between numbers
            {
                "pattern": r"(\d+)[\.\,\:\-\/](\d+)",
                "replacement": r"\1 \2",
                "transformation_type": "substitution",
            },
            # Replace whitespace between numbers
            {
                "pattern": r"(\d+) (\d+)",
                "replacement": r"\1\2",
                "transformation_type": "substitution",
            },
            # Replace § with 'paragrafen' if preceded by a number
            {
                "pattern": r"(?<=\d )§",
                "replacement": r"paragrafen",
                "transformation_type": "substitution",
            },
            # Replace § with 'paragraf' if succeeded by a number
            {
                "pattern": r"§(?= \d)",
                "replacement": r"paragraf",
                "transformation_type": "substitution",
            },
            # Remove punctuation
            {"pattern": r"[^\w\s]", "replacement": "", "transformation_type": "deletion"},
            # Remove multiple spaces (more than one) with a single space
            {"pattern": r"\s{2,}", "replacement": " ", "transformation_type": "substitution"},
            # Strip leading and trailing whitespace
            {"pattern": r"^\s+|\s+$", "replacement": "", "transformation_type": "deletion"},
            # Replace digits with words
            {
                "pattern": r"(\d+)",
                "replacement": lambda m: num2words(int(m.group(0)), lang="sv"),
                "transformation_type": "substitution",
            },
        ]
    )

    # Include user-supplied patterns
    if user_patterns:
        patterns.extend(user_patterns)

    return patterns


def record_transformation(mapping, original_text, start, end, transformation_type, replacement):
    """
    Records a transformation in the mapping with additional context for debugging.

    Args:
        mapping (list of dicts): The list that stores transformation records.
        original_text (str): The original text being normalized.
        start (int): The start index of the original text span.
        end (int): The end index of the original text span.
        transformation_type (str): The type of transformation ('substitution', 'deletion', 'insertion').
        replacement (str): The replacement text (empty string for deletions).
    """
    original_span = original_text[start:end] if start is not None and end is not None else ""
    transformation_record = {
        "original_start": start,
        "original_end": end,
        "transformation_type": transformation_type,
        "replacement": replacement,
        "normalized_start": None,  # To be filled in during the apply_transformations step
        "normalized_end": None,  # To be filled in during the apply_transformations step
        "original_span": original_span,
    }
    mapping.append(transformation_record)


def apply_transformations(text, mapping):
    """
    Applies recorded transformations to the text and updates the mapping with normalized positions.

    Args:
        text (str): The original text.
        mapping (list of dicts): The list of transformations.

    Returns:
        str: The transformed (normalized) text.
    """

    text_length = len(text)
    modified = np.zeros(text_length, dtype=bool)  # Track modified characters using a boolean mask

    offset = 0
    normalized_text = text

    # Sort transformations by their original start position
    mapping.sort(key=lambda x: x["original_start"])

    for transformation in mapping:
        original_start = transformation["original_start"]
        original_end = transformation["original_end"]

        if modified[original_start:original_end].any():
            continue
        else:
            # Mark the characters as modified
            modified[original_start:original_end] = True

        replacement = transformation["replacement"]

        # Calculate the adjusted start and end positions based on the current offset
        adjusted_start = original_start + offset
        adjusted_end = original_end + offset

        # Apply the transformation
        normalized_text = (
            normalized_text[:adjusted_start] + replacement + normalized_text[adjusted_end:]
        )

        # Update the normalized spans in the transformation record
        transformation["normalized_start"] = adjusted_start
        transformation["normalized_end"] = adjusted_start + len(replacement)

        # Update the offset for the next transformation
        offset += len(replacement) - (original_end - original_start)

    return normalized_text


def normalize_text_with_mapping(text, user_substitutions=None, user_deletions=None):
    """
    Normalize speech text transcript while keeping track of transformations.

    Args:
        text (str): The original text to normalize.
        user_substitutions (list of dicts, optional): User-supplied regex patterns, replacements, and types for substitutions.
        user_deletions (list of dicts, optional): User-supplied regex patterns and types for deletions.

    Returns:
        tuple: Normalized text and list of mappings from original to new text positions.
    """
    mapping = []
    text = text.lower()

    # Collect all regex patterns for substitutions and deletions
    transformations = collect_regex_patterns(user_substitutions)

    # Record transformations for each pattern match
    for pattern_dict in transformations:
        pattern = pattern_dict["pattern"]
        transformation_type = pattern_dict["transformation_type"]

        for match in re.finditer(pattern, text):
            start, end = match.span()

            # If pattern_dict["replacement"] is a lambda function, call it to get the replacement
            if callable(pattern_dict["replacement"]):
                pattern_dict["replacement"] = pattern_dict["replacement"](match)

            record_transformation(
                mapping, text, start, end, transformation_type, pattern_dict["replacement"]
            )

    text = unicodedata.normalize("NFKD", text)

    # Apply the recorded transformations to the text
    normalized_text = apply_transformations(text, mapping)

    return normalized_text, mapping


from pprint import pprint

pprint(
    normalize_text_with_mapping(
        "Vi har bl.a. sett kungl. maj:t vinka till oss - det gjorde han bra. Kungl. maj:t vad glad när han fick 20243 kronor. Den finns i 4 kap. 7 § i lagen.",
    )
)
