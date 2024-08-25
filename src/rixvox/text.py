import re
import string
import unicodedata
from collections import OrderedDict

import numpy as np
import pandas as pd
from num2words import num2words

from rixvox.parlaspeech.strings_sv import abbreviations, ocr_corrections, symbols


def format_symbols_abbreviations():
    """
    Formats abbreviations into dicts that include the pattern (abbreviation) and replacement (expansion).
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

    for symbol, expansion in symbols.items():
        abbreviation_patterns.append(
            {
                "pattern": re.escape(symbol),
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
    patterns.extend(format_symbols_abbreviations())

    patterns.extend(
        [
            # Capture pattern groups of type '(digits) kap. (digits) §'. For example "4 kap. 7 §".
            # Replace the numbers with ordinals: "fjärde kapitlet sjunde paragrafen"
            {
                "pattern": r"(\d+) kap\. (\d+) \§",
                "replacement": lambda m: f"{num2words(int(m.group(1)),lang='sv',ordinal=True)} kapitlet {num2words(int(m.group(2)),lang='sv', ordinal=True)} paragrafen",
                "transformation_type": "substitution",
            },
            {
                "pattern": r"(\d+)[\. ](\d+)",
                "replacement": lambda m: f"{num2words(m.group(1) + m.group(2), lang='sv')}",
                "transformation_type": "substitution",
            },
            # Replace : or / between digits with whitespace and num2words the digits
            {
                "pattern": r"(\d+):(\d+)",
                "replacement": lambda m: f"{num2words(int(m.group(1)), lang='sv')} {num2words(int(m.group(2)), lang='sv')}",
                "transformation_type": "substitution",
            },
            # Replace - between digits with " till " and num2words the digits
            {
                "pattern": r"(\d+)-(\d+)",
                "replacement": lambda m: f"{num2words(int(m.group(1)), lang='sv')} till {num2words(int(m.group(2)), lang='sv')}",
                "transformation_type": "substitution",
            },
            # Replace , between digits with " komma " and num2words the digits
            {
                "pattern": r"(\d+),(\d+)",
                "replacement": lambda m: f"{num2words(int(m.group(1)), lang='sv')} komma {num2words(int(m.group(2)), lang='sv')}",
                "transformation_type": "substitution",
            },
            {
                "pattern": r"(\d+)[\.\,\:\-\/](\d+)",
                "replacement": lambda m: f"{m.group(1)} {m.group(2)}",
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
            # Remove punctuation between whitespace
            {"pattern": r"\s[^\w\s]\s", "replacement": " ", "transformation_type": "substitution"},
            # Remove punctuation
            {"pattern": r"[^\w\s]", "replacement": "", "transformation_type": "deletion"},
            # Remove multiple spaces (more than one) with a single space
            {"pattern": r"\s{2,}", "replacement": " ", "transformation_type": "substitution"},
            # Strip leading and trailing whitespace
            {"pattern": r"^\s+|\s+$", "replacement": "", "transformation_type": "deletion"},
            # Replace digits with words
            {
                "pattern": r"(\d+)",
                "replacement": lambda m: num2words(int(m.group(1)), lang="sv"),
                "transformation_type": "substitution",
            },
            # Tokenize the rest of the text into words
            {
                "pattern": r"\w+",
                "replacement": lambda m: m.group(),
                "transformation_type": "substitution",  # Not really a substitution, but we need to record the transformation
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
        "original_token": original_span,
        "normalized_token": (
            replacement.lower()
            if transformation_type == "substitution" and replacement != " "
            else None
        ),
        "start_time": None,
        "end_time": None,
        "index": None,
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

    # Sort transformations by their original start position to ensure correct application order
    mapping.sort(key=lambda x: x["original_start"])

    for i, transformation in enumerate(mapping):
        original_start = transformation["original_start"]
        original_end = transformation["original_end"]

        if modified[original_start:original_end].any():
            # Skip this transformation if it overlaps with a previous transformation
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

        transformation["index"] = i

    return normalized_text


def normalize_text_with_mapping(text, user_patterns=None, combine_regexes=False):
    """
    Normalize speech text transcript while keeping track of transformations.

    Args:
        text (str): The original text to normalize.
        user_patterns (list of dicts, optional): User-supplied regex patterns, replacements, and type of transformation.

    Returns:
        tuple: Normalized text and list of mappings from original to new text positions.
    """
    mapping = []

    # Correct some OCR-errors before normalization
    for key, value in ocr_corrections.items():
        text = re.sub(key, value, text)

    # Collect all regex patterns for substitutions and deletions
    transformations = collect_regex_patterns(user_patterns)

    # Track already matched character spans using a boolean mask
    modified_chars = np.zeros(len(text), dtype=bool)

    # Record transformations for each pattern match
    for pattern_dict in transformations:
        pattern = pattern_dict["pattern"]
        transformation_type = pattern_dict["transformation_type"]

        for match in re.finditer(pattern, text.lower()):
            start, end = match.span()
            if modified_chars[start:end].any():
                # Skip this match if it overlaps with a previous match
                continue
            else:
                # Mark the characters as "to be modified"
                modified_chars[start:end] = True

            # If pattern_dict["replacement"] is a lambda function, call it to get the replacement string
            # Otherwise, use the replacement string
            if callable(pattern_dict["replacement"]):
                replacement = pattern_dict["replacement"](match)
            else:
                replacement = pattern_dict["replacement"]

            record_transformation(mapping, text, start, end, transformation_type, replacement)

    text = unicodedata.normalize("NFKC", text)

    # Apply the recorded transformations to the text
    normalized_text = apply_transformations(text, mapping)

    return normalized_text, mapping


def get_normalized_tokens(mapping, casing="lower"):
    normalized_mapping = OrderedDict()
    normalized_tokens = []
    for i, record in enumerate(mapping):
        if record["transformation_type"] == "substitution" and record["replacement"] != " ":
            normalized_token = (
                record["normalized_token"]
                if casing == "lower"
                else record["normalized_token"].upper()  # Swedish wav2vec2 has uppercase tokens
            )
            normalized_mapping[i] = {
                "token": normalized_token,
                "start_time": record["start_time"],  # Empty for now
                "end_time": record["end_time"],  # Empty for now
            }
            normalized_tokens.append(normalized_token)
    return normalized_mapping, normalized_tokens


# Assume  timestamps have been added to normalized_mapping
def add_timestamps_to_mapping(mapping, normalized_mapping):
    for i, record in enumerate(mapping):
        normalized_record = normalized_mapping.get(i)
        if normalized_record:
            record["start_time"] = normalized_record["start_time"]
            record["end_time"] = normalized_record["end_time"]
    return mapping


def expand_abbreviations(text):
    """
    Replace abbreviations with their full form in the text.
    """

    # Normalize variations of abbreviations
    for key, value in ocr_corrections.items():
        text = text.replace(key, value)

    # Replace abbreviations with their full form
    for key, value in abbreviations.items():
        text = text.replace(key, value)

    # Replace symbols with their full form
    for key, value in symbols.items():
        text = text.replace(key, value)

    return text


def normalize_text(text):
    """
    Normalize speech text transcript by removing punctuation, converting numbers to words,
    replacing hyphens joining words with whitespace, and lowercasing the text. The purpose
    is to make a normalized source text similar to wav2vec2 output for better string matching.

    Args:
        text (str): The text to normalize.
    Returns:
        str: The normalized text.
    """

    text = text.lower()
    text = expand_abbreviations(text)  # Replace abbreviations with their full form

    # Capture pattern groups of type '(digits) kap. (digits) §'. For example "4 kap. 7 §".
    # Replace the numbers with ordinals: "fjärde kapitlet sjunde paragrafen"
    text = re.sub(
        r"(\d+) kap\. (\d+) \§",
        lambda m: f"{num2words(int(m.group(1)),lang='sv',ordinal=True)} kapitlet {num2words(int(m.group(2)),lang='sv', ordinal=True)} paragrafen",
        text,
    )
    # Replace whitespace between numbers with no whitespace: 1 000 000 -> 1000000
    text = re.sub(r"(\d+) (\d+)", r"\1\2", text)

    # Replace punctuations with whitespace if there is a number before and after it
    # 3,14 -> 3 14, 4-5 -> 4 5, 4:5 -> 4 5, 4/5 -> 4 5 (closer to the actually pronounced number)
    text = re.sub(r"(\d+)[\.\,\:\-\/](\d+)", r"\1 \2", text)

    # Replace § with 'paragrafen' if there is a number before it
    text = re.sub(r"(\d+) §", r"\1 paragrafen", text)
    # Replace § with "paragraf" if there is a number after it
    text = re.sub(r"§ (\d+)", r"paragraf \1", text)
    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    # Replace all non-alphanumeric characters with space
    text = re.sub(r"[\W]+", " ", text)
    # remove \r and \n and multiple spaces
    text = re.sub(r"\s+", " ", text)
    # # Convert numbers to words
    text = re.sub(r"\d+", lambda m: num2words(int(m.group(0)), lang="sv"), text)
    # Strip leading and trailing whitespace
    text = text.strip()

    return text


def normalize_text_df(df, column_in, column_out):
    """
    Normalize speech text transcript by removing punctuation, converting numbers to words,
    replacing hyphens joining words with whitespace, and lowercasing the text.

    Args:
        df (pd.DataFrame): A pandas dataframe that contains text column anftext with speeches.
        column_in (str): The name of the text column to normalize.
        column_out (str): The name of the normalized text column.
    Returns:
        pd.DataFrame: A pandas dataframe with normalized text column `column_out`.
    """
    df[column_out] = df[column_out].str.lower()
    df[column_out] = df[column_out].apply(
        expand_abbreviations
    )  # Replace abbreviations with their full form

    # Capture pattern groups of type '(digits) kap. (digits) §'. For example "4 kap. 7 §".
    # Replace the numbers with ordinals: "fjärde kapitlet sjunde paragrafen"
    df[column_out] = df[column_in].apply(
        lambda x: (
            None
            if x is None
            else re.sub(
                r"(\d+) kap\. (\d+) \§",
                lambda m: f"{num2words(int(m.group(1)),lang='sv',ordinal=True)} kapitlet {num2words(int(m.group(2)),lang='sv', ordinal=True)} paragrafen",
                x,
            )
        )
    )
    # Replace § with 'paragrafen' if there is a number before it
    df[column_out] = df[column_out].apply(
        lambda x: None if x is None else re.sub(r"(\d+) §", r"\1 paragrafen", x)
    )
    # Replace § with "paragraf" if there is a number after it
    df[column_out] = df[column_out].apply(
        lambda x: None if x is None else re.sub(r"§ (\d+)", r"paragraf \1", x)
    )
    df[column_out] = df[column_in].apply(
        lambda x: None if x is None else x.translate(str.maketrans("", "", string.punctuation))
    )
    # df[column_out] = df[column_out].str.replace("\xa0", " ")
    df[column_out] = df[column_out].str.normalize("NFKC")  # Normalize unicode characters
    # Remove hyphen between words
    df[column_out] = df[column_out].str.replace("(?<=\w)-(?=\w)", " ", regex=True)
    # Remove \r and \n
    df[column_out] = df[column_out].str.replace(r"[\r\n]+", " ", regex=True)
    # Remove multiple spaces and replace with single space
    df[column_out] = df[column_out].str.replace(" +", " ", regex=True)
    # Remove whitespace between numbers
    df[column_out] = df[column_out].str.replace("(?<=\d) (?=\d)", "", regex=True)
    # # Convert numbers to words
    df[column_out] = df[column_out].apply(
        lambda x: (
            None
            if x is None
            else re.sub(r"\d+", lambda m: num2words(int(m.group(0)), lang="sv"), x)
        )
    )
    # Strip leading and trailing whitespace
    df[column_out] = df[column_out].str.strip()

    return df


def check_abbreviations(text):
    """
    Find all abbreviations with 1 or more dots
    that are not followed by a space
    """
    abbreviations = re.findall(r"(?:[a-zA-ZåäöÅÄÖ]+[\.\:]){1,}(?!\s)[a-zA-ZåäöÅÄÖ\.]", text)
    return abbreviations


def preprocess_audio_metadata(speech_metadata):
    """
    Preprocess the speech_metadata dict to a pandas dataframe on modern Swedish parliament speeches
    retrieved from the Riksdagen API.

    Args:
        speech_metadata (dict): Nested metadata fields with transcribed texts, media file
            URLs and more from the Riksdagen API.

    Returns:
        pd.DataFrame: A pandas dataframe with the relevant metadata fields.
    """

    df = pd.json_normalize(
        speech_metadata,
        record_path=["dokumentstatus", "debatt", "anforande"],
        meta=[["dokumentstatus", "webbmedia"]],
    )
    df_media = df["dokumentstatus.webbmedia"].apply(pd.Series)["media"].apply(pd.Series)
    df_dokument = pd.json_normalize(speech_metadata["dokumentstatus"]["dokument"])
    df = pd.concat([df, df_media], axis=1)
    df["debatt_namn"] = df_dokument["debattnamn"]

    df = df[
        [
            "dok_id",
            "parti",
            "startpos",
            "anf_sekunder",
            "debatt_titel",
            "anf_text",
            "debatt_namn",
            "anf_datum",
            "url",
            "debateurl",
            "debatt_id",
            "audiofileurl",
            "downloadurl",
            "debatt_typ",
            "anf_nummer",
            "anf_nummer2",
            "intressent_id",
            "intressent_id2",
            "anf_id",
        ]
    ]

    df = df.rename(columns={"startpos": "start", "anf_sekunder": "duration"})
    df = preprocess_text(df, is_audio_metadata=True)

    return df


def preprocess_text(df, textcol="anf_text", is_audio_metadata=False):
    """
    Preprocess the text field on modern Swedish parliament speeches retrieved from the
    Riksdagen API.

    Args:
        df (pd.DataFrame): A pandas dataframe that contains text column with speeches.
        textcol (str): The name of the text column.

    Returns:
        pd.DataFrame: A pandas dataframe with preprocessed text column.
    """

    # Remove all text within <p> tags that contain "STYLEREF".
    # These are headers mistakenly included in the text as paragraphs.
    df[textcol] = df[textcol].str.replace(r"(<p> STYLEREF.*?</p>)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(<p>Gransknings- STYLEREF.*?</p>)", "", regex=True)
    df[textcol] = df[textcol].str.replace(r"(<p><em></em><em> STYLEREF.*?</p>)", "", regex=True)

    # Some extra headers that don't contain "STYLEREF", but are still in <p> tags.
    # We grab the headers from the header column and remove "<p>{header}</p>" from the text column.
    # data/headers.csv is created in scripts/preprocess_speeches_metadata.py.
    headers = pd.read_csv("data/headers.csv")["avsnittsrubrik"].tolist()

    for header in headers:
        remove_header_p = f"<p>{header}</p>"
        df[textcol] = df[textcol].str.replace(remove_header_p, "", regex=False)

    df[textcol] = df[textcol].str.replace(r"<.*?>", " ", regex=True)  # Remove HTML tags
    # Remove text within parentheses, e.g. (applåder)
    df[textcol] = df[textcol].str.replace(r"\(.*?\)", "", regex=True)

    # Speaker of the house or other text not part of actual speech.
    # Found at the end of a transcript.
    df[textcol] = df[textcol].str.replace(
        r"(Interpellationsdebatten var [h|d]ärmed avslutad.*)", "", regex=True
    )
    df[textcol] = df[textcol].str.replace(
        r"(Partiledardebatten var [h|d]ärmed avslutad.*)", "", regex=True
    )
    df[textcol] = df[textcol].str.replace(
        r"(Frågestunden var [h|d]ärmed avslutad.*)", "", regex=True
    )
    df[textcol] = df[textcol].str.replace(
        r"(Överläggningen var [h|d]ärmed avslutad.*)", "", regex=True
    )
    df[textcol] = df[textcol].str.replace(
        r"(Den särskilda debatten var [h|d]ärmed avslutad.*)", "", regex=True
    )
    df[textcol] = df[textcol].str.replace(
        r"(Statsministerns frågestund var [h|d]ärmed avslutad.*)", "", regex=True
    )
    df[textcol] = df[textcol].str.replace(
        r"(Återrapporteringen var [h|d]ärmed avslutad.*)", "", regex=True
    )
    df[textcol] = df[textcol].str.replace(
        r"(Den muntliga frågestunden var [h|d]ärmed avslutad.*)", "", regex=True
    )
    df[textcol] = df[textcol].str.replace(
        r"(Den utrikespolitiska debatten var [h|d]ärmed avslutad.*)", "", regex=True
    )
    df[textcol] = df[textcol].str.replace(
        r"(Den allmänpolitiska debatten var härmed avslutad.*)", "", regex=True
    )
    df[textcol] = df[textcol].str.replace(
        r"(Den aktuella debatten var härmed avslutad.*)", "", regex=True
    )
    df[textcol] = df[textcol].str.replace(r"(Informationen var härmed avslutad.*)", "", regex=True)
    df[textcol] = df[textcol].str.replace(
        r"(Den EU-politiska (partiledar)?debatten var härmed avslutad.*)", "", regex=True
    )
    df[textcol] = df[textcol].str.replace(
        r"(Debatten med anledning av (vår|budget)propositionens avlämnande var härmed avslutad.*)",
        "",
        regex=True,
    )
    df[textcol] = df[textcol].str.replace(r"(I detta anförande instämde.*)", "", regex=True)

    df[textcol] = df[textcol].str.strip()

    # Normalize text
    df[textcol] = df[textcol].str.normalize("NFKC")  # Normalize unicode characters
    # Remove multiple spaces
    df[textcol] = df[textcol].str.replace(r"\s+", " ", regex=True)
    # Replace &amp; with &
    df[textcol] = df[textcol].str.replace(r"&amp;", "&", regex=True)

    return df
