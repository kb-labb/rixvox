import re
import string
import unicodedata

import pandas as pd
from num2words import num2words

from rixvox.parlaspeech.strings_sv import abbreviations, ocr_corrections, symbols


def tokenize_segment(text):
    pass


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
    replacing hyphens joining words with whitespace, and lowercasing the text.

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

    # TODO: Replace punctuations with whitespace if there is a number before and after it
    # 3,14 -> 3 14, 4-5 -> 4 5, 4:5 -> 4 5, 4/5 -> 4 5 (closer to the actually pronounced number)
    text = re.sub(r"(\d+)[\.\,\:\-\/](\d+)", r"\1 \2", text)

    # Replace § with 'paragrafen' if there is a number before it
    text = re.sub(r"(\d+) §", r"\1 paragrafen", text)
    # Replace § with "paragraf" if there is a number after it
    text = re.sub(r"§ (\d+)", r"paragraf \1", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)
    # remove \r and \n
    text = re.sub(r"\s+", " ", text)
    ## Remove whitespace between numbers
    # text = re.sub(r"(?<=\d) (?=\d)", "", text)
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

    df = pd.DataFrame(speech_metadata["videodata"])
    df = df.explode("speakers").reset_index(drop=True)
    df_files = pd.json_normalize(df["streams"], ["files"])
    df_speakers = pd.json_normalize(df["speakers"])
    df = df.drop(columns=["streams", "speakers"])
    df = pd.concat([df, df_files], axis=1)
    df = pd.concat([df, df_speakers], axis=1)

    df = df[
        [
            "dokid",
            "party",
            "start",
            "duration",
            "debateseconds",
            "title",
            "text",
            "debatename",
            "debatedate",
            "url",
            "debateurl",
            "id",
            "subid",
            "audiofileurl",
            "downloadfileurl",
            "debatetype",
            "number",
            "anftext",
        ]
    ]

    df = preprocess_text(df, is_audio_metadata=True)

    return df


def preprocess_text(df, textcol="anftext", is_audio_metadata=False):
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
    df[textcol] = df[textcol].str.replace(r"(\s){2,}", " ", regex=True)
    # Replace &amp; with &
    df[textcol] = df[textcol].str.replace(r"&amp;", "&", regex=True)

    return df
