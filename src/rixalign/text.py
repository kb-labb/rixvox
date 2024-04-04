import re
import string
import unicodedata

from num2words import num2words

from rixalign.parlaspeech.strings_sv import abbreviations, ocr_corrections, symbols


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
    # TODO: Replace punctuations with whitespace if there is a number before and after it
    text = re.sub(r"(\d+)[\.\,\:\-\/](\d+)", r"\1 \2", text)

    # Replace § with 'paragrafen' if there is a number before it
    text = re.sub(r"(\d+) §", r"\1 paragrafen", text)
    # Replace § with "paragraf" if there is a number after it
    text = re.sub(r"§ (\d+)", r"paragraf \1", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)
    # Remove hyphen between words with regex
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    # remove \r and \n
    text = re.sub(r"[\r\n]+", " ", text)
    # Remove multiple spaces and replace with single space
    text = re.sub(" +", " ", text)
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
