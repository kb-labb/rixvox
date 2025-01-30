from jiwer import wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rapidfuzz import fuzz


def calculate_bleu(reference, hypothesis):
    """
    Calculate BLEU score between two texts.
    """

    if reference is None or hypothesis is None:
        return None
    else:
        chencherry = SmoothingFunction()
        return sentence_bleu(
            references=[reference.split()],
            hypothesis=hypothesis.split(),
            smoothing_function=chencherry.method4,
        )


def calculate_rouge(reference, hypothesis):
    """
    Calculate weighted ROUGE-N score between two texts.
    """

    if reference is None or hypothesis is None:
        return None
    else:
        chencherry = SmoothingFunction()
        return sentence_bleu(
            references=[hypothesis.split()],
            hypothesis=reference.split(),
            smoothing_function=chencherry.method4,
            weights=(0, 0.25, 0.5, 0.25),
        )


def calculate_wer(text1, text2):
    """
    Calculate Word Error Rate between two texts.
    """
    try:
        return wer(text1, text2)
    except:
        return None


def first_word_fuzzy_score(text1, text2):
    """
    Calculate fuzzy ratio between the first word of two texts.
    """
    if text1 is None or text2 is None:
        return None
    elif text1 == "" or text2 == "":
        return 0
    else:
        return fuzz.ratio(text1.split()[0], text2.split()[0])


def last_word_fuzzy_score(text1, text2):
    """
    Calculate fuzzy ratio between the last word of two texts.
    """
    if text1 is None or text2 is None:
        return None
    elif text1 == "" or text2 == "":
        return 0
    else:
        return fuzz.ratio(text1.split()[-1], text2.split()[-1])
