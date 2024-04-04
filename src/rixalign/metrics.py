from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def calculate_bleu(text1, text2):
    """
    Calculate BLEU score between two texts.
    """

    if text1 is None or text2 is None:
        return None
    else:
        chencherry = SmoothingFunction()
        return sentence_bleu(
            references=[text1.split()],
            hypothesis=text2.split(),
            smoothing_function=chencherry.method4,
        )
