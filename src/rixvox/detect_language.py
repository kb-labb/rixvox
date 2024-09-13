from typing import Collection, Dict, List, Optional

import torch
import transformers

WHISPER_LANGUAGE_IDS = {
    "en": 50259,
    "zh": 50260,
    "de": 50261,
    "es": 50262,
    "ru": 50263,
    "ko": 50264,
    "fr": 50265,
    "ja": 50266,
    "pt": 50267,
    "tr": 50268,
    "pl": 50269,
    "ca": 50270,
    "nl": 50271,
    "ar": 50272,
    "sv": 50273,
    "it": 50274,
    "id": 50275,
    "hi": 50276,
    "fi": 50277,
    "vi": 50278,
    "he": 50279,
    "uk": 50280,
    "el": 50281,
    "ms": 50282,
    "cs": 50283,
    "ro": 50284,
    "da": 50285,
    "hu": 50286,
    "ta": 50287,
    "no": 50288,
    "th": 50289,
    "ur": 50290,
    "hr": 50291,
    "bg": 50292,
    "lt": 50293,
    "la": 50294,
    "mi": 50295,
    "ml": 50296,
    "cy": 50297,
    "sk": 50298,
    "te": 50299,
    "fa": 50300,
    "lv": 50301,
    "bn": 50302,
    "sr": 50303,
    "az": 50304,
    "sl": 50305,
    "kn": 50306,
    "et": 50307,
    "mk": 50308,
    "br": 50309,
    "eu": 50310,
    "is": 50311,
    "hy": 50312,
    "ne": 50313,
    "mn": 50314,
    "bs": 50315,
    "kk": 50316,
    "sq": 50317,
    "sw": 50318,
    "gl": 50319,
    "mr": 50320,
    "pa": 50321,
    "si": 50322,
    "km": 50323,
    "sn": 50324,
    "yo": 50325,
    "so": 50326,
    "af": 50327,
    "oc": 50328,
    "ka": 50329,
    "be": 50330,
    "tg": 50331,
    "sd": 50332,
    "gu": 50333,
    "am": 50334,
    "yi": 50335,
    "lo": 50336,
    "uz": 50337,
    "fo": 50338,
    "ht": 50339,
    "ps": 50340,
    "tk": 50341,
    "nn": 50342,
    "mt": 50343,
    "sa": 50344,
    "lb": 50345,
    "my": 50346,
    "bo": 50347,
    "tl": 50348,
    "mg": 50349,
    "as": 50350,
    "tt": 50351,
    "haw": 50352,
    "ln": 50353,
    "ha": 50354,
    "ba": 50355,
    "jw": 50356,
    "su": 50357,
}

WHISPER_LANGUAGES = [
    "en",
    "zh",
    "de",
    "es",
    "ru",
    "ko",
    "fr",
    "ja",
    "pt",
    "tr",
    "pl",
    "ca",
    "nl",
    "ar",
    "sv",
    "it",
    "id",
    "hi",
    "fi",
    "vi",
    "he",
    "uk",
    "el",
    "ms",
    "cs",
    "ro",
    "da",
    "hu",
    "ta",
    "no",
    "th",
    "ur",
    "hr",
    "bg",
    "lt",
    "la",
    "mi",
    "ml",
    "cy",
    "sk",
    "te",
    "fa",
    "lv",
    "bn",
    "sr",
    "az",
    "sl",
    "kn",
    "et",
    "mk",
    "br",
    "eu",
    "is",
    "hy",
    "ne",
    "mn",
    "bs",
    "kk",
    "sq",
    "sw",
    "gl",
    "mr",
    "pa",
    "si",
    "km",
    "sn",
    "yo",
    "so",
    "af",
    "oc",
    "ka",
    "be",
    "tg",
    "sd",
    "gu",
    "am",
    "yi",
    "lo",
    "uz",
    "fo",
    "ht",
    "ps",
    "tk",
    "nn",
    "mt",
    "sa",
    "lb",
    "my",
    "bo",
    "tl",
    "mg",
    "as",
    "tt",
    "haw",
    "ln",
    "ha",
    "ba",
    "jw",
    "su",
]


def detect_language(
    model: transformers.WhisperForConditionalGeneration,
    tokenizer: transformers.WhisperTokenizer,
    input_features,
    possible_languages: Optional[Collection[str]] = None,
) -> List[Dict[str, float]]:
    # hacky, but all language tokens and only language tokens are 6 characters long
    # except for haw
    language_tokens = [
        t for t in tokenizer.additional_special_tokens if len(t) == 6 or len(t) == 7
    ]
    if possible_languages is not None:
        language_tokens = [t for t in language_tokens if t[2:-2] in possible_languages]
        if len(language_tokens) < len(possible_languages):
            raise RuntimeError(
                f"Some languages in {possible_languages} did not have associated language tokens"
            )

    language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)

    # 50258 is the token for transcribing
    logits = model(
        input_features,
        decoder_input_ids=torch.tensor([[50258] for _ in range(input_features.shape[0])]).to(
            "cuda:0",
            # max_length=1,
        ),
    ).logits
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[language_token_ids] = False
    logits[:, :, mask] = -float("inf")

    output_probs = logits.softmax(dim=-1).cpu()
    return [
        {
            lang.strip("<|>"): output_probs[input_idx, 0, token_id].item()
            for token_id, lang in zip(language_token_ids, language_tokens)
        }
        for input_idx in range(logits.shape[0])
    ]


def get_language_probs(scores):
    """
    Note: Language detection may not be done seperately for each observation
    in HuggingFace when data is batched.
    """
    keys, ids = zip(*WHISPER_LANGUAGE_IDS.items())
    probs = torch.softmax(scores, dim=-1)
    top_language = torch.argmax(probs, dim=-1, keepdim=True)
    top_language_probs = probs.gather(1, top_language)
    all_languages = probs[:, ids]

    top_language = top_language.to("cpu").squeeze(-1).tolist()
    top_language_probs = top_language_probs.to("cpu").squeeze(-1).tolist()
    all_languages = all_languages.to("cpu").tolist()

    all_languages = [{lang: prob for lang, prob in zip(keys, langs)} for langs in all_languages]

    return top_language, top_language_probs, all_languages
