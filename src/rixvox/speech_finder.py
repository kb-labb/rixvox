import multiprocessing as mp

import numpy as np
from nltk import ngrams
from nltk.tokenize import TreebankWordTokenizer
from rapidfuzz import fuzz
from tqdm import tqdm


def get_ngrams_array(text, n):
    """
    Returns an array of ngrams for a given text

    args:
        text: str
        n: int
    """
    n_grams = ngrams(text.split(), n)
    n_grams = np.array(list(n_grams))
    return n_grams


def get_ngram_index_match(haystack_ngrams, ngram):
    """
    Get ngrams for haystack and check if argument ngram is in text_inference.
    Returns boolean array with True value(s) where the given ngram match is found

    Args:
        haystack_ngrams (np.array): Ngrams of text transcription of speech
            audio file by wav2vec2. Via function get_ngrams_array().
        ngram (list | tuple | np.array): Ngram to check if it is in haystack

    Returns:
        np.array: Boolean array with True value(s) where the given ngram match is found
    """

    ngram_bool_matrix = haystack_ngrams == tuple(ngram)

    if ngram_bool_matrix is bool:
        return np.array(ngram_bool_matrix)
    else:
        return ngram_bool_matrix.all(axis=1)


def get_weighted_ngram_score(needle, haystack, n):
    """
    Get weighted ngram scores for haystack and needle using
    ngrams of several different sizes (from 1 to n).

    Args:
        needle (str): A "needle" substring that is possibly contained in the "haystack".
        haystack (str): The larger text string that possibly contains the "needle".
        n (int): Maximum ngram size. Will use 1 to n ngram size.

    Returns:
        np.array: Array with different ngram size occurences weighted together.
    """

    ngrams_bool_list = []
    for i in range(1, n):
        needle_ngrams = get_ngrams_array(needle, n=i)
        haystack_ngrams = get_ngrams_array(haystack, i)
        array_list = []
        for ngram in needle_ngrams:
            array_list.append(get_ngram_index_match(haystack_ngrams, ngram))

        ngram_matches = np.vstack(array_list)
        # Which indices in haystack_ngrams that matches needle_ngrams
        ngram_matches = ngram_matches.any(axis=0)
        # Add some zeroes at the end because ngrams of different size are not the same length
        ngram_matches = np.concatenate([ngram_matches, np.zeros(i - 1, dtype=bool)])
        ngram_matches = np.convolve(
            ngram_matches, np.ones(i + 2) * np.sqrt(np.log(i + 1)), mode="same"
        )  # Longer convolutions and higher weights for longer ngrams
        ngrams_bool_list.append(ngram_matches)

    ngram_matches = np.vstack(ngrams_bool_list)
    ngram_matches = ngram_matches.sum(axis=0)
    ngram_matches = ngram_matches / 3

    return ngram_matches


def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.

    Source: https://stackoverflow.com/q/4494404
    """

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    (idx,) = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def contiguous_ngram_match(needle, haystack, n=6, threshold=1, min_continuous_match=8, max_gap=30):
    """
    Get (fuzzy-ish) contiguous matching indices for haystack and needle
    using ngrams of several different sizes (from 1 to n) to construct weighted scores.

    Args:
        needle (str): A "needle" substring that is possibly contained in the "haystack".
        haystack (str): The larger text string that possibly contains the "needle".
        n (int): Maximum ngram size. Will use 1 to n ngram size.
        threshold (float): Weighted ngram score threshold for being considered a match.
        min_continous_match (int): Minimum continuous word matches for the region to
            be considered contiguous.
        max_gap (int): Maximum gap (in words) between contiguous region and the next/previous
            region for it to be seen as the start/end index of a larger joined together contiguous
            region.

    Returns:
        tuple: Start and end indices of contiguous fuzzy match in haystack
            (or whatever text is input as second arg).
    """

    if needle is None or haystack is None or len(haystack.split()) == 1:
        return None, None, None

    ngram_match_scores = get_weighted_ngram_score(needle, haystack, n=n)
    # Contiguous region indices satisfying the condition ngram_match_scores > threshold
    ngram_match_indices = contiguous_regions(ngram_match_scores > threshold)

    start_index = None
    end_index = None

    for i in range(0, len(ngram_match_indices)):
        if ngram_match_indices[i][1] - ngram_match_indices[i][0] > min_continuous_match:
            try:
                next_region_gap = ngram_match_indices[i + 1][0] - ngram_match_indices[i][1]

                if next_region_gap > max_gap:
                    # If gap is larger than max_gap, then this is not the start of a contiguous region
                    continue
            except IndexError:
                pass

            start_index = ngram_match_indices[i][0]
            break

    for i in reversed(range(0, len(ngram_match_indices))):
        if ngram_match_indices[i][1] - ngram_match_indices[i][0] > min_continuous_match:

            try:
                previous_region_gap = ngram_match_indices[i][0] - ngram_match_indices[i - 1][1]

                if previous_region_gap > max_gap:
                    continue
            except IndexError:
                pass

            end_index = ngram_match_indices[i][1]
            break

    score = ngram_match_scores[start_index:end_index].sum()

    if start_index is None or end_index is None:
        return None, None, None
    else:
        return start_index, end_index, score


def contiguous_ngram_match_star(args):
    """
    Wrapper for multiprocessing.
    Unpacks arguments and calls contiguous_ngram_match().
    """
    return contiguous_ngram_match(*args)


def get_fuzzy_match_word_indices(haystack, alignment):
    """
    Rapidfuzz and fuzzysearch return character indices for fuzzy matches.
    This function converts them to word indices.
    """

    word_spans = TreebankWordTokenizer().span_tokenize(haystack)
    word_spans = np.array(list(word_spans))

    if alignment.src_start == 0:
        start_index = 0
    else:
        start_index = np.where(word_spans <= alignment.src_start)[0][-1]

    if alignment.src_end == len(haystack):
        end_index = len(word_spans)
    else:
        end_index = np.where(word_spans >= alignment.src_end)[0][0]

    return start_index, end_index


def contiguous_fuzzy_match(needle, haystack, threshold=55):
    """
    Fuzzy contiguous index match for haystack and needle using
    fuzz.partial_ratio_alignment().

    Args:
        needle (str): A "needle" substring that is possibly contained in the "haystack".
        haystack (str): The larger text string that possibly contains the "needle".
        threshold (int): Fuzzy match score threshold for being considered a match.
            0 to 100, 100 being exact match.

    Returns:
        tuple: Start and end indices of contiguous fuzzy match in haystack,
            along with fuzzy match score of the matching segment.
    """

    if needle is None or haystack is None or len(haystack.split()) <= 1:
        return None, None, None

    align = fuzz.partial_ratio_alignment(haystack, needle)

    align_check = fuzz.partial_ratio_alignment(haystack[align.src_start : align.src_end], needle)

    # Sanity check to make sure the suggested alignment is correct
    if align_check.score < threshold and align.score < threshold:
        return None, None, None

    try:
        start_index, end_index = get_fuzzy_match_word_indices(haystack, align)
    except IndexError:
        return None, None, None

    return start_index, end_index, align_check.score


def contiguous_fuzzy_match_split(needle, haystack, threshold=55, max_length=300):
    """
    If input needle is too long, split it into start and end parts and run
    contiguous_fuzzy_match() on each part.

    Args:
        needle (str): A "needle" substring that is possibly contained in the "haystack".
        haystack (str): The larger text string that possibly contains the "needle".
        threshold (int): Threshold score for the fuzzy match to return indices.
        max_length (int): Length of the split needle parts.
    """
    needle_split = needle.split()
    nr_words_needle = len(needle_split)

    if nr_words_needle > (max_length * 2):
        start_index, end_index, score = contiguous_fuzzy_match(
            " ".join(needle_split[:max_length]), haystack, threshold=threshold
        )
        start_index_end, end_index_end, score_end = contiguous_fuzzy_match(
            " ".join(needle_split[-max_length:]), haystack, threshold=threshold
        )
    else:
        start_index, end_index, score = contiguous_fuzzy_match(
            needle, haystack, threshold=threshold
        )
        return start_index, end_index, score

    if score is not None and score_end is not None:
        combined_score = (score + score_end) / 2
    elif score is not None:
        combined_score = score
    elif score_end is not None:
        combined_score = score_end
    else:
        combined_score = None

    return start_index, end_index_end, combined_score


def contiguous_fuzzy_match_star(args):
    """
    Wrapper function for multiprocessing.
    Unpacks arguments and calls contiguous_fuzzy_match().
    """
    return contiguous_fuzzy_match(*args)


def contiguous_ngram_indices(
    df,
    column_in,
    column_out,
    n=6,
    threshold=1.3,
    min_continuous_match=8,
    max_gap=30,
    processes=None,
):
    """
    Find and return the indices of the contiguous text in column_out that
    matches text in column_in.

    Args:
        df (pd.DataFrame): DataFrame containing column_in and column_out.
        column_in (str): Column of "needle" substrings that are possibly contained in the "haystack".
        column_out (str): Column name of larger text string ("haystack").
        n (int): N-gram sizes 1 to n.
        threshold (float): Threshold score for contiguous n-gram match to be considered a match.
        min_continous_match (int): Minimum continuous word matches for the region to
            be considered contiguous.
        max_gap (int): Maximum gap (in words) between contiguous region and the next/previous
            region for it to be seen as the start/end index of a larger joined together contiguous
            region.
        processes (int | NoneType): Number of processes to use for multiprocessing.
            If None, use all available processes.
    """

    df

    with mp.Pool(processes) as pool:
        args = [
            (text1, text2, n, threshold, min_continuous_match, max_gap)
            for text1, text2 in zip(df[column_in], df[column_out])
        ]
        contiguous_ngram_list = list(
            tqdm(
                pool.imap(
                    contiguous_ngram_match_star,
                    args,
                    chunksize=1,
                ),
                total=len(df),
            )
        )

    return contiguous_ngram_list


def contiguous_fuzzy_indices(
    df,
    column_in,
    column_out,
    threshold=55,
    processes=None,
):
    """
    Find and return the indices of the contiguous text in column_out that
    matches text in column_in.

    Args:
        df (pd.DataFrame): DataFrame containing column_in and column_out.
        column_in (str): Column of "needle" substrings that are possibly contained in the "haystack".
        column_out (str): Column name of larger text string ("haystack").
        threshold (int): Threshold score for the fuzzy match to return indices.
        processes (int | NoneType): Number of processes to use for multiprocessing.
            If None, use all available processes.
    """

    with mp.Pool(processes) as pool:
        args = [(text1, text2, threshold) for text1, text2 in zip(df[column_in], df[column_out])]
        contiguous_fuzzy_list = list(
            tqdm(
                pool.imap(
                    contiguous_fuzzy_match_star,
                    args,
                    chunksize=1,
                ),
                total=len(df),
            )
        )

    return contiguous_fuzzy_list
