"""Final answer parser for reasoning tasks.

The common forms of outputs to be parsed are like:
- "the answer: XXX"
- "XXX is the answer"
- "XXX is the final/right/correct answer"
"""

import dataclasses
import re
import string
from typing import Dict, List, Sequence
from rouge import Rouge
import pdb

word2num = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12"
}

all_lowercase_letters = string.ascii_lowercase  # "abcd...xyz"
all_uppercase_letters = string.ascii_uppercase
bracketed_letters_list = set([f"({l})" for l in (all_lowercase_letters + all_uppercase_letters)])  # ['(a)', ...]

SPECIAL_NUM_CHARS = frozenset({".", "/", ","})
# The logic for identifying patterns for the answer behind:
# First check if the primary patterns are in the string, then if not, check the
# secondary ones.
FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY = ["answer is ", "answer: ", "answer is: "]
FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY = ["is: ", "are: "]
FINAL_ANSWER_AHEAD_PATTERNS = [
    " is the correct answer",
    " is the right answer",
    " is the final answer",
    " is the answer",
]
GSM8K_ANSWER = "#### "
# the Boolean symbols appeared in BBH tasks
BOOLEAN_SYMBOLS = [["false", "true"], ["no", "yes"], ["invalid", "valid"]]

def calc_rouge(pred, ans):
    rouge = Rouge()
    if isinstance(ans, str):
        scores = rouge.get_scores(pred, ans)[0]['rouge-l']['f']
    elif isinstance(ans, list):
        scores_list = [rouge.get_scores(pred, a)[0]['rouge-l']['f'] for a in ans]
        # pdb.set_trace()
        scores = sum(scores_list) / len(scores_list)
    return scores

def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def remove_punctuation_from_string(input_string):
    output_string = input_string.translate(str.maketrans("", "", string.punctuation))
    return output_string


def _extract_bracketed_choice_from_string(prediction, is_lower=True):
    """Extract bracketed ABCD...XYZ choices there's exactly one bracketed choice.

    Args:
      prediction (str): the unprocessed prediction.

    Returns:
      prediction (str): the processed prediction.
    """
    if is_lower:
        prediction = prediction.lower()
    choice_in_pred_all = set([item in prediction for item in bracketed_letters_list])
    if sum(choice_in_pred_all) >= 1:
        prediction = re.findall(r"\([A-Za-z]\)", prediction)[0]
    return prediction


def get_normalized_prediction(
    prediction: str,
    is_multiple_choice: bool,
    treat_as_number: bool,
    num_decimals: int = 0,
    treat_as_bool: bool = False,
) -> str:
    """Returns a normalized prediction for use in `number_included_accuracy`.

    Args:
      prediction: The original model prediction.
      treat_as_number: Whether to treat the prediction as a number (and perform
        additional post-processing relevant to numbers, such as stripping of units
        or normalization of thousand separators, etc.).
      num_decimals: Number of decimal places to which to round the answer. Only
        applicable when treat_as_number==True.
      treat_as_bool: Whether to treat the prediction as a Boolean object. Only set
        it to True when the target is Boolean. The parser will then convert an 0/1
        answer to False/True.

    Returns:
      A normalized answer string that can be directly compared with the normalized
      golden answer in order to determine the `number_included_accuracy`.
    """

    prediction_parsed = prediction.lower().strip()
    
    FINAL_ANSWER_BEHIND_PATTERNS = ( 
        FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY 
        if any([item in prediction for item in FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY])
        else FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY
    )
    
    DELIMITERS_FOR_ANSWER_BEHIND = (
        [GSM8K_ANSWER]
        + FINAL_ANSWER_BEHIND_PATTERNS
    )
    DELIMITERS_FOR_ANSWER_AHEAD = (
        FINAL_ANSWER_AHEAD_PATTERNS 
    )
    
    answer_indicated = False
    for answer_delimiter in DELIMITERS_FOR_ANSWER_BEHIND:
        if answer_delimiter.lower() in prediction_parsed:
            prediction_parsed = prediction_parsed.split(answer_delimiter.lower())[-1]
            answer_indicated = True

    for answer_delimiter in DELIMITERS_FOR_ANSWER_AHEAD:
        if answer_delimiter.lower() in prediction_parsed:
            prediction_parsed = prediction_parsed.split(answer_delimiter.lower())[0]
            answer_indicated = True

    # extract the bracketed choices: "(A) apple" -> "(a)"
    if is_multiple_choice:
        prediction_parsed = _extract_bracketed_choice_from_string(prediction_parsed)

    def _parse_without_treating_as_number(prediction_parsed):
        prediction_parsed = prediction_parsed.split(".")[0]
        return prediction_parsed
        
    def _parse_with_treating_as_number(prediction_parsed):
        
        prediction_parsed_list = prediction_parsed.split(' ')
        new_prediction_parsed = []
        for pred_parsed_word in prediction_parsed_list:
            if pred_parsed_word in word2num:
                new_prediction_parsed.append(word2num[pred_parsed_word])
            else:
                new_prediction_parsed.append(pred_parsed_word)
        
        prediction_parsed = ' '.join(new_prediction_parsed)
        
        prediction_parsed = prediction_parsed.split("=")[-1]
        for c in ["$", ",", "%", "€", "£", ":"]:
            prediction_parsed = prediction_parsed.replace(c, "")
        prediction_parsed = prediction_parsed.strip()
        corrected_answer = False

        if not corrected_answer:  # If no calculator errors were made.
            # '5600 pounds' -> '5600'; 'the 6th' -> '6'.
            if answer_indicated:
                # Take the first token that has numerical values.
                parts = prediction_parsed.split(" ")
            else:
                # Take the last token that has numerical values.
                parts = list(reversed(prediction_parsed.split(" ")))

            prediction_parsed = parts[0]  # Default
            for part in parts:
                if any(chr.isdigit() for chr in part):  # Filter out any digit tokens.
                    prediction_parsed = part
                    break

            # '156kgs' -> 156. '823-yard' -> 823.
            while prediction_parsed and prediction_parsed[-1].isalpha():
                prediction_parsed = prediction_parsed[:-1]
            if prediction_parsed and prediction_parsed[-1] == "-":
                prediction_parsed = prediction_parsed[:-1]

        if _is_float(prediction_parsed):
            prediction_parsed_float = round(float(prediction_parsed), num_decimals)
            prediction_parsed = "{:.{num_decimals}f}".format(
                prediction_parsed_float, num_decimals=num_decimals
            )
        else:
            if re.search(r"(\d+)(?!.*\d)", prediction_parsed):
                prediction_parsed = re.search(r"(\d+)(?!.*\d)", prediction_parsed)[0]
        return prediction_parsed
    
    if treat_as_number:
        prediction_parsed = _parse_with_treating_as_number(prediction_parsed)


    if treat_as_bool:
        prediction_parsed_as_not_number = _parse_without_treating_as_number(
            prediction_parsed
        )
        prediction_parsed_as_not_number = prediction_parsed_as_not_number.split(' ')
        if any(
            [prediction_parsed_as_not_number in item for item in BOOLEAN_SYMBOLS]
        ):
            prediction_parsed = prediction_parsed_as_not_number
        # remove punctuations like ":" and then strip
        prediction_parsed = remove_punctuation_from_string(prediction_parsed).strip()

    return prediction_parsed


@dataclasses.dataclass
class NormalizationResult:
    """Bundle of return values of get_normalized_target_and_prediction.

    Attributes:
      target: Normalized target string, suitable for direct comparison with the
        normalized prediction.
      prediction: Normalized prediction string, suitable for direct comparison
        with the normalized target.
      treat_as_number: Whether it was determined to treat the prediction as a
        number (and perform additional post-processing relevant to numbers, such
        as stripping of units or normalization of thousand separators, etc.).
      num_decimals: Number of decimal places to which it was determined to round
        the answer. Only relevant when treat_as_number==True.
    """

    target: str
    prediction: str
    treat_as_number: bool
    num_decimals: int


def get_normalized_target_and_prediction(
    target: str, prediction: str
) -> NormalizationResult:
    """Returns a normalized target and prediction for `number_included_accuracy`.

    Args:
      target: Target (i.e., golden answer). The function will automatically
        perform light normalization on the target, such as stripping off any
        answer indication prefixes like "The answer is".
      prediction: Original model prediction. The function will automatically
        normalize the prediction by stripping off trailing punctuation and any
        answer indication prefixes like "The answer is". If the target is numeric,
        will further strip units and round to the same precision as the target.

    Returns:
      The normalized target and prediction, along with related information
      indicating the types of normalization that were performed.
    """

    def _any_list_item_in_string(test_list, test_string):
        return any(item in test_string for item in test_list)

    primary_after_patterns_in_target = _any_list_item_in_string(
        FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY, target
    )
    secondary_after_patterns_in_target = _any_list_item_in_string(
        FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY, target
    )
    target = target.lower()
    if (
        primary_after_patterns_in_target
        or (secondary_after_patterns_in_target and not primary_after_patterns_in_target)
        or _any_list_item_in_string(FINAL_ANSWER_AHEAD_PATTERNS, target)
        or GSM8K_ANSWER in target
    ):
        if primary_after_patterns_in_target:
            target = re.split(r"|".join(FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY), target)[
                -1
            ]
        elif (
            secondary_after_patterns_in_target and not primary_after_patterns_in_target
        ):
            target = re.split(
                r"|".join(FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY), target
            )[-1]
        target = re.split(r"|".join(FINAL_ANSWER_AHEAD_PATTERNS), target)[0]
        target = target.split(GSM8K_ANSWER)[-1]
        if target and target[-1] in [";", ",", "."] and _is_float(target[:-1]):
            target = target[:-1]

    treat_as_number = _is_float(target)
    if treat_as_number and "." in target:
        num_decimals = len(target.split(".")[-1])
    else:
        num_decimals = 0

    normalized_prediction = get_normalized_prediction(
        prediction, treat_as_number=treat_as_number, num_decimals=num_decimals
    )

    return NormalizationResult(
        target=target,
        prediction=normalized_prediction,
        treat_as_number=treat_as_number,
        num_decimals=num_decimals,
    )


def number_included_accuracy_list(
    targets: Sequence[str],
    predictions: Sequence[str],
) -> List[bool]:
    """Returns a list of booleans for if the target is anywhere in the prediction.

    Args:
      targets: Targets (i.e., golden answers).
      predictions: Original model predictions (before normalization).
    """

    correct_list = []
    for prediction, target in zip(predictions, targets):
        normalization_result = get_normalized_target_and_prediction(
            target=target, prediction=prediction
        )

        # If answer is not a number, then look for exact match.
        if not normalization_result.treat_as_number:
            correct_list.append(
                normalization_result.target == normalization_result.prediction
            )

        else:  # If the target is a number, then compare numerically.
            correct = False  # pylint: disable=unused-variable
            try:
                prediction_parsed_float = round(
                    float(normalization_result.prediction),
                    normalization_result.num_decimals,
                )
                correct = (
                    abs(prediction_parsed_float - float(normalization_result.target))
                    <= 1e-5
                )
            except ValueError:
                correct = False
            except IndexError:
                correct = False
            correct_list.append(correct)
    return correct_list


def number_included_accuracy(
    targets: Sequence[str], predictions: Sequence[str]
) -> Dict[str, float]:
    """Special accuracy for if the target is anywhere in the prediction."""

    correct_list = number_included_accuracy_list(targets, predictions)

    correct_list_with_calc = number_included_accuracy_list(targets, predictions)

    return {
        "accuracy": sum(correct_list) / len(correct_list) * 100,
        "accuracy_with_calc": sum(correct_list_with_calc)
        / len(correct_list_with_calc)
        * 100,
    }


if __name__ == "__main__":
    ans = get_normalized_prediction(prediction="  Sure! Faye has 66 pencils in total.", is_multiple_choice=False, treat_as_number=True, num_decimals=0, treat_as_bool=False)
    print(ans)