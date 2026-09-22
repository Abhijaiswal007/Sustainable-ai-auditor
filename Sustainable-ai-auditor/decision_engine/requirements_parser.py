import re


DEFAULT_MIN_ACCURACY = 0.90
DEFAULT_MAX_LATENCY_MS = 200.0


def convert_accuracy(value, unit=None):
    """
    Convert 90%, 90 percent, 90, or 0.90 into 0.90.
    """

    number = float(value)

    if unit or number > 1:
        number = number / 100

    if not 0 <= number <= 1:
        raise ValueError(
            f"Parsed accuracy {value} is invalid. "
            "Accuracy must be between 0 and 1, "
            "or between 0% and 100%."
        )

    return number


def parse_accuracy(text):
    """
    Find an accuracy requirement without accidentally using
    a latency or carbon number.
    """

    patterns = [
        # minimum accuracy of 90%
        r"(?:minimum|min|required)\s+accuracy\s*"
        r"(?:of|=|>=)?\s*"
        r"(\d+(?:\.\d+)?)\s*(%|percent)?",

        # accuracy at least 90%
        # accuracy of at least 90%
        # accuracy above 90%
        r"accuracy\s*"
        r"(?:of\s+at\s+least|at\s+least|of|=|>=|above|over|minimum\s+of)?"
        r"\s*(\d+(?:\.\d+)?)\s*(%|percent)?",

        # at least 90% accuracy
        # 0.90 accuracy
        r"(?:at\s+least|minimum|above|over)?\s*"
        r"(\d+(?:\.\d+)?)\s*(%|percent)?\s+accuracy"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return convert_accuracy(
                match.group(1),
                match.group(2)
            )

    return DEFAULT_MIN_ACCURACY


def parse_latency(text):
    """Find a latency or response-time requirement."""

    patterns = [
        # latency below 200 ms
        r"(?:latency|response\s+time)\s*"
        r"(?:must\s+be|of|=|<=|below|under|less\s+than|"
        r"at\s+most|maximum\s+of|max\s+of)?\s*"
        r"(\d+(?:\.\d+)?)\s*(?:ms|milliseconds?)",

        # below 200 ms latency
        r"(?:below|under|less\s+than|at\s+most|maximum|max)?\s*"
        r"(\d+(?:\.\d+)?)\s*(?:ms|milliseconds?)\s*"
        r"(?:latency|response\s+time)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            latency = float(match.group(1))

            if latency < 0:
                raise ValueError("Latency cannot be negative.")

            return latency

    return DEFAULT_MAX_LATENCY_MS


def parse_carbon_limit(text):
    """Find an optional maximum lifecycle-carbon requirement."""

    patterns = [
        # carbon below 50 kg
        r"(?:carbon|co2e|co2)\s*"
        r"(?:must\s+be|of|=|<=|below|under|less\s+than|"
        r"at\s+most|maximum\s+of|max\s+of)?\s*"
        r"(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)",

        # below 50 kg carbon
        r"(?:below|under|less\s+than|at\s+most|maximum|max)?\s*"
        r"(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)\s*"
        r"(?:carbon|co2e|co2)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            carbon = float(match.group(1))

            if carbon < 0:
                raise ValueError(
                    "Carbon limit cannot be negative."
                )

            return carbon

    return None


def parse_requirements(description):
    """
    Extract accuracy, latency, and carbon constraints
    from ordinary text without using an external API.
    """

    if not description or not description.strip():
        return {
            "min_accuracy": DEFAULT_MIN_ACCURACY,
            "max_latency_ms": DEFAULT_MAX_LATENCY_MS,
            "max_carbon_kgco2e": None
        }

    text = description.lower().strip()

    return {
        "min_accuracy": parse_accuracy(text),
        "max_latency_ms": parse_latency(text),
        "max_carbon_kgco2e": parse_carbon_limit(text)
    }