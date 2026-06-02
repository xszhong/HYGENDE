import re
from .register import Register
from .logger_config import LoggerConfig

extract_label_register = Register("extract_label")


@extract_label_register.register("default")
def default_extract_label(text):
    logger = LoggerConfig.get_logger("extract_label")
    if text is None:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"

    text = text.lower()
    pattern = r"final answer:\s+<begin>(.*)<end>"

    match = re.findall(pattern, text)
    if len(match) > 0:
        return match[-1]
    else:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"

#This is for the Paddle and Harvard task
@extract_label_register.register("paddle")
def depression_extract_label(text):
    logger = LoggerConfig.get_logger("extract_label")
    if text is None:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"

    # only keep the part after "Final answer:"
    text = text.lower()

    pattern = r"final answer:\s+(yes|no|other)"

    match = re.findall(pattern, text.lower())
    if match:
        answer = match[-1] if len(match) > 0 else None
        if answer == "yes":
            return "yes"
        elif answer == "no":
            return "no"
    logger.warning(f"Could not extract label from text: {text}")
    return "other"

@extract_label_register.register("harvard")
def depression_extract_label(text):
    logger = LoggerConfig.get_logger("extract_label")
    if text is None:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"

    # only keep the part after "Final answer:"
    text = text.lower()

    pattern = r"final answer:\s+(yes|no|other)"

    match = re.findall(pattern, text.lower())
    if match:
        answer = match[-1] if len(match) > 0 else None
        if answer == "yes":
            return "yes"
        elif answer == "no":
            return "no"
    logger.warning(f"Could not extract label from text: {text}")
    return "other"

#This is for the Dept-2 task
@extract_label_register.register("bidetweet")
def bidetweet_extract_label(text):
    logger = LoggerConfig.get_logger("extract_label")
    if text is None:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"

    # only keep the part after "Final answer:"
    text = text.lower()

    pattern = r"final answer:\s+(yes|no|other)"

    match = re.findall(pattern, text.lower())
    if match:
        answer = match[-1] if len(match) > 0 else None
        if answer == "yes":
            return "yes"
        elif answer == "no":
            return "no"
    logger.warning(f"Could not extract label from text: {text}")
    return "other"

#This is for the Dept-4 task
@extract_label_register.register("fourdetweet")
def fourdetweet_extract_label(text):
    logger = LoggerConfig.get_logger("extract_label")
    if text is None:
        logger.warning(f"Could not extract label from text: None")
        return "other"

    # only keep the part after "Final answer:"
    text = text.lower()

    pattern = r"final answer:\s+(0|1|2|3)"

    match = re.findall(pattern, text)
    if match:
        answer = match[-1] if len(match) > 0 else None
        if answer == "0":
            return "0"
        elif answer == "1":
            return "1"
        elif answer== "2":
            return "2"
        elif answer == "3":
            return "3"
    logger.warning(f"Could not extract label from text: {text}")
    return "other"
