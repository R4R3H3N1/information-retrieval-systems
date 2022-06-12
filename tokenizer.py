import os
import re
from typing import List, Generator, Tuple

import configuration


# --------------------------------------------------------------------------- #
def get_token_from_line(line: str) -> List[str]:

    line = line.lower()

    for character in configuration.TERM_SPLIT_CHARACTERS:
        line = line.replace(character, " ")

    #if line[len(line) - 1] == ".":
    if line.endswith('.'):
        line = line[:-1]

    return [x.strip() for x in re.split(" ", line) if x not in configuration.STOP_WORDS]


# --------------------------------------------------------------------------- #
def exclude_abstract_beginnings(abstract: str) -> str:
    """
    remove terms like preface, summary, etc. from document abstarct
    """
    for beginning in configuration.ABSTRACT_BEGINNINGS:
        abstract = re.sub(r'^' + beginning, '', abstract, re.IGNORECASE).strip()
    return abstract


# --------------------------------------------------------------------------- #
def tokenize_documents(documents: List[str]) -> Generator[Tuple[str, List[str]], None, None]:
    """
    :param documents: list of documents
    :return: generator - docID, tokens in order of apperance
    """
    for line in documents:
        tmp = re.split(r'\t', line)
        if len(tmp) != 2:
            continue
        docID, text = int(tmp[0].split("-")[1].strip()), tmp[1]
        new_tokens = get_token_from_line(text)

        yield int(docID), new_tokens


