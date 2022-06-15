import os
import re
from typing import List, Generator, Tuple

import configuration


# --------------------------------------------------------------------------- #
def create_token_stream(line: str) -> List[str]:

    line = line.lower()

    for character in configuration.TERM_SPLIT_CHARACTERS:
        line = line.replace(character, " ")

    if line.endswith('.'):
        line = line[:-1]

    return [x.strip() for x in re.split(" ", line) if x not in configuration.STOP_WORDS]


# --------------------------------------------------------------------------- #
def tokenize_documents(documents: List[str]) -> Generator[Tuple[int, List[str]], None, None]:
    """
    :param documents: list of documents
    :return: generator - docID, tokens in order of apperance
    """
    for line in documents:
        tmp = re.split(r'\t', line)
        if len(tmp) != 2:
            continue
        docid, text = tmp[0].split("-")[1].strip(), tmp[1]
        docid = docid.replace('MED-', '')
        tokens = create_token_stream(text)

        yield int(docid), tokens