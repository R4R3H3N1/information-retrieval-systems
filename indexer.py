from __future__ import annotations

import numpy as np

from tokenizer import tokenize_documents
import json
import time
import os
from typing import Type, List, Set

import configuration


# =========================================================================== #
class Index:
    __slots__ = ('dictionary', 'termClassMapping', 'documentIDs', 'kgramMap', "average_doc_len")

    def __init__(self, filename: str):

        self.dictionary = {}
        self.termClassMapping = {}
        self.kgramMap = {}
        self.documentIDs = {}
        self.average_doc_len = 0.0

        print("Started creating index")
        start = time.time()
        self.invoke_toknizer(filename)
        self.calculate_average_doc_len()
        print(f"Creating index took {round(time.time() - start, 3)} seconds.")

    # --------------------------------------------------------------------------- #
    def invoke_toknizer(self, filename: str) -> None:

        try:
            with open(filename, 'r', encoding='utf8') as f:
                file = f.read()
                docs = file.split('\n')
        except FileNotFoundError:
            print(f'ERROR: file {filename} not found')
            return

        for docID, tokens in tokenize_documents(docs):
            positionCounter = 1
            int_doc_id = int(docID)
            self.documentIDs[int_doc_id] = len(tokens)
            for token in tokens:
                try:
                    ti = self.termClassMapping[token]
                except KeyError:
                    ti = TermIndex(token)
                    self.termClassMapping[token] = ti

                try:
                    self.dictionary[ti].append(int_doc_id, positionCounter)
                    ti.occurence += 1
                except KeyError:
                    self.dictionary[ti] = Postinglist(int_doc_id, positionCounter)

                positionCounter += 1

        for key, val in self.dictionary.items():
            val.final_sort_postinglist()

    # --------------------------------------------------------------------------- #
    def calculate_average_doc_len(self):
        average_length = 0
        for doc_id, doc_length in self.documentIDs.items():
            average_length += doc_length
        self.average_doc_len = average_length / len(self.documentIDs.items())


# =========================================================================== #
class TermIndex:
    __slots__ = ('term', 'occurence')

    def __init__(self, term: str):
        self.term = term
        self.occurence = 1

    def __hash__(self):
        return hash(self.term)


# =========================================================================== #
class Postinglist:
    __slots__ = ('plist', 'seenDocIDs', 'positions', 'occurrence')

    def __init__(self, docID: int = None, position: int = None):
        self.plist = []   # List of sorted DocIDs
        self.positions = {}  # map docID:positions within docID
        self.occurrence = 1

        self.seenDocIDs = set()
        if docID:
            self.append(docID, position)

    def __len__(self):
        return len(self.plist)

    def __getitem__(self, idx):
        return self.plist[idx]

    # --------------------------------------------------------------------------- #
    def append(self, docID: str, position: int | List[int]) -> None:
        if isinstance(position, list):
            try:
                [self.positions[docID].append(pos) for pos in position]
                self.occurrence += len(position)
            except KeyError:
                self.positions[docID] = position
                self.occurrence += len(position)
        else:
            try:
                self.positions[docID].append(position)
                self.occurrence += 1
            except KeyError:
                self.positions[docID] = [position]
                self.occurrence += 1

        if docID in self.seenDocIDs:
            pass
        else:
            self.plist.append(docID)
            self.seenDocIDs.add(docID)

    # --------------------------------------------------------------------------- #
    def final_sort_postinglist(self) -> None:
        self.plist = sorted(self.plist)