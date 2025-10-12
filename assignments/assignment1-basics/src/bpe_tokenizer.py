from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable, Iterator
import regex as re

def merge(indices: list[list[int]], pair: tuple[int, int]) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    indices_len=len(indices)
    while i < indices_len:
        if i + 1 < indices_len and indices[i] == list(pair[0]) and indices[i + 1] == list(pair[1]):
            new_indices.append(indices[i] + indices[i + 1]) 
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError

class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.pattern = re.compile("|".join(map(re.escape, special_tokens))) if special_tokens!=None else None
        reverse_dict = {}
        for key, value in vocab.items():
            reverse_dict[value] = key
        self.reverse_dict=reverse_dict

    def encode_non_spec(self, string: str) -> list[int]:
        indices = [[char] for char in string.encode("utf-8")]
        for pair in self.merges: 
            indices = merge(indices, pair)
        tokens=[]
        for token in indices:
            tokens.append(self.reverse_dict[bytes(token)])
        return tokens

    def encode(self, string: str) -> list[int]:
        start=0
        tokens=[]
        if self.pattern != None:
            list_matches = list(self.pattern.finditer(string))
            len_matches=len(list_matches)
            if len_matches == 0:
                tokens=self.encode_non_spec(string)
            else:
                for match in list_matches:
                    tokens=tokens+self.encode_non_spec(string[start:match.start()])
                    start=match.end()
                    tokens=tokens+[self.reverse_dict[match.group().encode("utf-8")]]
                tokens=tokens+self.encode_non_spec(string[start:])
        else:
            tokens=self.encode_non_spec(string)
        return tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            token_ids = self.encode(text_chunk)
            yield token_ids

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string

