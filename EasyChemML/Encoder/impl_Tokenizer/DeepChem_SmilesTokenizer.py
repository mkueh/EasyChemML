import os
from typing import List

from deepchem.feat.smiles_tokenizer import SmilesTokenizer


class DeepChem_SmilesTokenizer:
    smile_tok: SmilesTokenizer
    token_dict: dict
    transposed_tokenDict: dict
    max_length: int
    padding: str
    truncation: bool

    def __init__(self, vocab=None, max_length=100, padding='max_length', truncation=True):
        if vocab is None:
            vocab = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vocab_SCHWALLER.txt')

        self.smile_tok = SmilesTokenizer(vocab)
        self.token_dict = self.smile_tok.get_vocab()
        self.transposed_tokenDict = self._transpose_dict(self.token_dict)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def convert_toIDs(self, seq: str):
        return self.smile_tok.encode(seq, padding='max_length', truncation=True, max_length=self.max_length)

    def idsToToken(self, id_seq: List[int]):
        out = []
        for entry in id_seq:
            out.append(self.transposed_tokenDict[entry])
        return out

    def _transpose_dict(self, input_dict) -> dict:
        from collections import defaultdict
        d = defaultdict(list)
        for k, v in input_dict.items():
            d[v].append(k)

        return d
