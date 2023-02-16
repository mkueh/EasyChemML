import re

from typing import Dict, List, Tuple

from rdkit import Chem
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import CharDelimiterSplit
from tokenizers.processors import TemplateProcessing

vocab_BERT_RXN = [
    "[PAD]"
    , "[unused0]"
    , "[UNK]"
    , "[CLS]"
    , "[SEP]"
    , "[MASK]"
    , "c"
    , "C"
    , "O"
    , "1"
    , "2"
    , "="
    , "N"
    , "S"
    , "("
    , ")"
    , "Br"
    , "B"
    , "n"
    , "3"
    , "[N+](=O)[O-]"
    , "-"
    , "[O-]"
    , "[nH]"
    , "[N+]"
    , "Cl"
    , "s"
    , "F"
    , "#"
    , "o"
    , "P"
    , "I"
    , "4"
    , "[H]"
    , "[Si]"
    , "[n+]"
    , "[NH+]"
    , "[N-]"
    , "5"
    , "[B-]"
    , "6"
    , "[SiH]"
    , "[o+]"
    , "[c-]"
    , "7"
    , "8"
    , "[O]"
    , "[C-]"
    , "[S+]"
    , "[n-]"
    , "[S-]"
    , "[PH]"
    , "[CH-]"
    , "[SiH3]"
    , "[C+]"
    , "[SiH2]"
    , "[NH2+]"
    , "[SH]"
    , "p"
    , "[C]"
    , "[I+]"
    , "[BH3-]"
    , "[s+]"
    , "9"
    , "[O+]"
    , "[nH+]"
    , "[cH-]"
    , "[IH]"
    , "[Cl+3]"
    , "[P+]"
    , "[CH]"
    , "[N]"
    , "[BH-]"
    , "[PH+]"
    , "[Si-]"
    , "[NH-]"
    , "[SH-]"
    , "[S]"
    , "[siH]"
    , "[NH3+]"
    , "[BH2-]"
    , "[CH+]"
    , "[pH]"
    , "[OH+]"
    , "[c+]"
    , "[I-]"
    , "[P-]"
    , "[Si+]"
    , "[PH2]"
    , "[SH+]"
    , "[Cl+]"
    , "[CH2]"
    , "[IH+]"
    , "[SiH4]"
    , "."
    , "[Cl-]"
    , "[Br-]"
    , "%10"
    , "%11"
    , "[cH+]"
    , "[F-]"
    , "%12"
    , "%13"
    , "[P]"
    , "[BH4-]"]

SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

class NotCanonicalizableSmilesException(ValueError):
    pass

class SmilesTokenzier():
    vocab: Dict[str, int]
    max_length: int
    bert_tokenizer: Tokenizer

    def convertListToDict(self) -> dict:
        out_dict = {}
        for index, val in enumerate(vocab_BERT_RXN):
            out_dict[val] = index
        return out_dict

    def __init__(self, max_length: int = 70, padding:bool = True, truncation:bool = True):
        self.vocab = self.convertListToDict()
        self.max_length = max_length

        self._constructBERT_Tokenizer(padding, truncation)
        self.regex_splitter = re.compile(SMI_REGEX_PATTERN)

    def _constructBERT_Tokenizer(self, padding:bool, truncation:bool):
        self.bert_tokenizer = Tokenizer(WordPiece(vocab=self.vocab, unk_token="[UNK]"))
        self.bert_tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
        self.bert_tokenizer.pre_tokenizer = CharDelimiterSplit(delimiter=' ')
        self.bert_tokenizer.post_processor = TemplateProcessing(
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            single="[CLS] $A [SEP]",
            special_tokens=[
                ("[CLS]", self.vocab['[CLS]']),
                ("[SEP]", self.vocab['[SEP]']),
            ],
        )

        if padding:
            self.bert_tokenizer.enable_padding(length=self.max_length)

        if truncation:
            self.bert_tokenizer.enable_truncation(max_length=self.max_length)

    def _splitSMILES(self, text: str) -> List[str]:
        tokens = [token for token in self.regex_splitter.findall(text)]
        return tokens

    def _boundariesCheck(self, splitted_seq: List[str]):
        if len(splitted_seq) > self.max_length:
            raise Exception('The sequence generates more tokens than defined via max_length')

    def encode(self, seq: str) -> Tuple[List[str], List[int]]:
        #seq = self.process_reaction(seq)
        splitted_seq = self._splitSMILES(seq)
        #self._boundariesCheck(splitted_seq)
        filled_whitespace = ' '.join(splitted_seq)

        encoded_data = self.bert_tokenizer.encode(filled_whitespace)
        return encoded_data.tokens, encoded_data.ids

    def decode(self, seq: List[int]) -> List[str]:
        return self.bert_tokenizer.decode(seq)
