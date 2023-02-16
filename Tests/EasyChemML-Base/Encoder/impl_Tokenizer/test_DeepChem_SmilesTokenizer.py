from EasyChemML.Encoder.impl_Tokenizer.DeepChem_SmilesTokenizer import DeepChem_SmilesTokenizer

def test_vocabConvertion():
    test_smiles = "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"

    tokenizer = DeepChem_SmilesTokenizer()
    token_ids = tokenizer.convert_toIDs(test_smiles)
    token_str = tokenizer.idsToToken(token_ids)

    assert token_ids == [3, 7, 7, 9, 11, 7, 14, 7, 11, 7, 14, 7, 11, 7, 9, 24, 14, 11, 8, 15, 22, 15, 24, 14, 11, 8, 15, 22, 15, 24, 14, 11, 8, 15, 22, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert token_str == [['[CLS]'], ['C'], ['C'], ['1'], ['='], ['C'], ['('], ['C'], ['='], ['C'], ['('], ['C'], ['='], ['C'], ['1'], ['[N+]'], ['('], ['='], ['O'], [')'], ['[O-]'], [')'], ['[N+]'], ['('], ['='], ['O'], [')'], ['[O-]'], [')'], ['[N+]'], ['('], ['='], ['O'], [')'], ['[O-]'], ['[SEP]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]'], ['[PAD]']]

