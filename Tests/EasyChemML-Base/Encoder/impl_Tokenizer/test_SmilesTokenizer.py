from EasyChemML.Encoder.impl_Tokenizer.SmilesTokenizer_SchwallerEtAll import SmilesTokenzier

def test_molWithNitro_Encode():
    test_smiles = "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"

    tokenizer = SmilesTokenzier(max_length=100)
    out_tokens, out_ids = tokenizer.encode(test_smiles)

    assert out_ids == [3, 7, 7, 9, 11, 7, 14, 7, 11, 7, 14, 7, 11, 7, 9, 24, 14, 11, 8, 15, 22, 15, 24, 14, 11, 8, 15,
                       22, 15, 24, 14, 11, 8, 15, 22, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert out_tokens == ['[CLS]', 'C', 'C', '1', '=', 'C', '(', 'C', '=', 'C', '(', 'C', '=', 'C', '1', '[N+]', '(', '=', 'O', ')', '[O-]', ')', '[N+]', '(', '=', 'O', ')', '[O-]', ')', '[N+]', '(', '=', 'O', ')', '[O-]', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']

def test_molWithNitro_Decode():
    in_ids = [3, 7, 7, 9, 11, 7, 14, 7, 11, 7, 14, 7, 11, 7, 9, 24, 14, 11, 8, 15, 22, 15, 24, 14, 11, 8, 15,
                       22, 15, 24, 14, 11, 8, 15, 22, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    correct_tokens = ['[CLS]', 'C', 'C', '1', '=', 'C', '(', 'C', '=', 'C', '(', 'C', '=', 'C', '1', '[N+]', '(',
                          '=', 'O', ')', '[O-]', ')', '[N+]', '(', '=', 'O', ')', '[O-]', ')', '[N+]', '(', '=', 'O',
                          ')', '[O-]', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
                          '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
                          '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
                          '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
                          '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
                          '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
                          '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']

    tokenizer = SmilesTokenzier(max_length=100)

    result = tokenizer.decode(in_ids)

    assert result.split(' ') == correct_tokens
