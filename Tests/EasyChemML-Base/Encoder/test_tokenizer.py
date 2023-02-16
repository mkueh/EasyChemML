from rdkit import Chem

from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Environment import Environment

from EasyChemML.Encoder.BertTokenizer import BertTokenizer
from EasyChemML.Encoder import RdkitSmilesConverter

thread_count = 12

def load_data():
    env = Environment()

    load_dataset = {}
    load_dataset['doyle']          = XLSX('../../_TestDataset/DreherDoyle.xlsx', sheet_name='FullCV_01', columns=['Ligand', 'Additive', 'Base', 'Aryl halide', 'Output'], range=[0, 10])
    load_dataset['doyle_noChange'] = XLSX('../../_TestDataset/DreherDoyle.xlsx', sheet_name='FullCV_01', columns=['Ligand', 'Additive', 'Base', 'Aryl halide', 'Output'], range=[0, 10])
    di = DataImporter(env)
    dh = di.load_data_InNewBatchPartition(load_dataset)

    smiles_converter = RdkitSmilesConverter()
    smiles_converter.convert(dh['doyle'],columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=thread_count)
    smiles_converter.convert(dh['doyle_noChange'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=thread_count)

    return dh['doyle'], dh['doyle_noChange']

def test_tokenization():
    datatable, datatable_noChange = load_data()

    tokenizer = BertTokenizer()
    tokenizer.convert(datatable, ['Ligand'], 2)

    test = datatable[:]
    pass
