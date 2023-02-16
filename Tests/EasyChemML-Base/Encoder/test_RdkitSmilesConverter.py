from rdkit import Chem

from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Encoder import RdkitSmilesConverter
from EasyChemML.Environment import Environment

thread_count = 12

def load_data():
    env = Environment()

    load_dataset = {}
    load_dataset['nplscore']          = XLSX('../../_TestDataset/DreherDoyle.xlsx', sheet_name='FullCV_01', columns=['Ligand', 'Additive', 'Base', 'Aryl halide', 'Output'], range=[0, 500])
    load_dataset['nplscore_noChange'] = XLSX('../../_TestDataset/DreherDoyle.xlsx', sheet_name='FullCV_01', columns=['Ligand', 'Additive', 'Base', 'Aryl halide', 'Output'], range=[0, 500])
    di = DataImporter(env)
    dh = di.load_data_InNewBatchPartition(load_dataset)
    return dh['nplscore'], dh['nplscore_noChange']

def test_convert_singleRow_noNew_noReplace():
    datatable, datatable_noChange = load_data()
    RdkitSmilesConverter().convert(datatable=datatable, columns=['Ligand'], n_jobs=thread_count)

    #print(X.to_String())

    size = len(datatable)
    for i in range(size):
        smiles = datatable[i]['Ligand'].decode("utf-8")
        assert smiles == Chem.MolToSmiles(Chem.MolFromSmiles(datatable_noChange[i]['Ligand']),isomericSmiles=False)

def test_convert_multiRow_noNew_noReplace():
    datatable, datatable_noChange = load_data()
    RdkitSmilesConverter().convert(datatable=datatable, columns=['Ligand', 'Additive'], n_jobs=thread_count)

    #print(X.to_String())

    size = len(datatable)
    for i in range(size):
        smiles_ligand = datatable[i]['Ligand'].decode("utf-8")
        smiles_add = datatable[i]['Additive'].decode("utf-8")
        assert smiles_ligand == Chem.MolToSmiles(Chem.MolFromSmiles(datatable_noChange[i]['Ligand']),isomericSmiles=False)
        assert smiles_add == Chem.MolToSmiles(Chem.MolFromSmiles(datatable_noChange[i]['Additive']), isomericSmiles=False)

def test_convert_singleRow_New_noReplace():
    datatable, datatable_noChange = load_data()
    RdkitSmilesConverter().convert(datatable=datatable, columns=['Ligand'], createNewColumns=['Ligand_MOL'], n_jobs=thread_count)
    #print(X.to_String())

    size = len(datatable)
    for i in range(size):
        smiles = datatable[i]['Ligand_MOL'].decode("utf-8")
        assert smiles == Chem.MolToSmiles(Chem.MolFromSmiles(datatable_noChange[i]['Ligand']),isomericSmiles=False)

def test_convert_multiRow_New_noReplace():
    datatable, datatable_noChange = load_data()
    RdkitSmilesConverter().convert(datatable=datatable, columns=['Ligand', 'Additive'], createNewColumns=['Ligand_MOL', 'Additive_MOL'], n_jobs=thread_count)
    #print(X.to_String())

    size = len(datatable)
    for i in range(size):
        smiles_ligand = datatable[i]['Ligand_MOL'].decode("utf-8")
        smiles_add = datatable[i]['Additive_MOL'].decode("utf-8")
        assert smiles_ligand == Chem.MolToSmiles(Chem.MolFromSmiles(datatable_noChange[i]['Ligand']),isomericSmiles=False)
        assert smiles_add == Chem.MolToSmiles(Chem.MolFromSmiles(datatable_noChange[i]['Additive']), isomericSmiles=False)

def test_convert_singleRow_New_noReplace_addHs():
    datatable, datatable_noChange = load_data()
    RdkitSmilesConverter().convert(datatable=datatable, columns=['Ligand'], createNewColumns=['Ligand_MOL'],  isomericSmiles=False, AddHs=True, n_jobs=thread_count)

    #print(X.to_String())

    size = len(datatable)
    for i in range(size):
        smiles = datatable[i]['Ligand_MOL'].decode("utf-8")
        assert smiles == Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(datatable_noChange[i]['Ligand'])),isomericSmiles=False)

def test_convert_multiRow_New_noReplace_addHs():
    datatable, datatable_noChange = load_data()
    RdkitSmilesConverter().convert(datatable=datatable, columns=['Ligand', 'Additive'], createNewColumns=['Ligand_MOL', 'Additive_MOL'],  isomericSmiles=False, AddHs=True, n_jobs=thread_count)

    #print(X.to_String())

    size = len(datatable)
    for i in range(size):
        smiles_ligand = datatable[i]['Ligand_MOL'].decode("utf-8")
        smiles_add = datatable[i]['Additive_MOL'].decode("utf-8")
        assert smiles_ligand == Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(datatable_noChange[i]['Ligand'])),isomericSmiles=False)
        assert smiles_add == Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(datatable_noChange[i]['Additive'])),isomericSmiles=False)