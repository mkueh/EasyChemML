from rdkit import Chem

from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Encoder import MolRdkitConverter
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
    MolRdkitConverter().convert(datatable=datatable, columns=['Ligand'], n_jobs=thread_count)

    size = len(datatable)
    for i in range(size):
        mol:Chem.Mol = datatable[i]['Ligand']
        smilesOfMol = Chem.MolToSmiles(mol, isomericSmiles=False)
        assert smilesOfMol == Chem.MolToSmiles(Chem.MolFromSmiles(datatable_noChange[i]['Ligand']),isomericSmiles=False)

def test_convert_singleRow_New_noReplace():
    datatable, datatable_noChange = load_data()
    MolRdkitConverter().convert(datatable=datatable, columns=['Ligand'], createNewColumns=['Ligand_MOL'], n_jobs=thread_count)

    size = len(datatable)
    for i in range(size):
        mol:Chem.Mol = datatable[i]['Ligand_MOL']
        smilesOfMol = Chem.MolToSmiles(mol, isomericSmiles=False)
        assert smilesOfMol == Chem.MolToSmiles(Chem.MolFromSmiles(datatable_noChange[i]['Ligand']),isomericSmiles=False)

def test_convert_singleRow_New_Replace_addHs():
    datatable, datatable_noChange = load_data()
    MolRdkitConverter().convert(datatable=datatable, columns=['Ligand'], createNewColumns=['Ligand_MOL'], overrideSmiles=True, AddHs=True, n_jobs=thread_count)

    size = len(datatable)
    for i in range(size):
        mol:Chem.Mol = datatable[i]['Ligand_MOL']
        smilesOfMol = Chem.MolToSmiles(mol, isomericSmiles=False)
        assert smilesOfMol == Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(datatable_noChange[i]['Ligand'])),isomericSmiles=False)

def test_convert_multiRow_noNew_noReplace():
    datatable, datatable_noChange = load_data()
    MolRdkitConverter().convert(datatable=datatable, columns=['Ligand', 'Additive'], n_jobs=thread_count)

    #print(X.to_String())

    size = len(datatable)
    for i in range(size):
        smiles_ligand = Chem.MolToSmiles(datatable[i]['Ligand'], isomericSmiles=False)
        smiles_add = Chem.MolToSmiles(datatable[i]['Additive'], isomericSmiles=False)
        assert smiles_ligand == Chem.MolToSmiles(Chem.MolFromSmiles(datatable_noChange[i]['Ligand']),isomericSmiles=False)
        assert smiles_add == Chem.MolToSmiles(Chem.MolFromSmiles(datatable_noChange[i]['Additive']), isomericSmiles=False)

def test_convert_multiRow_New_noReplace():
    datatable, datatable_noChange = load_data()
    MolRdkitConverter().convert(datatable=datatable, columns=['Ligand', 'Additive'], createNewColumns=['Ligand_MOL', 'Additive_MOL'], n_jobs=thread_count)

    size = len(datatable)
    for i in range(size):
        smiles_ligand = Chem.MolToSmiles(datatable[i]['Ligand_MOL'], isomericSmiles=False)
        smiles_add = Chem.MolToSmiles(datatable[i]['Additive_MOL'], isomericSmiles=False)
        assert smiles_ligand == Chem.MolToSmiles(Chem.MolFromSmiles(datatable_noChange[i]['Ligand']),isomericSmiles=False)
        assert smiles_add == Chem.MolToSmiles(Chem.MolFromSmiles(datatable_noChange[i]['Additive']), isomericSmiles=False)

def test_convert_multiRow_New_noReplace_addHs():
    datatable, datatable_noChange = load_data()
    MolRdkitConverter().convert(datatable=datatable, columns=['Ligand', 'Additive'], createNewColumns=['Ligand_MOL', 'Additive_MOL'],  isomericSmiles=False, AddHs=True, n_jobs=thread_count)

    size = len(datatable)
    for i in range(size):
        smiles_ligand = Chem.MolToSmiles(datatable[i]['Ligand_MOL'], isomericSmiles=False)
        smiles_add = Chem.MolToSmiles(datatable[i]['Additive_MOL'], isomericSmiles=False)
        assert smiles_ligand == Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(datatable_noChange[i]['Ligand'])),isomericSmiles=False)
        assert smiles_add == Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(datatable_noChange[i]['Additive'])),isomericSmiles=False)