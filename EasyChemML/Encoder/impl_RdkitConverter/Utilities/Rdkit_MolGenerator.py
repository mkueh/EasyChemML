from rdkit import Chem
from rdkit.Chem import Mol

from typing import List, Union, Tuple
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList
import math, numpy as np


class MolGenerator:

    @staticmethod
    def translateSMILES(SMILES: str, AddHs: bool = False, isomericSmiles=False) -> (Mol, str):
        out_mol = None
        out_smiles = None

        try:
            out_mol: Mol = Chem.MolFromSmiles(SMILES)

            if AddHs:
                out_mol = Chem.AddHs(out_mol)

            out_smiles: str = Chem.MolToSmiles(out_mol, isomericSmiles)
            out_mol = Chem.MolFromSmiles(out_smiles)

            # secound time because FromSmiles remove H
            if AddHs:
                out_mol = Chem.AddHs(out_mol)

        except Exception as e:
            print(e)

        return out_mol, out_smiles
