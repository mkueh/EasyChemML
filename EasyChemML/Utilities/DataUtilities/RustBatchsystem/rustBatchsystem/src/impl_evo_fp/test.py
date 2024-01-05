from rdkit import Chem

smart = "[OH]c1ccccc1"
prim = Chem.MolFromSmarts(smart)
print(prim.GetNumAtoms())


