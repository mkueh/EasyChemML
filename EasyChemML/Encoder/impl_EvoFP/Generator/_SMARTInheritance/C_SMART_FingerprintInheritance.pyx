from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Pattern import SMART_pattern
from typing import List
import random


class C_SMART_FingerprintInheritance():

    bonds = [   #Bindungen ->  atoms + bonds + atoms
             '-',           #single bond (aliphatic)
             #'/',           #directional bond "up"
             #'\ '.strip(),  #directional bond "down"
             #'/?',          #directional bond "up or unspecified"
             #'\?',          #directional bond "down or unspecified"
             '=',           #double bond
             '#',           #triple bond
             ':',           #aromatic bond
             '~',           #any bond (wildcard)
             '@']           #any ring bond

    def __init__(self):
        pass

    def C_createMutationSlice(self, father: SMART_pattern, mother: SMART_pattern):

        atomics = []
        bonds = []

        try:
            father_start = random.randrange(0, len(father))
            mother_start = random.randrange(0, len(mother))
            father_end = random.randrange(father_start, len(father))
            mother_end = random.randrange(mother_start, len(mother))
        except Exception as e:
            return None

        # from which gender is the connection-bond that connect the two slices
        gender = random.random()
        #Fatherbond
        if gender >= 0.5 and father_end < len(father)-1:
            father_slice_atomics, father_slice_bonds = father.getgeneticSlice(father_start, father_end, True)
            mother_slice_atomics, mother_slice_bonds = mother.getgeneticSlice(mother_start, mother_end, False)
            atomics.extend(father_slice_atomics)
            atomics.extend(mother_slice_atomics)

            bonds.extend(father_slice_bonds)
            bonds.extend(mother_slice_bonds)
        #Motherbond
        elif mother_end < len(mother) - 1:
            mother_slice_atomics, mother_slice_bonds = mother.getgeneticSlice(mother_start, mother_end, True)
            father_slice_atomics, father_slice_bonds = father.getgeneticSlice(father_start, father_end, False)
            atomics.extend(mother_slice_atomics)
            atomics.extend(father_slice_atomics)

            bonds.extend(mother_slice_bonds)
            bonds.extend(father_slice_bonds)
        #a new bond when the slice is the full gen of the parents
        else:
            mother_slice_atomics, mother_slice_bonds = mother.getgeneticSlice(mother_start, mother_end, False)
            father_slice_atomics, father_slice_bonds = father.getgeneticSlice(father_start, father_end, False)
            tmp_bond = []
            tmp_bond.extend(mother.getBounds())
            tmp_bond.extend(father.getBounds())
            if len(tmp_bond) == 1 or len(tmp_bond) == 0 or max(len(father)-1, len(mother)-1) == len(tmp_bond):
                connectionBond = self.bonds[random.randrange(0, len(self.bonds))]
            else:
                connectionBond = tmp_bond[random.randrange(max(len(father)-1, len(mother)-1),len(tmp_bond))]

            gender = random.random()
            if gender >= 0.5:
                father_slice_bonds.append(connectionBond)
                atomics.extend(father_slice_atomics)
                atomics.extend(mother_slice_atomics)

                bonds.extend(father_slice_bonds)
                bonds.extend(mother_slice_bonds)
            else:
                mother_slice_bonds.append(connectionBond)
                atomics.extend(mother_slice_atomics)
                atomics.extend(father_slice_atomics)

                bonds.extend(mother_slice_bonds)
                bonds.extend(father_slice_bonds)

        return SMART_pattern(bonds, atomics, createInfo={'createTyp': 'MutationSlice', 'fatherPattern': father, 'motherPattern': mother})

    def C_FullyRandom_genMutation(self, father: SMART_pattern, mother: SMART_pattern):
        maxPossibleLength = len(father) + len(mother)
        length = random.randrange(2, maxPossibleLength)

        atomics = []
        bonds = []

        father_atomicCounter = 0
        mother_atomicCounter = 0
        father_bondsCounter = 0
        mother_bondsCounter = 0
        for i in range(length):
            # Atomic
            gender = random.random()
            if gender >= 0.5 and father_atomicCounter < len(father):
                atomics.append(father.getAtomicAt(father_atomicCounter))
                father_atomicCounter += 1
            elif mother_atomicCounter < len(mother):
                atomics.append(mother.getAtomicAt(mother_atomicCounter))
                mother_atomicCounter += 1
            else:
                atomics.append(father.getAtomicAt(father_atomicCounter))
                father_atomicCounter += 1

            # Bond
            if i < length - 1:
                gender = random.random()
                if gender >= 0.5 and father_bondsCounter < len(father)-1:
                    bonds.append(father.getBondAt(father_bondsCounter))
                    father_bondsCounter += 1
                elif mother_bondsCounter < len(mother)-1:
                    bonds.append(mother.getBondAt(mother_bondsCounter))
                    mother_bondsCounter += 1
                else:
                    atomics.append(father.getBondAt(father_atomicCounter))
                    father_bondsCounter += 1

        return SMART_pattern(bonds, atomics, createInfo={'createTyp': 'NormalReproduction', 'fatherPattern': father, 'motherPattern': mother})

