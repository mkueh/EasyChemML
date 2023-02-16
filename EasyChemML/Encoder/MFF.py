from .FingerprintEncoder import FingerprintEncoder, FingerprintTyp, FingerprintHolder

from typing import List

from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable


class MFF():

    def __init__(self):
        pass

    def fit(self, datatable, columns: List[str], n_jobs: int, **kwargs):
        return

    """
    Input is raw SMILES!
    Output is the Fingerprints Vector

    Parameter
    FP_length:
    coulmns: if none than all
    """

    # @usage_monitoring
    def convert(self, datatable: BatchTable, columns: List[str], n_jobs: int, fp_length: int,
                ignore_errors: bool = False,
                return_nonZero_indices: bool = False):
        g = FingerprintEncoder()
        length = fp_length

        fingerprint_names = [FingerprintTyp.RDKit] * 8
        fingerprint_names.extend([FingerprintTyp.ECFP] * 8)
        fingerprint_names.extend([FingerprintTyp.LAYERDFINGERPRINT] * 4)
        fingerprint_names.extend([FingerprintTyp.AVALON])
        fingerprint_names.extend([FingerprintTyp.MACCS])
        fingerprint_names.extend([FingerprintTyp.ATOM_PAIRS])
        fingerprint_names.extend([FingerprintTyp.TOPOLOGICAL_TORSIONS])

        # RDKIT
        fingerprints_args = [{'length': length, 'maxPath': 2}, {'length': length, 'maxPath': 4},
                             {'length': length, 'maxPath': 6}, {'length': length, 'maxPath': 8}]
        # RDKITlinear
        fingerprints_args.extend([{'length': length, 'maxPath': 2, 'branchedPaths': False},
                                  {'length': length, 'maxPath': 4, 'branchedPaths': False},
                                  {'length': length, 'maxPath': 6, 'branchedPaths': False},
                                  {'length': length, 'maxPath': 8, 'branchedPaths': False}])
        # MorganCircle
        fingerprints_args.extend(
            [{'length': length, 'radius': 0}, {'length': length, 'radius': 2}, {'length': length, 'radius': 4},
             {'length': length, 'radius': 6}])
        # MorganeCircle Feature
        fingerprints_args.extend(
            [{'length': length, 'radius': 0, 'useFeatures': True}, {'length': length, 'radius': 2, 'useFeatures': True},
             {'length': length, 'radius': 4, 'useFeatures': True},
             {'length': length, 'radius': 6, 'useFeatures': True}])
        # layerdfingerprint
        fingerprints_args.extend(
            [{'length': length, 'maxPath': 2}, {'length': length, 'maxPath': 4}, {'length': length, 'maxPath': 6},
             {'length': length, 'maxPath': 8}])
        # Avalon
        fingerprints_args.extend([{'length': length}])

        # maccs
        fingerprints_args.extend([{}])

        # atom_pairs
        fingerprints_args.extend([{'length': length}])

        # topological_torsions
        fingerprints_args.extend([{'length': length}])

        fingerprints = []
        for i, fingerprintTyp in enumerate(fingerprint_names):
            fingerprints.append(FingerprintHolder(fingerprintTyp, fingerprints_args[i]))

        return g.convert(datatable, columns, n_jobs, fingerprints=fingerprints, ignore_errors=ignore_errors,
                         return_nonZero_indices=return_nonZero_indices)

    @staticmethod
    def convert_foreach_outersplit():
        return False

    @staticmethod
    def is_parallel():
        return True
