from deprecated.Encoder.EncoderInterface import EncoderInterface
from .FingerprintEncoder import FingerprintEncoder
from EasyChemML.Utilities.Application_env import Application_env

from typing import List

class MFF(EncoderInterface):

    def __init__(self, APP_ENV:Application_env):
        super().__init__(APP_ENV)

    """
    Input is raw SMILES!
    Output is the Fingerprints Vector

    Parameter
    FP_length:
    coulmns: if none than all
    """

    # @usage_monitoring
    def convert(self, dataset, columns:List[str], n_jobs: int, **kwargs):
        if not 'FP_length' in kwargs:
            print('MFFcode: FP_length is not set')

        g = FingerprintEncoder(self.APP_ENV)
        length = kwargs['FP_length']

        fingerprints = ['rdkit'] * 8
        fingerprints.extend(['morgan_circular'] * 8)
        fingerprints.extend(['layerdfingerprint'] * 4)
        fingerprints.extend(['avalon'])
        fingerprints.extend(['maccs'])
        fingerprints.extend(['atom_pairs'])
        fingerprints.extend(['topological_torsions'])

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

        args = {'fp_names': fingerprints, 'fp_settings': fingerprints_args}

        return g.convert(dataset, columns, n_jobs, **args)

    @staticmethod
    def getItemname():
        return "e_mff"
