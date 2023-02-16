import numpy as np

from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Encoder.FingerprintEncoder import FingerprintTyp, FingerprintHolder, FingerprintEncoder
from EasyChemML.Encoder.impl_FingerprintEncoder.FingerprintGenerator import FingerprintGenerator_Mode, \
    FingerprintGenerator
from EasyChemML.Encoder.impl_RdkitConverter.MolRdkitConverter import MolRdkitConverter
from EasyChemML.Environment import Environment

thread_count = 12


def load_data():
    env = Environment()

    load_dataset = {}
    load_dataset['doyle'] = XLSX('../../_TestDataset/DreherDoyle.xlsx', sheet_name='FullCV_01',
                                 columns=['Ligand', 'Additive', 'Base', 'Aryl halide', 'Output'], range=[0, 500])
    load_dataset['doyle_noChange'] = XLSX('../../_TestDataset/DreherDoyle.xlsx',
                                          sheet_name='FullCV_01',
                                          columns=['Ligand', 'Additive', 'Base', 'Aryl halide', 'Output'],
                                          range=[0, 500])
    di = DataImporter(env)
    dh = di.load_data_InNewBatchPartition(load_dataset)

    molconvert = MolRdkitConverter()
    molconvert.convert(datatable=dh['doyle'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=12)
    molconvert.convert(datatable=dh['doyle_noChange'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=12)

    return dh['doyle'], dh['doyle_noChange']


def test_ECFC4_count():
    dataset, dataset_unchange = load_data()

    define_ECFC4_setting = FingerprintHolder(FingerprintTyp.ECFC, {'length': 32, 'radius': 4})

    encoder = FingerprintEncoder()
    encoder.convert(dataset, columns=['Ligand'], n_jobs=12, fingerprints=[define_ECFC4_setting], default_NoneToke=0,
                    return_nonZero_indices=True)

    fp_names = []
    fp_settings = []

    for fingerprint in [define_ECFC4_setting]:
        fp_names.append(fingerprint.fingerprint_typ.value)
        fp_settings.append(fingerprint.fingerprint_settings)
    FG = FingerprintGenerator(mode=FingerprintGenerator_Mode.NONZERO_INDICES, default_NoneToke=0)

    for i, datapoint in enumerate(dataset[:]):
        assert list(datapoint['Ligand']) == \
               FG.generateArrOfFingerprints([dataset_unchange[i]['Ligand']], fp_names, fp_settings)[0].tolist()


def test_ECFP4_count():
    dataset, dataset_unchange = load_data()

    define_ECFP4_setting = FingerprintHolder(FingerprintTyp.ECFP, {'length': 128, 'radius': 4})

    encoder = FingerprintEncoder()
    encoder.convert(dataset, columns=['Ligand'], n_jobs=12, fingerprints=[define_ECFP4_setting],
                    return_nonZero_indices=True, offsetForNonZeroIndices=100, default_NoneToke=0)

    fp_names = []
    fp_settings = []

    for fingerprint in [define_ECFP4_setting]:
        fp_names.append(fingerprint.fingerprint_typ.value)
        fp_settings.append(fingerprint.fingerprint_settings)
    FG = FingerprintGenerator(mode=FingerprintGenerator_Mode.NONZERO_INDICES, offsetForNonZeroIndices=100, default_NoneToke=0)

    print(dataset.to_String())

    for i, datapoint in enumerate(dataset[:]):
        assert list(datapoint['Ligand']) == \
               FG.generateArrOfFingerprints([dataset_unchange[i]['Ligand']], fp_names, fp_settings)[0].tolist()


def test_ECFP4_count_newCol():
    dataset, dataset_unchange = load_data()

    define_ECFP4_setting = FingerprintHolder(FingerprintTyp.ECFP, {'length': 128, 'radius': 4})

    encoder = FingerprintEncoder()
    encoder.convert(dataset, columns=['Ligand'], n_jobs=12, fingerprints=[define_ECFP4_setting], default_NoneToke=-1,
                    createNewColumns=['Ligand_FP'])

    fp_names = []
    fp_settings = []

    for fingerprint in [define_ECFP4_setting]:
        fp_names.append(fingerprint.fingerprint_typ.value)
        fp_settings.append(fingerprint.fingerprint_settings)
    FG = FingerprintGenerator(mode=FingerprintGenerator_Mode.PLAIN_FP, default_NoneToke=0)

    for i, datapoint in enumerate(dataset[:]):
        assert list(datapoint['Ligand_FP']) == \
               FG.generateArrOfFingerprints([dataset_unchange[i]['Ligand']], fp_names, fp_settings)[0].tolist()
