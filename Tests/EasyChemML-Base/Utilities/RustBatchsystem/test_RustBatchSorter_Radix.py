from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Encoder import MolRdkitConverter
from EasyChemML.Encoder.FingerprintEncoder import FingerprintHolder, FingerprintEncoder, FingerprintTyp
from EasyChemML.Environment import Environment
from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyWrapper.RustBatchholder import RustBatchholder

def test_loadBack():
    env = Environment()

    load_dataset = {}
    load_dataset['doyle'] = XLSX('../../../_TestDataset/DreherDoyle.xlsx', sheet_name='FullCV_01',
                                 columns=['Ligand', 'Additive', 'Base', 'Aryl halide', 'Output'], range=[0, 500])

    di = DataImporter(env)
    dh = di.load_data_InNewBatchPartition(load_dataset)

    molconvert = MolRdkitConverter()
    molconvert.convert(datatable=dh['doyle'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=12)

    define_ECFC4_setting = FingerprintHolder(FingerprintTyp.ECFC, {'length': 32, 'radius': 4})

    encoder = FingerprintEncoder()
    encoder.convert(dh['doyle'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=12,
                    fingerprints=[define_ECFC4_setting], default_NoneToke=0,
                    return_nonZero_indices=True)

    print(dh['doyle'])

    rb = RustBatchholder('./tmp', 100000)
    rb.transferToRust(dh, 'doyle')

    rust_table = rb.getRustBatchTable('doyle')
    rb.transferToBatchtable('doyle',dh, 'doyle_new')

    rb.clean()