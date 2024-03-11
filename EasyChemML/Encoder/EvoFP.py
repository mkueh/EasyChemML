from typing import List

from EasyChemML.Encoder.AbstractEncoder.AbstractEncoder import AbstractEncoder

from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Fingerprint import SMART_Fingerprint
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Pattern import SMART_pattern

from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable
from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyWrapper.RustBatchholder import RustBatchholder
from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyRustBatchsystem.evo_fingerprint import PyEvolutionConfig
from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyRustBatchsystem.evo_fingerprint import PyFitnessFunctionConfig
from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyRustBatchsystem.evo_fingerprint import PySmartsFingerprint
from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyRustBatchsystem.evo_fingerprint import PySmartsPattern


class EvoFP(AbstractEncoder):

    def __init__(self):
        super().__init__()

    @staticmethod
    def is_data_dependency():
        return True

    @staticmethod
    def is_parallel():
        return False

    @staticmethod
    def getItemName():
        return "e_evofp"

    @staticmethod
    def train(evo_config: PyEvolutionConfig, fitness_config: PyFitnessFunctionConfig, rust_batch_holder: RustBatchholder,
              feature_table_name: str, target_table_name: str, is_regression: bool):
        """
        This method is used to train a fingerprint by accessing the Python interface of the EVO-Fingerprint-Encoder.

        Args:
            evo_config (PyEvolutionConfig): The evolution configuration.
            fitness_config (PyFitnessConfig): The fitness function configuration.
            rust_batch_holder (RustBatchholder): The batch holder containing the SMILES data.
            feature_table_name (str): The name of the table containing the feature data.
            target_table_name (str): The name of the table containing the target data.
            is_regression (bool): Whether the task is a regression task.

        Returns:
            SMART_Fingerprint: The trained fingerprint.
            fingerprint_metrics: The metrics of the trained fingerprint.
        """

        feature_batch_table = rust_batch_holder.getRustBatchTable(feature_table_name)
        target_batch_table = rust_batch_holder.getRustBatchTable(target_table_name)

        rust_fingerprint: PySmartsFingerprint = evo_fingerprint.train_the_one_fingerprint(
            py_evolution_config=evo_config,
            py_fitness_function_config=fitness_config,
            smiles_batch_table=feature_batch_table,
            target_batch_table_f64=target_batch_table,
            is_regression=is_regression,
        )

        pattern_array = []
        for pattern in rust_fingerprint.patterns:
            python_pattern: SMART_pattern = SMART_pattern(pattern.bonds, pattern.atomics, pattern.createInfo)
            pattern_array.append(python_pattern)

        return SMART_Fingerprint(pattern_array), rust_fingerprint.metrics

    def convert(self, rust_batch_holder: RustBatchholder, fingerprint: SMART_Fingerprint, table_name: str,
                bit_feature: bool = False, datatable: BatchTable = None, columns: List[str] = None, n_jobs: int = 1, ):
        """
        This method is used to convert the input data into molecular fingerprints.

        Args:
            rust_batch_holder (RustBatchholder): The batch holder containing the SMILES data.
            fingerprint (SMART_Fingerprint): The fingerprint to be used for conversion.
            bit_feature (bool): Whether to use bit features.
            datatable (BatchTable): The batch table containing the input data.
            columns (List[str]): The columns to be converted.
            n_jobs (int): The number of jobs to be used.
            table_name (str): The name of the table containing the SMILES data.

        Returns:
            converted Dataset
        """
        batch_table = rust_batch_holder.getRustBatchTable(table_name)
        py_pattern_array = []
        for pattern in fingerprint.pattern:
            py_pattern: PySmartsPattern = PySmartsPattern(pattern.getBounds(), pattern.getAtomics(), pattern.createInfo)
            py_pattern_array.append(py_pattern)
        py_fingerprint = PySmartsFingerprint(py_pattern_array)

        feature_data = evo_fingerprint.convert(smiles_batch_table=batch_table, py_fingerprint=py_fingerprint,
                                               bool_matching=bit_feature)

        return feature_data
