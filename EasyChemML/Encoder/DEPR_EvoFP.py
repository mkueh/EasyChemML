
from EasyChemML.Encoder.AbstractEncoder.AbstractEncoder import AbstractEncoder
import logging, datetime, os, pickle, numpy as np, csv, io

from EasyChemML.Encoder.impl_EvoFP.EvoFingerCreator import EvoFingerCreator, Member
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Fingerprint import SMART_Fingerprint
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Pattern import SMART_pattern
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatyp, BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.ParallelUtilities.IndexQueues.IndexQueue_settings import IndexQueue_settings
from .impl_EvoFP.EVOFingerprint_Enum import FeatureTyp
from .impl_EvoFP.Fitnessfunction.Abstract_Fitnessfunction import Abstract_Fitnessfunction
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchAccess, BatchTable
from EasyChemML.Utilities.ParallelUtilities.ParallelHelper import ParallelHelper
from typing import List, Tuple
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList
from ..Environment import Environment


class EvoFP(AbstractEncoder):

    def __init__(self):
        super().__init__()

    @staticmethod
    def is_data_dependency():
        return True

    @staticmethod
    def is_parallel():
        return True

    @staticmethod
    def getItemname():
        return "e_evofp"

    def train(self, n_jobs: int, populationSize: int, newgen_rate: float,
              gen_recombinationrate: float, newpop_perStep: float, usebestforkids: float, keepbest_perStep: float,
              aging_rate: float,
              fp_size: int, evo_steps: int, fitfunc: Abstract_Fitnessfunction, feature_typ: FeatureTyp,
              fitfunc_worker: int, environment: Environment, train_data_path: str = 'CV_EVOFP'):
        logging.info('***************************************************************')
        logging.info(' [[[[ -----------------> LOADING EVO-FP <----------------- ]]]]')
        logging.info('***************************************************************')

        logging.info('!!! Start EVO-FP generation: !!!')
        logging.info('----------! Train EVO-FP for OuterCV: !----------')
        logging.info('Start: ' + str(datetime.datetime.now()))

        logging.info('Prepare DATASETS and load them into the RAM')
        logging.info('DATASETS loaded : ' + str(datetime.datetime.now()))

        workfolder = environment.WORKING_path
        workfolder = os.path.join(workfolder, train_data_path)
        if not os.path.exists(workfolder):
            os.mkdir(workfolder)

        SMART_FP: SMART_Fingerprint = self.__trainEVOFP(workfolder=workfolder, n_jobs=n_jobs,
                                                        populationSize=populationSize,
                                                        newgen_rate=newgen_rate,
                                                        gen_recombinationrate=gen_recombinationrate,
                                                        newpop_perStep=newpop_perStep, usebestforkids=usebestforkids,
                                                        keepbest_perStep=keepbest_perStep, fp_size=fp_size,
                                                        evo_steps=evo_steps, fitfunc=fitfunc, feature_typ=feature_typ,
                                                        fitfunc_worker=fitfunc_worker, aging_rate=aging_rate)

        logging.info('***************************************************************')
        logging.info(' [[[[ ---------------> EVO-FP training is finished <--------------- ]]]]')
        logging.info('Finish: ' + str(datetime.datetime.now()))
        logging.info('***************************************************************')

        return SMART_FP

    """
    Input is raw SMILES!
    Output is the impl_FingerprintEncoder dict

    Parameter
    FP_length:
    coulmns: if none than all
    """

    def convert(self, smart_fp: SMART_Fingerprint, feature_typ:FeatureTyp, datatable: BatchTable, columns: List[str], n_jobs: int):
        """

        Args:
            smart_fp: SMART_Fingerprint
            batchtable: A BatchTable contains Rdkit Mol objects
            columns: A list of columns
            n_jobs: job count

        Returns:
            converted Dataset
        """
        iterator: BatchAccess = iter(datatable)
        batch: np.ndarray
        dataTypHolder: BatchDatatypHolder = datatable.getDatatypes()

        for column in columns:
            dataTypHolder[column] = BatchDatatyp(BatchDatatypClass.NUMPY_INT8, (len(smart_fp),))

        for batch in iterator:
            out = dataTypHolder.createAEmptyNumpyArray(len(batch))
            shared_batch = Shared_PythonList(batch, datatable.getDatatypes())
            parallel_executer = ParallelHelper(n_jobs)
            IQ_settings = IndexQueue_settings(start_index=0, end_index=len(batch), chunksize=128)
            out = parallel_executer.execute_map_orderd_return(self._parallel_convert, IQ_settings, out.dtype,
                                                              input_arr=shared_batch, columns=columns,
                                                              smart_fp=smart_fp, feature_typ= feature_typ)

            iterator <<= out
            shared_batch.destroy()

    def _parallel_convert(self, input_arr: Shared_PythonList, columns: List[str], out_dtypes, current_chunk: int, feature_typ:FeatureTyp, smart_fp: SMART_Fingerprint):
        out_array = np.empty(shape=(len(current_chunk),), dtype=out_dtypes)
        index_counter = 0

        for current_index in current_chunk:
            raise_exception = False
            for col in columns:
                try:
                    out_array[index_counter][col] = smart_fp.getFingerprintMOL(input_arr[current_index][col], feature_typ)
                except Exception as e:
                    print(f'Data (row: {current_index}) can not translate in a fingerprint')
                    print('Exception : ' + str(e))
                    raise Exception('Data could not be converted')

            for exists_col in list(input_arr.getcolumns()):
                if exists_col not in columns:
                    out_array[index_counter][exists_col] = input_arr[current_index][exists_col]

            index_counter += 1
        return out_array

    def load_fingerprint_popFile(self, path:str, index:int) -> (SMART_Fingerprint, Tuple):
        """

        loads a SMART_Fingerprint from a created pop file.

        Args:
            path: path to the pop file
            index: index of the SMART fingerprint in the pop file. the popfile is sorted by the fitness value. For the best fingerprint use 0.
        Returns: Tuple of SMART_Fingerprint and his metric/fitnessvalue

        """
        inputfile = open(path, 'rb')
        data = inputfile.read()
        try:
            members: List[Member] = pickle.loads(data)
        except ModuleNotFoundError as e:
            members: List[Member] = self.__load_oldPOP_pickels(data)
        inputfile.close()
        return members[index].S_FP

    def __load_oldPOP_pickels(self, path:str) -> List[Member]:
        class RenameUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                renamed_module = module
                if module == "Encoder.impl_EvoFP.EvoFingerCreator":
                    renamed_module = "EasyChemML.Encoder.impl_EvoFP.EvoFingerCreator"
                elif module == 'Encoder.impl_EvoFP.Utilities.SMART_Fingerprint':
                    renamed_module = "EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Fingerprint"
                elif module == 'Encoder.impl_EvoFP.Utilities.SMART_Pattern':
                    renamed_module = "EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Pattern"

                return super(RenameUnpickler, self).find_class(renamed_module, name)

        def renamed_load(file_obj):
            return RenameUnpickler(file_obj).load()

        def renamed_loads(pickled_bytes):
            file_obj = io.BytesIO(pickled_bytes)
            return renamed_load(file_obj)

        return renamed_loads(path)

    def load_fingerprint(self, path: str) -> SMART_Fingerprint:
        """

        loads a saved SMART_Fingerprint.

        Args:
            path: path to saved SMART_Fingerprint

        Returns: a SMART_Fingerprint object

        """
        return SMART_Fingerprint.load(path)

    def convert_CSV2Fingerprint(self, path: str) -> SMART_Fingerprint:
        """

        loads and converts a csv file into a SMART Fingerprint.
        The SMARTs pattern must be stored in the first column, one below the other, without the header row.

        Args:
            path: path to csv file

        Returns: a SMART_Fingerprint object

        """
        SMART_patterns = []
        with open(path, newline='') as csvfile:
            patternreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in patternreader:
                SMART_patterns.append(SMART_pattern(bonds=[], atomics=row, createInfo={'info': 'loaded'}))

        return SMART_Fingerprint(SMART_patterns)


    def __trainEVOFP(self, workfolder: str, n_jobs: int,
                     populationSize: int, newgen_rate: float,
                     gen_recombinationrate: float, newpop_perStep: float, usebestforkids: float, keepbest_perStep: float,
                     aging_rate: float,
                     fp_size: int, evo_steps: int, fitfunc: Abstract_Fitnessfunction, feature_typ: FeatureTyp,
                     fitfunc_worker: bool):

        EVO_creator = EvoFingerCreator()
        SMART_FP: SMART_Fingerprint = EVO_creator.create(workfolder=workfolder, n_jobs=n_jobs,
                                                         populationSize=populationSize,
                                                         newgen_rate=newgen_rate,
                                                         gen_recombinationrate=gen_recombinationrate,
                                                         newpop_perStep=newpop_perStep, usebestforkids=usebestforkids,
                                                         keepbest_perStep=keepbest_perStep, fp_size=fp_size,
                                                         evo_steps=evo_steps, fitfunc=fitfunc, feature_typ=feature_typ,
                                                         fitfunc_worker=fitfunc_worker, aging_rate=aging_rate)
        return SMART_FP

    def __createFeaturesubset(self, inputMols, SMART_FP: SMART_Fingerprint, bitfeature, columns: List[str], n_jobs):
        logging.info('Generate all fingerprints of the SMILES-dataset')
        logging.info('Time: ' + str(datetime.datetime.now()))
        width = len(SMART_FP)

        iterator: BatchAccess = iter(inputMols)
        batch: np.ndarray
        dataTypHolder: BatchDatatypHolder = inputMols.getDatatypes()

        for column in columns:
            if not bitfeature:
                dataTypHolder[column] = BatchDatatyp(BatchDatatypClass.NUMPY_INT32, (width,))
            else:
                dataTypHolder[column] = BatchDatatyp(BatchDatatypClass.NUMPY_INT8, (width,))

        for batch in iterator:
            shared_batch = Shared_PythonList(batch, inputMols.getDatatypes())
            out = dataTypHolder.createAEmptyNumpyArray(len(batch))

            parallel_executer = ParallelHelper(n_jobs)
            IQ_settings = IndexQueue_settings(start_index=0, end_index=len(batch), chunksize=16)
            out = parallel_executer.execute_map_orderd_return(self.__createFPvector, IQ_settings, out.dtype,
                                                              columns=columns, smart=SMART_FP, input_arr=shared_batch,
                                                              bitfeature=bitfeature)
            shared_batch.destroy()
            iterator <<= out

        return None

    def __createFPvector(self, input_arr, smart: SMART_Fingerprint, bitfeature, columns: List[str], out_dtypes,
                         current_chunk: int):
        out_array = np.empty(shape=(len(current_chunk),), dtype=out_dtypes)
        index_counter = 0
        for current_index in current_chunk:
            for i, mol in enumerate(input_arr[current_index]):
                out_array[index_counter][i][:] = smart.getFingerprintMOL(mol, bitfeature)

            for exists_col in list(input_arr.getcolumns()):
                if exists_col not in columns:
                    out_array[index_counter][exists_col] = input_arr[current_index][exists_col]

            index_counter += 1
        return out_array

    @staticmethod
    def convert_foreach_outersplit():
        return True

    """    def createFittingPlot(self):
            self.__createFittingPlot()"""

    """def __createFittingPlot(self, fullDataset: Dataset, splitt, workingdir, bitfeature, fitfunc_param, n_jobs):
            failed = False
            if not 'metricscore' in fitfunc_param:
                print('EvoFP-CreateFittingPlot: Parameter metricscore is not set')
                failed = True
            if not 'model' in fitfunc_param:
                print('EvoFP-CreateFittingPlot: Parameter model is not set')
                failed = True
            if not 'model_args' in fitfunc_param:
                print('EvoFP-CreateFittingPlot: Parameter model_args is not set')
                failed = True

            if failed:
                print('error during reading settings for createFittingPlot')
                print('################### skip ##########################')
                return

            print(f'Create Plots')
            evo_steps = [os.path.abspath(f) for f in glob.glob(os.path.join(workingdir, '*.pop'), recursive=True)]
            scores = []

            for step in evo_steps:
                print(f'Plot no. {step}')
                saved_evo = open(step, 'rb')
                population = pickle.load(saved_evo)
                saved_evo.close()
                member: Member = population[0]
                smartFP = member.S_FP

                fitfunc_param['fix_splitt'] = [(splitt.train, splitt.test)]
                mWF = ModelWrapper_Fitness(None, bitfeature, self.APP_ENV, **fitfunc_param)

                metricScore = mWF.calc_fitness(smartFP, fullDataset.getFeature_data(), fullDataset.getTargets_data(),
                                               workingdir)
                scores.append((step, metricScore[0]))

            perfomance_csv = os.path.join(workingdir, 'EVO_Performance' + '.csv')

            with open(perfomance_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                writer.writerow(['File:', 'Metric_Score'])

                for item in scores:
                    writer.writerow(item)"""
