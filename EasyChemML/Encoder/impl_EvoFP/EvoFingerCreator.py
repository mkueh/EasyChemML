import copy

from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder, BatchDatatyp
from .Fitnessfunction.Abstract_Fitnessfunction import Abstract_Fitnessfunction
from .Generator.SMART_FingerprintGenerator import SMARTFingerprintGenerator
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Fingerprint import SMART_Fingerprint
from .Generator.SMART_FingerprintInheritance import SMART_FingerprintInheritance
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList

from .Fitnessfunction.IsFullRelevant_Fitness import IsFullRelevant_Fitness
from .Fitnessfunction.ModelWrapper_Fitness import ModelWrapper_Fitness
from .Fitnessfunction.MultiDataset_Fitness import MultiDataset_Fitness

from EasyChemML.Utilities.ParallelUtilities.ParallelHelper import ParallelHelper
from EasyChemML.Utilities.ParallelUtilities.IndexQueues.IndexQueue_settings import IndexQueue_settings
import logging, datetime, os, pandas, pickle, numpy as np, math, glob, random
from time import perf_counter
from typing import List, Dict, Union

from .EVOFingerprint_Enum import FeatureTyp
from ...Utilities.SharedDataset import SharedDataset


class Member(object):
    S_FP: SMART_Fingerprint = -1
    metric = -1

    def __init__(self, S_FP):
        self.S_FP = S_FP


class EvoFingerCreator(object):
    feature_typ: FeatureTyp = False
    fitfunc_multidataset_mode = False
    fitfunc_worker = 0
    fitfunc: Abstract_Fitnessfunction

    _SMARTG: SMARTFingerprintGenerator

    def __init__(self):
        random.seed(1337)

    def getfitfunc(self, name: str):
        if name == 'model':
            return ModelWrapper_Fitness
        elif name == 'diversity':
            return IsFullRelevant_Fitness
        elif name == 'multidataset_model':
            return MultiDataset_Fitness

    def create(self, workfolder: str, n_jobs: int, populationSize: int, newgen_rate: float,
               gen_recombinationrate: float, newpop_perStep: float, usebestforkids: float, keepbest_perStep: float,
               fp_size: int, evo_steps: int,
               fitfunc: Abstract_Fitnessfunction, feature_typ: FeatureTyp, aging_rate: float, fitfunc_worker: int):

        start = self._starttimer('Create Fitnessfuction')
        self.fitfunc_worker = fitfunc_worker
        self.feature_typ = feature_typ
        self.fitfunc = fitfunc
        dataset = fitfunc.get_datasets()

        if not isinstance(fitfunc.get_datasets(), list):
            self._fitfunc_multidataset_mode = False
            self._SMARTG = SMARTFingerprintGenerator(dataset, feature_typ, self._fitfunc_multidataset_mode)
        else:
            self._fitfunc_multidataset_mode = True
            self._SMARTG = SMARTFingerprintGenerator(dataset, feature_typ,
                                                     self._fitfunc_multidataset_mode)
        self._stoptimer(start)

        print('-----------------------------------------------------------------')
        print('search for old pops to be continued')
        found, population, index = self._searchForOldPoP(workfolder)
        if index >= evo_steps:
            print(f'found a pop that is higher pop:{index} than the current evo_steps {evo_steps}')
        if not found:
            print('No old pops were found')
            start = self._starttimer('Generate first population')
            population = self._generateNewPopulation(populationSize, fp_size, feature_typ, n_jobs)
            self._stoptimer(start)
        print('-----------------------------------------------------------------')

        if index == 0:
            start = self._starttimer('Calc fitness of Population')
            self._generatePopulation_Metrics(population, fitfunc_worker, workfolder, 0)
            self._stoptimer(start)

            start = self._starttimer('save Population')
            best_pops = self._getXbest(1, population, aging_rate, feature_typ, n_jobs)
            self._saveMemebersOFstep(population, workfolder, 0)
            self._stoptimer(start)

            logging.info('best MetricScore: ' + str(best_pops[0].metric[0]))
            print('best MetricScore: ' + str(best_pops[0].metric[0]))
            print('best Fingerprint id: ' + str(best_pops[0].S_FP.id()))
            logging.info('Finish: ' + str(datetime.datetime.now()))

        index += 1
        for i in range(index, evo_steps + 1):
            population = self._evo_step(i, population, gen_recombinationrate, newpop_perStep, usebestforkids,
                                        keepbest_perStep, fp_size, newgen_rate, aging_rate, dataset, n_jobs,
                                        feature_typ, workfolder, fitfunc_worker)

        return population[0].S_FP

    def _evo_step(self, iteration: int, population: List[Member], gen_recombinationrate,
                  newpop_perStep, usebestforkids, keepbest_perStep, fp_size, newgen_rate, aging_rate,
                  dataset: Union[SharedDataset, List[SharedDataset]], n_jobs,
                  feature_typ: FeatureTyp, outputPath, fitfunc_worker):
        logging.info('-----------------------------------------------------------------')
        logging.info('##########  EVO-STEP: ' + str(iteration) + '  ##########')

        start = self._starttimer('Generate evolutionStep')
        population = self._next_evolutionStep(population, newpop_perStep, newgen_rate, aging_rate,
                                              gen_recombinationrate,
                                              usebestforkids, keepbest_perStep, fp_size, train_dataset=dataset,
                                              feature_typ=feature_typ,
                                              n_jobs=n_jobs)
        self._stoptimer(start)

        start = self._starttimer('Generate new population metrics')
        self._generatePopulation_Metrics(population, fitfunc_worker, outputPath, iteration)
        self._stoptimer(start)

        start = self._starttimer('save Population')
        best_pops = self._getXbest(1, population, aging_rate, feature_typ, n_jobs)
        self._saveMemebersOFstep(population, outputPath, iteration)
        self._stoptimer(start)

        logging.info('best MetricScore: ' + str(best_pops[0].metric[0]))
        print('best MetricScore: ' + str(best_pops[0].metric[0]))
        print('best Fingerprint id: ' + str(best_pops[0].S_FP.id()))
        logging.info('Finish: ' + str(datetime.datetime.now()))

        return population

    def _searchForOldPoP(self, path: str) -> (bool, List[Member]):
        search_query = os.path.join(path, '*.pop')
        pop_files = glob.glob(search_query)

        if len(pop_files) == 0:
            return False, [], 0

        newest = (-1, '')
        for file in pop_files:
            remove_front = file.split('_')[-1]
            remove_extention = remove_front[:-4]
            asNumber = int(remove_extention)

            if newest[0] < asNumber:
                newest = (asNumber, file)

        print(f'find a old pop with index {newest[0]}')
        inputfile = open(newest[1], 'rb')
        data = inputfile.read()
        members: List[Member] = pickle.loads(data)
        inputfile.close()
        return True, members, newest[0]

    def _saveMemebersOFstep(self, population, workfolder, step):
        # CSV Datei
        stepcsv = os.path.join(workfolder, 'EVOstep_' + str(step) + '.csv')

        DataContainer = {}
        for i, p in enumerate(population):
            filename = 'FP_' + str(i)
            DataContainer[filename] = p.S_FP.getPatternsAsString()
            DataContainer[filename].append(p.metric[0])

        try:
            dataFrame_results = pandas.DataFrame(DataContainer)
            dataFrame_results.to_csv(stepcsv, header=True, index=False)
        except:
            # TODO stoped her
            for i, p in enumerate(population):
                filename = 'FP_' + str(i)
                print(DataContainer[filename])


        if isinstance(self.fitfunc, MultiDataset_Fitness):
            self._generatePopulation_Metrics_csvMultiDataset(population, workfolder, step)

        # Pickel-Object
        stepcsv = os.path.join(workfolder, 'EVOstep_' + str(step) + '.pop')
        outfile = open(stepcsv, 'wb')
        pickle.dump(population, outfile)
        outfile.close()

    def _next_evolutionStep(self, population, newpop_perStep, newgen_rate, aging_rate, gen_recombinationrate,
                            usebestforkids,
                            keepbest_perSte, fp_size, train_dataset: Union[SharedDataset, List[SharedDataset]],
                            feature_typ: FeatureTyp,
                            n_jobs: int = 1) -> []:
        keepN_best = int(len(population) * keepbest_perSte)
        n_newPops = int(len(population) * newpop_perStep)
        n_kids = int(len(population) - keepN_best - n_newPops)

        next_population = []

        start = self._starttimer('Generate new population', '[EVO-STEP]')
        new_pops = self._generateNewPopulation(n=n_newPops, fp_size=fp_size, bit_feature=feature_typ, n_jobs=n_jobs)

        self._stoptimer(start)

        start = self._starttimer('Get ' + str(keepN_best) + ' pops of last population', '[EVO-STEP]')
        best_pops = self._getXbest(keepN_best, population, aging_rate, feature_typ, n_jobs)
        self._stoptimer(start)

        n_bestForKids = self._getXbest(int(len(population) * usebestforkids), population, aging_rate,
                                       feature_typ, n_jobs)
        start = self._starttimer('generate ' + str(n_kids) + ' kids of last population', '[EVO-STEP]')
        kids = self._crossoverAndMutation(n_bestForKids, n_kids, newgen_rate, gen_recombinationrate, train_dataset,
                                          feature_typ,
                                          n_jobs=n_jobs)
        self._stoptimer(start)

        next_population.extend(best_pops)
        next_population.extend(new_pops)
        next_population.extend(kids)

        del population
        self._printS('Next Population is generated', '[EVO-STEP]')
        return next_population

    def _crossoverAndMutation(self, best: List[Member], n_kids: int, newgen_rate, gen_recombinationrate,
                              train_dataset: Union[SharedDataset, List[SharedDataset]],
                              feature_typ: FeatureTyp, n_jobs: int):

        fp_size = len(best[0].S_FP)
        count_generated_patter = n_kids * fp_size

        fp_settings_dataTypHolder = BatchDatatypHolder()
        fp_settings_dataTypHolder['father'] = BatchDatatyp(BatchDatatypClass.PYTHON_OBJECT)
        fp_settings_dataTypHolder['mother'] = BatchDatatyp(BatchDatatypClass.PYTHON_OBJECT)
        fp_settings = fp_settings_dataTypHolder.createAEmptyNumpyArray(n_kids)

        best_len = len(best)
        for fp in fp_settings:
            father_index = random.randrange(best_len)
            father = best[father_index].S_FP

            mother_index = random.randrange(best_len)
            if mother_index == father_index:
                if mother_index == best_len - 1:
                    mother_index -= 1
                else:
                    mother_index += 1

            mother = best[mother_index].S_FP
            father.shuffelpattern()
            mother.shuffelpattern()
            fp['father'] = father
            fp['mother'] = mother

        # List of IDs
        pattern_ids_dataTypHolder = BatchDatatypHolder()
        pattern_ids_dataTypHolder['ID'] = BatchDatatyp(BatchDatatypClass.NUMPY_STRING)
        pattern_ids = pattern_ids_dataTypHolder.createAEmptyNumpyArray(size=count_generated_patter)

        shared_pattern_ids = Shared_PythonList(pattern_ids, pattern_ids_dataTypHolder)
        shared_fp_settings = Shared_PythonList(fp_settings, fp_settings_dataTypHolder)

        smart_inher = SMART_FingerprintInheritance(self._SMARTG, feature_typ, self.fitfunc_multidataset_mode)
        IQ_settings = IndexQueue_settings(start_index=0, end_index=count_generated_patter, chunksize=1)
        parallelExecuter = ParallelHelper(n_jobs)
        return_list = parallelExecuter.execute_map_orderd_return(self._parallel_crossoverAndMutation, IQ_settings,
                                                                 np.dtype('O'),
                                                                 feature_data=train_dataset,
                                                                 newgen_rate=newgen_rate,
                                                                 gen_recombinationrate=gen_recombinationrate,
                                                                 fp_settings=shared_fp_settings,
                                                                 pattern_ids=shared_pattern_ids,
                                                                 fp_size=fp_size, smart_inher=smart_inher)

        gen_list = []
        c_MutationSlicer_tries = 0
        c_generateNewGen_tries = 0
        c_genRandomizer_tries = 0

        c_MutationSlicer_sucess = 0
        c_generateNewGen_sucess = 0
        c_genRandomizer_sucess = 0
        for item in return_list:
            gen_list.append(item[0])
            c_MutationSlicer_tries += item[1]
            c_generateNewGen_tries += item[2]
            c_genRandomizer_tries += item[3]

            if item[4] == 1:
                c_MutationSlicer_sucess += 1
            elif item[4] == 3:
                c_generateNewGen_sucess += 1
            elif item[4] == 2:
                c_genRandomizer_sucess += 1
            else:
                print('failed Score')

        member_list: List[Member] = []
        for i in range(n_kids):
            start = i * fp_size
            end = (i + 1) * fp_size

            tmp_member = Member(self._SMARTG.create_SMARTfp(list(gen_list[start:end])))
            member_list.append(tmp_member)

        print(f'generated {len(member_list)} kids | {count_generated_patter} pattern')
        print(f'MutationSlicer tries: {c_MutationSlicer_tries} | max_tries is 100')
        print(f'GenerateNewGen tries: {c_generateNewGen_tries} | has no max value')
        print(f'GenRandomizer tries: {c_genRandomizer_tries}')
        print('')
        print(f'MutationSlicer sucess: {c_MutationSlicer_sucess}')
        print(f'GenerateNewGen sucess: {c_generateNewGen_sucess}')
        print(f'GenRandomizer sucess: {c_genRandomizer_sucess}')

        return member_list

    def _parallel_crossoverAndMutation(self, newgen_rate, gen_recombinationrate, feature_data:Union[SharedDataset, List[SharedDataset]],
                                       fp_settings: Shared_PythonList, pattern_ids: Shared_PythonList, out_dtypes,
                                       current_chunk: List[int],
                                       fp_size: int, smart_inher: SMART_FingerprintInheritance):
        out_array = np.empty(shape=(len(current_chunk),), dtype=out_dtypes)
        out_pointer = 0

        # Metriken einbauen
        for index in current_chunk:
            # Index calculation
            kid_index = math.floor(float(index) / fp_size)
            gen_index = index - (kid_index * fp_size)
            father = fp_settings[kid_index]['father']
            mother = fp_settings[kid_index]['mother']
            out_array[out_pointer] = smart_inher.createInheritance_Pop(father, mother, newgen_rate,
                                                                       gen_recombinationrate, feature_data, gen_index,
                                                                       pattern_ids, index)
            out_pointer += 1
        return out_array

    def _getXbest(self, n: int, population: List[Member], aging_rate: float, bit_feature: FeatureTyp, n_jobs: int = 1):
        if aging_rate == 0:
            population.sort(key=lambda x: x.metric[0], reverse=True)
            return population[0:n]
        else:
            population.sort(key=lambda x: x.metric[0], reverse=True)
            best_n_Member: List[Member] = population[0:n]

            generate_indices = []
            for i, member in enumerate(best_n_Member):
                S_FP = member.S_FP

                for x, pattern in enumerate(S_FP):
                    aging_value = random.random()
                    if aging_value <= aging_rate:
                        generate_indices.append((i, x))

            if len(generate_indices) > 0:
                new_population = self._SMARTG.generate_NewSMARTfps(len(generate_indices), length=1,
                                                                   max_primitivCounts=2,
                                                                   max_boundsCounts=8, bit_feature=bit_feature,
                                                                   n_jobs=n_jobs)

                new_patter = []
                for fp in new_population[0]:
                    new_patter.append(fp[0])

                for i, indicies in enumerate(generate_indices):
                    # print(f'vorher: {best_n_Member[indicies[0]].S_FP[indicies[1]]}')
                    best_n_Member[indicies[0]].S_FP[indicies[1]] = new_patter[i]
                    # print(f'nachher: {best_n_Member[indicies[0]].S_FP[indicies[1]]}')
                print(f'Aging needs {new_population[1]} tries to create {len(generate_indices)} pattern')

            return best_n_Member

    def _generateNewPopulation(self, n: int, fp_size: int, bit_feature: FeatureTyp,
                               n_jobs: int = 1):
        new_population = self._SMARTG.generate_NewSMARTfps(n, length=fp_size,
                                                           max_primitivCounts=2,
                                                           max_boundsCounts=8, bit_feature=bit_feature,
                                                           n_jobs=n_jobs)

        MemberList = []
        for pop in new_population[0]:
            MemberList.append(Member(pop))

        print(new_population[1], ' patter are tested for create population')
        return MemberList

    def _generatePopulation_Metrics(self, population, fitfunc_worker: int, workfolder=None, step=0):
        IQ_settings = IndexQueue_settings(start_index=0, end_index=len(population), chunksize=1)

        parallelExecuter = ParallelHelper(fitfunc_worker)
        metrics = parallelExecuter.execute_map_orderd_return(self._parallel_generatePopulation_Metrics, IQ_settings,
                                                             np.dtype('O'),
                                                             population=population,
                                                             working_path=workfolder)

        if isinstance(self.fitfunc, MultiDataset_Fitness):
            self.fitfunc.new_random_state()
            weighted_fitnesses = self.fitfunc.calc_multiDataset_fitness(copy.deepcopy(metrics))

            for i, pop in enumerate(population):
                population[i].metric = (weighted_fitnesses[i], metrics[i])
        else:
            for i, pop in enumerate(population):
                population[i].metric = metrics[i]

    def _generatePopulation_Metrics_csvMultiDataset(self, population, workfolder: str, step):
        payload_order = self.fitfunc.get_payload_order()
        stepcsv = os.path.join(workfolder, 'EVOstep_' + str(step) + '_MultiDataset' + '.csv')

        DataContainer = {}

        # first_row FP names
        DataContainer['index'] = []
        for i, p in enumerate(population):
            DataContainer['index'].append(i)

        for i, p in enumerate(payload_order):
            DataContainer[p] = []
            for j, pop in enumerate(population):
                DataContainer[p].append(pop.metric[1][i])

        DataContainer['calc_fitness'] = []
        for i, p in enumerate(population):
            DataContainer['calc_fitness'].append(p.metric[0])

        self._generatePopulation_Metrics_csvMultiDataset_fillup(DataContainer)
        dataFrame_results = pandas.DataFrame(DataContainer)
        dataFrame_results.to_csv(stepcsv, header=True, index=False)

    def _generatePopulation_Metrics_csvMultiDataset_fillup(self, item: Dict[str, List]):
        max_length = 0
        for k in list(item.keys()):
            if len(item[k]) > max_length:
                max_length = len(item[k])

        for k in list(item.keys()):
            current_length = len(item[k])
            padding_length = max_length - current_length
            item[k].extend([''] * padding_length)

    def _parallel_generatePopulation_Metrics(self, population: List[Member], out_dtypes: np.dtype, current_chunk: int,
                                             working_path):
        out_array = np.empty(shape=(len(current_chunk),), dtype=out_dtypes)
        out_pointer = 0
        for current_index in current_chunk:
            out_array[out_pointer] = self.fitfunc.calc_fitness(population[current_index].S_FP, working_path)
            out_pointer += 1
        return out_array

    def _printS(self, text, seperator='!--!'):
        print(seperator + ' ' + text + ' ' + seperator)

    def _starttimer(self, text, seperator='!--!'):
        print(seperator + ' ' + text + ' ' + seperator)
        return perf_counter()

    def _stoptimer(self, startTimer):
        stopTimer = perf_counter()
        print('takes ', stopTimer - startTimer, ' time in seconds')
