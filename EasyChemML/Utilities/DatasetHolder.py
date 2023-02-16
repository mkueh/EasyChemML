from EasyChemML.Utilities.Dataset import Dataset
from EasyChemML.Splitter.Splitcreator import Splitset

from typing import List

import os

class DatasetHolder(object):

    splitted:bool = False
    splitted_by:str = ''
    _holder:List[Dataset]

    def __init__(self):
        self._holder = []
        self._splitted_by = ''
        self._splitted = False

    def remove_all(self):
        self._holder = []

    def __lshift__(self, dataset:Dataset):
        self._holder.append(dataset)

    def __getitem__(self, item):
        return self._holder.__getitem__(item)

    def __len__(self):
        return len(self._holder)

    def __iter__(self):
        return self._holder.__iter__()

    def split_and_copy(self):
        new_holder = []
        self.splitted = True
        for dataset in self._holder:
            delete = False

            batch_partition = dataset.getBatchPartition()
            dataset_key_feature = dataset.getFeature_key()
            dataset_key_target = dataset.getTarget_key()
            dataset_split = dataset.getSplits()

            settings_outer = dataset_split.get_out_splits_settings()
            settings_inner = dataset_split.get_in_splits_settings()

            if dataset_split.get_outerCount() > 1:
                dataset_outer_splitts = dataset_split.get_outer_splitts()

                for i, outer_splitt in enumerate(dataset_outer_splitts):
                    innersplit = dataset_split.get_inner_splitts(i)
                    split = Splitset([outer_splitt], innersplit, settings_outer, settings_inner)

                    new_feature_key = dataset_key_feature + f'_{i}'
                    new_target_key = dataset_key_target + f'_{i}'

                    batch_partition.cloneBatchTable(dataset_key_feature, new_feature_key)
                    batch_partition.cloneBatchTable(dataset_key_target, new_target_key)

                    new_name = dataset.name + f'_{i}'
                    workingfolder = dataset.getOutputFolder()
                    workingfolder = os.path.join(workingfolder, f'outsplit_{i}')
                    if not os.path.exists(workingfolder):
                        os.mkdir(workingfolder)

                    TMP_FOLDER = dataset.getTMP_FOLDER()
                    TMP_FOLDER = os.path.join(TMP_FOLDER, f'TMP_outsplit_{i}')
                    if not os.path.exists(TMP_FOLDER):
                        os.mkdir(TMP_FOLDER)

                    columns = dataset.getEncoding_Columns()

                    new_Dataset = Dataset(new_feature_key, new_target_key, split, new_name, workingfolder, TMP_FOLDER, columns, batch_partition)
                    new_holder.append(new_Dataset)
                    delete = True
            else:
                new_holder.append(dataset)

            if delete:
                del batch_partition[dataset_key_target]
                del batch_partition[dataset_key_feature]

        if len(new_holder) > 0:
            self._holder = new_holder