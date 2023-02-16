from EasyChemML.Utilities.ParallelUtilities.Shared_Value import Shared_Value
from EasyChemML.Utilities.ParallelUtilities.IndexQueues.IndexQueue_settings import IndexQueue_settings
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass

import math

class IndexQueue_Inorder:

    in_order=False

    chunksize: int

    worker_id:int
    worker_count:int

    start_index:int
    end_index:int

    already_processed:Shared_Value #for the processbar


    _generated_items:int = 0
    _generated_chunks: int = 0
    itemCount:int
    chunkCount:int

    def __init__(self, settings: IndexQueue_settings, worker_id: int, worker_count: int):
        blueprint = settings.toTuple()

        self.already_processed = Shared_Value(0, BatchDatatypClass.NUMPY_INT64)

        self.chunksize = blueprint[0]
        self.start_index = blueprint[1]
        self.end_index = blueprint[2]
        self.worker_id = worker_id
        self.worker_count = worker_count

        if self.chunksize == 0:
            total_itemcount = self.end_index - self.start_index
            self.chunksize = math.ceil(total_itemcount/self.worker_count)

        self.current_batch = 0
        self.itemCount = self._worker_itemCount()

    def __iter__(self):
        return self

    def __len__(self):
        return self.end_index - self.start_index

    def reset_closeShared(self):
        self.current_batch = 0
        self._generated_items = 0
        self.itemCount = self._worker_itemCount()
        self.already_processed.destroy()
        self.already_processed = None

    def update_already_processed(self, value:int):
        if not self.already_processed is None:
            self.already_processed.set_val(value)

    def get_already_processed(self):
        if not self.already_processed is None:
            return self.already_processed.get_val()

    def __next__(self):
        self.update_already_processed(self._generated_items)
        if self._generated_items >= self.itemCount:
            raise StopIteration

        chunkblock_length = self.worker_count * self.chunksize
        rest_items = len(self) - chunkblock_length * self._generated_chunks

        if rest_items < self.worker_count:
            #some rest items not processed now
            rest_tmp = len(self) - chunkblock_length * (self._generated_chunks-1)
            rest_items = rest_tmp - (math.floor(rest_tmp / self.worker_count) * self.worker_count)

            if rest_items >= self.worker_id+1:
                range_start = self.start_index + len(self) - rest_items
                range_start += self.worker_id
                self._generated_chunks += 1
                self._generated_items += 1
                return range(range_start, range_start + 1)
            else:
                raise StopIteration

        elif self.worker_count > len(self):
            # more worker than items_left
                self._generated_items += 1
                self._generated_chunks += 1
                val = self.start_index + self.worker_id
                return range(val, val+1)
        elif chunkblock_length > rest_items:
            # last chunkblock
            items_per_worker = math.floor(rest_items / self.worker_count)


            range_start_tmp = self.start_index + chunkblock_length * self._generated_chunks
            range_start = range_start_tmp + items_per_worker * self.worker_id
            range_end = range_start + items_per_worker

            self._generated_chunks += 1
            self._generated_items += range_end - range_start
            return range(range_start, range_end)
        else:
            # not last chunkblock
            range_start = self.start_index + chunkblock_length * self._generated_chunks + self.worker_id * self.chunksize
            range_end = range_start + self.chunksize
            self._generated_chunks += 1
            self._generated_items += range_end - range_start
            return range(range_start, range_end)


    def _worker_itemCount(self):
        minimum_Chunks = math.floor(len(self) / (self.worker_count * self.chunksize))
        items_in_normalChunks = minimum_Chunks * self.chunksize

        items_lastChunk = len(self) - (minimum_Chunks * self.chunksize * self.worker_count)

        minimum_work_perWorker = math.floor(items_lastChunk / self.worker_count)
        restItems = items_lastChunk - (minimum_work_perWorker * self.worker_count)

        if restItems > 0:
            if restItems >= self.worker_id + 1:
                return items_in_normalChunks + minimum_work_perWorker + 1
            else:
                return items_in_normalChunks + minimum_work_perWorker
        else:
            return items_in_normalChunks + minimum_work_perWorker








