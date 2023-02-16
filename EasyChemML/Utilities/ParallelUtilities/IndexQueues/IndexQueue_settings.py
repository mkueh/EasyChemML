class IndexQueue_settings():

    chunksize:int
    start_index:int
    end_index:int
    fullUtilization:bool

    def __init__(self, chunksize:int=0, start_index:int = 0, end_index:int = 0):
        self.chunksize = chunksize
        self.start_index = start_index
        self.end_index = end_index

    def toTuple(self):
        return (self.chunksize, self.start_index, self.end_index)

    def __len__(self):
        return self.end_index - self.start_index