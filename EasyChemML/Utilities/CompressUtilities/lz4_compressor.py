import lz4.frame, pickle


class lz4_compressor(object):

    def __init__(self):
        pass

    def compress_object(self, object, lz4_args={}):
        pickled_object = pickle.dumps(object, protocol=pickle.HIGHEST_PROTOCOL)
        return lz4.frame.compress(pickled_object, **lz4_args)

    def compress_object_to_file(self, object, file_path, lz4_args={}):
        data = self.compress_object(object, lz4_args)

        f = open(file_path, 'wb')
        f.write(data)
        f.flush()
        f.close()

    def decompress_object(self, data, lz4_args={}):
        raw_pickle = lz4.frame.decompress(data, **lz4_args)
        return pickle.loads(raw_pickle)

    def decompress_object_from_file(self, file_path, lz4_args={}):
        f = open(file_path, 'rb')
        data = f.read()
        f.close()
        return self.decompress_object(data, lz4_args)
