import pandas, os

class CSVExporter:

    @staticmethod
    def __lagestArray(ars: []):
        max_size = -1

        for arr in ars:
            try:
                tmp_arr = list(arr)
            except:
                continue

            if isinstance(tmp_arr, list) and max_size < len(tmp_arr):
                max_size = len(tmp_arr)
        return max_size

    @staticmethod
    def __checkFileEnding(path:str):
        if path[-4:] == '.csv':
            return path
        else:
            return path + '.csv'

    @staticmethod
    def exportToCSV(arrays: [], columns_a: [str], path, CSV_filename):
        DataContainer = {}
        maxsize = CSVExporter.__lagestArray(ars=arrays)

        full_name = CSVExporter.__checkFileEnding(CSV_filename)
        i = 0
        for c in columns_a:
            # Todo SO UGLY
            val = arrays[i][0]
            if isinstance(val, dict):
                arrays[i]: dict
                cols = list(val.keys())
                col_arrays = []
                for col in cols:
                    if isinstance(val[col], list):
                        col_arrays.append(val[col])
                    else:
                        col_arrays.append([val[col]])
                CSVExporter.exportToCSV(col_arrays, cols, path, CSV_filename + f'_metric_{c}')
                i = i + 1
            else:
                arrays[i] = list(arrays[i])
                if len(arrays[i]) < maxsize:
                    arrays[i].extend([''] * (maxsize - len(arrays[i])))
                DataContainer[c] = arrays[i]
                i = i + 1

        dataFrame_results = pandas.DataFrame(DataContainer)
        dataFrame_results.to_csv(os.path.join(path, full_name), header=True, index=False)