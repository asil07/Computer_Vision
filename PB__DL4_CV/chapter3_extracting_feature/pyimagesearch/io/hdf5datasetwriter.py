import h5py
import os


class HDF5DatasetWriter:

    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):

        if os.path.exists(outputPath):
            raise ValueError("The supplied ‘outputPath‘ already "
                             "exists and cannot be overwritten. Manually delete "
                             "the file before continuing. \nBerilgan 'outputPath' allaqachon"
                             "bor va qayta yoza olmaydi. "
                             "Davom etishdan oldin faylni o'chiring", outputPath)

        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.labels = self.db.create_dataset("label", (dims[0],), dtype="int")

        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)

        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx: i] = self.buffer['data']
        self.labels[self.idx: i] = self.labels['labels']
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        dt = h5py.special_dtype(vlen="unicode")
        labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()




