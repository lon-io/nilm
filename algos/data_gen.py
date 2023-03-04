import numpy as np


class BatchDataGenerator(object):
    # we default stride to 1 since we dont want future values to affect past values
    def __init__(self, mains, meter, seq_per_batch, seq_len, stride=1):
        assert(len(mains) == len(meter))
        self.data_x = mains
        self.data_y = meter
        self.seq_per_batch = seq_per_batch # the number of seqeunces per batch
        self.seq_len = seq_len # the length of each sequence
        self.stride = stride
        self.current_idx = 0
        self.data_len = len(self.data_x)

    def load_sequence(self, x, y, seq_index):
        seq = self.data_x[self.current_idx:self.current_idx + self.seq_len]
        # Pad initial sequences since they will surely have zero padding if seq_len >
        x[seq_index, self.seq_len - len(seq):] = seq
        y[seq_index] = self.data_y[self.current_idx + 1]

    def load_all(self):
        n_batches = self.data_len / self.stride / self.seq_per_batch
        n_batches = np.ceil(n_batches).astype(int)

        x = np.zeros((n_batches * self.seq_per_batch * self.stride, self.seq_len, 1), dtype=np.float32)
        y = np.zeros((n_batches * self.seq_per_batch * self.stride), dtype=np.float32)

        print('self.data_len', self.data_len)
        print('stride', self.stride)
        print('seq_per_batch', self.seq_per_batch)
        print('seq_len', self.seq_len)
        print('n_batches', n_batches)
        print('x.shape, y.shape', x.shape, y.shape)

        seq_index = 0
        for batch_index in range(n_batches):
            for _ in range(self.seq_per_batch):
                if self.current_idx + self.seq_len >= self.data_len:
                    self.current_idx = 0
                self.load_sequence(x, y, seq_index)
                self.current_idx += self.stride
                seq_index += 1

            if batch_index % 200 == 0:
                print('batch_index', batch_index)

        return x, y

    def generate(self):
        x = np.zeros((self.seq_per_batch, self.seq_len, 1), dtype=np.float32)
        y = np.zeros((self.seq_per_batch), dtype=np.float32)
        while True:
            for seq_index in range(self.seq_per_batch):
                if self.current_idx + self.seq_len >= self.data_len:
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                self.load_sequence(x, y, seq_index)
                self.current_idx += self.stride
            yield x, y

class Seq2PointDataGenerator(object):
    # we default stride to 1 since we dont want future values to affect past values
    def __init__(self, mains, meter, seq_len):
        assert(len(mains) == len(meter))
        self.data_x = mains
        self.data_y = meter
        self.seq_len = seq_len # the length of each sequence
        self.data_len = len(self.data_x)

    def load_sequence(self, x, y, data_index, seq_index):
        seq = self.data_x[data_index:data_index + self.seq_len]
        output = self.data_y[data_index + self.seq_len - 1]
        x[seq_index] = seq
        y[seq_index] = output

    def load_all(self):
        n_sequences = self.data_len - self.seq_len

        x = np.zeros((n_sequences, self.seq_len, 1), dtype=np.float32)
        y = np.zeros((n_sequences), dtype=np.float32)

        print('self.data_len', self.data_len)
        print('seq_len', self.seq_len)
        print('n_sequences', n_sequences)
        print('x.shape, y.shape', x.shape, y.shape)

        data_index = 0
        for seq_index in range(n_sequences):
            self.load_sequence(x, y, data_index, seq_index)
            data_index += 1

            if seq_index % (5000 * self.seq_len) == 0:
                print('seq_index', seq_index)

        return x, y
