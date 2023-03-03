import numpy as np


class Seq2PointMultiDataGenerator(object):
    def __init__(self, mains, meters, seq_len):
        print('mains.shape, meters.shape => ', mains.shape, meters.shape)
        assert(mains.shape[0] == meters.shape[0])
        self.data_x = mains
        self.data_y = meters
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
        y = np.zeros((n_sequences, 3), dtype=np.float32)

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

