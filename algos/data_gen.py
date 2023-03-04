import numpy as np


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
