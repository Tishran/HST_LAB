import numba as nb
import numpy as np
import argparse
import utils

from tqdm import tqdm
from numpy.ma.testutils import assert_almost_equal


@nb.njit(fastmath=True)
def norm(l):
    s = 0.
    for i in range(l.shape[0]):
        s += l[i]**2
    return np.sqrt(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Input file path')
    args = parser.parse_args()
    data_path = args.data_path

    h5file, vectors = utils.load_data(data_path)
    lengths = h5file['lengths']
    n = vectors.attrs['num_vectors']
    m = vectors.attrs['dim_vectors']

    print("Vector shape: ", n, m)

    true_lengths = []
    for i in tqdm(range(0, len(vectors), m)):
        true_lengths.append(norm(vectors[i: i + m]))

    true_lengths = np.array(true_lengths)

    assert_almost_equal(true_lengths, lengths, decimal=3)
    h5file.close()
    print("Everything is alright!")


if __name__ == '__main__':
    main()
