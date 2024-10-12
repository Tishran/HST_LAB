import numpy as np
from numpy.ma.testutils import assert_almost_equal
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Input file path')
    parser.add_argument('lengths_path', type=str, help='Input file path')
    args = parser.parse_args()
    input_path = args.data_path
    output_path = args.lengths_path

    vectors = np.memmap(input_path, dtype=np.int32, mode='r')
    n = vectors[0]
    m = vectors[1]

    lengths = np.load(output_path)
    lengths = lengths[:-1]

    true_lengths = []
    #np.array([np.linalg.norm(i) for i in vectors])
    for i in range(2, len(vectors), m):
        true_lengths.append(np.linalg.norm(vectors[i: i + m]))

    true_lengths = np.array(true_lengths)

    assert_almost_equal(true_lengths, lengths, decimal=3)


if __name__ == '__main__':
    main()