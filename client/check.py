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

    vectors = np.load(input_path)
    lengths = np.load(output_path)
    lengths = lengths[:-1]

    true_lengths = np.array([np.linalg.norm(i) for i in vectors])

    assert_almost_equal(true_lengths, lengths, decimal=3)


if __name__ == '__main__':
    main()