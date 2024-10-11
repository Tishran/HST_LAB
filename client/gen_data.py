import numpy as np
import argparse
import pickle

MAX_VALUE = int(1e6)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Input file name')
    parser.add_argument('n_vectors', type=str, help='Input number of vectors')
    parser.add_argument('dim_vectors', type=str, help='Input dimension of vectors')
    args = parser.parse_args()
    file_path = args.data_path

    num_vectors = int(args.n_vectors)
    dim_vectors = int(args.dim_vectors)

    vectors = np.random.randint(-MAX_VALUE, MAX_VALUE, size=(num_vectors, dim_vectors), dtype=np.int32)
    with open(file_path, 'wb') as f:
        np.save(f, vectors)


if __name__ == '__main__':
    main()