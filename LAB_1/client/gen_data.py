import numpy as np
import argparse

from tqdm import tqdm

MAX_VALUE = int(1e6)
CHUNK_SIZE = int(1e7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Input file name')
    parser.add_argument('n_vectors', type=str, help='Input number of vectors')
    parser.add_argument('dim_vectors', type=str, help='Input dimension of vectors')
    args = parser.parse_args()
    file_path = args.data_path

    num_vectors = int(args.n_vectors)
    dim_vectors = int(args.dim_vectors)

    mmap_array = np.memmap(file_path, dtype="<i4", mode='w+', shape=num_vectors * dim_vectors + 2)
    mmap_array[0] = num_vectors
    mmap_array[1] = dim_vectors

    total_numbers = num_vectors * dim_vectors
    for i in tqdm(range(2, len(mmap_array), CHUNK_SIZE)):
        mmap_array[i: i + min(CHUNK_SIZE, total_numbers)] = np.random.randint(-MAX_VALUE, MAX_VALUE,
                                                                              size=min(CHUNK_SIZE, total_numbers))
        total_numbers -= CHUNK_SIZE


if __name__ == '__main__':
    main()
