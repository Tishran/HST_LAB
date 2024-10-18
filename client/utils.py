import numpy as np
from tqdm import tqdm

DELIMITER = b'\n'
CHUNK_SIZE = int(1e7)
MAX_VALUE = int(1e6)


def generate_random_data(data_path, num_vectors, dim_vectors):
    print("Generating data...")
    mmap_array = np.memmap(data_path, dtype="<i4", mode='w+', shape=num_vectors * dim_vectors + 2)
    mmap_array[0] = num_vectors
    mmap_array[1] = dim_vectors

    total_numbers = num_vectors * dim_vectors
    for i in tqdm(range(2, len(mmap_array), CHUNK_SIZE)):
        mmap_array[i: i + min(CHUNK_SIZE, total_numbers)] = np.random.randint(-MAX_VALUE, MAX_VALUE,
                                                                              size=min(CHUNK_SIZE, total_numbers))
        total_numbers -= CHUNK_SIZE

    return mmap_array


def load_data(data_path):
    print("Loading data...")
    data = np.memmap(data_path, dtype=np.int32, mode='r')
    n, m = data[0], data[1]

    return n, m, data
