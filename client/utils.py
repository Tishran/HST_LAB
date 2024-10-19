import h5py
import numpy as np
from tqdm import tqdm

DELIMITER = b'\n'
CHUNK_SIZE = int(1e7)
MAX_VALUE = int(1e6)


def generate_random_data(data_path, num_vectors, dim_vectors, verbose=True):
    if verbose:
        print("Generating data...")

    h5file = h5py.File(data_path, 'w')
    total_count = num_vectors * dim_vectors
    suitable_chunk_size = min(CHUNK_SIZE, total_count)

    dataset = h5file.create_dataset("vectors", shape=(total_count,), dtype='<i4', chunks=suitable_chunk_size)
    dataset.attrs['num_vectors'] = num_vectors
    dataset.attrs['dim_vectors'] = dim_vectors

    for chunk in tqdm(dataset.iter_chunks(), total=(num_vectors * dim_vectors // CHUNK_SIZE), disable=(not verbose)):
        dataset[chunk] = np.random.randint(-MAX_VALUE, MAX_VALUE,
                                           size=min(suitable_chunk_size, total_count))

        total_count -= suitable_chunk_size

    return h5file, dataset


def load_data(data_path, verbose=True):
    if verbose:
        print("Loading data...")

    h5file = h5py.File(data_path, 'r+')
    dataset = h5file['vectors']

    return h5file, dataset
