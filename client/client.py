import time
import socket
import struct
import numpy as np

import utils
import argparse

from tqdm import tqdm


class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def __del__(self):
        self.sock.close()

    def establish_connection(self):
        self.sock.connect((self.host, self.port))

    def communicate(self, dataset, n, m):
        print("Vectors shape: ", n, m)

        self._send_data(dataset, n, m)
        self._check_acceptance()
        return self._receive_result(n)

    def _send_data(self, dataset, n, m):
        print("Sending data...")
        self.sock.sendall(struct.pack('<i', 1) + utils.DELIMITER +
                          struct.pack('<ii', *[n, m]) + utils.DELIMITER)

        for i in tqdm(dataset.iter_chunks(), total=(n * m // utils.CHUNK_SIZE)):
            data_chunk = dataset[i].tobytes()
            self.sock.sendall(data_chunk)
            del data_chunk

        self.sock.sendall(utils.DELIMITER)

    def _check_acceptance(self):
        send_res = self.sock.recv(4)
        send_res = struct.unpack('i', send_res)[0]
        if send_res == 2:
            print("Data sent successfully!")
        else:
            print(send_res)
            raise RuntimeError("Data is not received!")

    def _receive_result(self, n):
        while True:
            time.sleep(0.1)

            self.sock.sendall(struct.pack('<i', 0) + utils.DELIMITER)
            result = b''
            while len(result) < (n + 1) * 8:
                result += self.sock.recv((n + 1) * 8 - len(result))

            if result:
                break

        result = struct.unpack(f'{n + 1}d', result)
        execution_time = result[-1]
        result = np.array(result[: -1])

        return result, execution_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('host', type=str, help='Input host')
    parser.add_argument('port', type=int, help='Input port')
    parser.add_argument('data_path', type=str, help='Input file path')
    parser.add_argument('-n', '--num_vectors', type=int, help='Input number of vectors', default=0)
    parser.add_argument('-d', '--dim_vectors', type=int, help='Input dimension of vectors', default=0)

    args = parser.parse_args()
    host = args.host
    port = int(args.port)
    data_path = args.data_path
    num_vectors = int(args.num_vectors)
    dim_vectors = int(args.dim_vectors)

    if num_vectors == 0:
        h5file, dataset = utils.load_data(data_path)
        num_vectors = dataset.attrs['num_vectors']
        dim_vectors = dataset.attrs['dim_vectors']
    else:
        h5file, dataset = utils.generate_random_data(data_path, num_vectors, dim_vectors)

    client = Client(host, port)
    client.establish_connection()

    lengths, execution_time = client.communicate(dataset, num_vectors, dim_vectors)
    assert len(lengths) == num_vectors

    # maybe we dont need chunks=True
    if "lengths" not in h5file:
        h5file.create_dataset("lengths", shape=lengths.shape, data=lengths)

    h5file.attrs["execution_time"] = execution_time
    h5file.close()

    print(f'Calculation_time: {execution_time} microseconds')
    print()


if __name__ == "__main__":
    main()
