import gc
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

    def communicate(self, data, n, m):
        print("Vectors shape: ", n, m)

        self._send_data(data, n, m)
        self._check_acceptance()
        return self._receive_result(n)

    def _send_data(self, data, n, m):
        print("Sending data...")
        self.sock.sendall(struct.pack('<i', 1) + utils.DELIMITER +
                          struct.pack('<ii', *[n, m]) + utils.DELIMITER)

        for i in tqdm(range(2, len(data), utils.CHUNK_SIZE)):
            data_chunk = data[i: min(len(data), i + utils.CHUNK_SIZE)].tobytes()
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
        result = result[: -1]

        return result, execution_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('host', type=str, help='Input host')
    parser.add_argument('port', type=int, help='Input port')
    parser.add_argument('result_path', type=str, help='Input file path')
    parser.add_argument('data_path', type=str, help='Input file path')
    parser.add_argument('-n', '--num_vectors', type=int, help='Input number of vectors', default=0)
    parser.add_argument('-d', '--dim_vectors', type=int, help='Input dimension of vectors', default=0)

    args = parser.parse_args()
    host = args.host
    port = int(args.port)
    result_path = args.result_path
    data_path = args.data_path
    num_vectors = int(args.num_vectors)
    dim_vectors = int(args.dim_vectors)

    if num_vectors == 0:
        num_vectors, dim_vectors, data = utils.load_data(data_path)
    else:
        data = utils.generate_random_data(data_path, num_vectors, dim_vectors)

    client = Client(host, port)
    client.establish_connection()

    lengths, execution_time = client.communicate(data, num_vectors, dim_vectors)
    assert len(lengths) == num_vectors

    del data
    gc.collect()

    with open(result_path, 'wb') as f:
        np.save(f, lengths)

    print(f'Calculation_time: {execution_time} microseconds')
    print()


if __name__ == "__main__":
    main()
