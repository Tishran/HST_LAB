import gc
import socket
import struct
import time
import numpy as np
import argparse

# TODO: fix performance and refactor

DELIMITER = b'\n'
CHUNK_SIZE = int(1e8)  # optimal value for my machine with linux to run without crushes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Input file name')
    parser.add_argument('output_file', type=str, help='Input file name')
    parser.add_argument('host', type=str, help='Input host address')
    parser.add_argument('port', type=str, help='Input port number')
    args = parser.parse_args()
    data_path = args.input_file
    result_path = args.output_file
    host = args.host
    port = int(args.port)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        data = np.load(data_path, mmap_mode='r')
        n, m = data.shape

        print("vectors shape: ", data.shape)

        data = data.flatten()

        print("flattened vectors len: ", len(data))

        s.sendall(struct.pack('<i', 1) + DELIMITER +
                  struct.pack('<ii', *[n, m]) + DELIMITER)

        total_len = n * m
        for i in range(0, n * m, CHUNK_SIZE):
            s.sendall(struct.pack(f"<{min(total_len, CHUNK_SIZE)}i",
                                  *(data[i: i + min(total_len, CHUNK_SIZE)])))
            total_len -= CHUNK_SIZE
        s.sendall(DELIMITER)

        del data
        gc.collect()
        print("Cleared data in python script")

        # checking for data accept
        send_res = s.recv(4)
        send_res = struct.unpack('i', send_res)[0]
        if send_res == 2:
            print("Data received!")
        else:
            print(send_res)
            raise RuntimeError("Data is not received!")

        while True:
            time.sleep(0.5)

            s.sendall(struct.pack('<i', 0) + DELIMITER)
            result = b''
            while len(result) < (n + 1) * 8:
                result += s.recv((n + 1) * 8 - len(result))

            if result:
                break

        print(len(result))
        result = struct.unpack(f'{n + 1}d', result)
        execution_time = result[-1]
        with open(result_path, 'wb') as f:
            np.save(f, result)

        print(f'Calculation_time: {execution_time} microseconds')
        print()


if __name__ == "__main__":
    main()
