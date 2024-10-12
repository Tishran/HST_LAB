import gc
import socket
import struct
import numpy as np
import argparse
import time

from tqdm import tqdm

# TODO: fix performance and refactor

DELIMITER = b'\n'

# optimal value for my machine with linux to run without crushes, but maybe i dont need this anymore
CHUNK_SIZE = int(1e7)


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

        data = np.memmap(data_path, dtype=np.int32, mode='r')

        n, m = data[0], data[1]
        print("vectors shape: ", n, m)
        print(data[2])

        print("flattened vectors len: ", len(data))

        print("sending data...")
        s.sendall(struct.pack('<i', 1) + DELIMITER +
                  struct.pack('<ii', *[n, m]) + DELIMITER)

        for i in tqdm(range(2, len(data), CHUNK_SIZE)):
            data_chunk = data[i: min(len(data), i + CHUNK_SIZE)].tobytes()
            s.sendall(data_chunk)
            del data_chunk

        s.sendall(DELIMITER)

        del data
        gc.collect()

        # checking for data accept
        send_res = s.recv(4)
        send_res = struct.unpack('i', send_res)[0]
        if send_res == 2:
            print("data sent successfully!")
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
