import argparse
import random
import numpy as np
import subprocess
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('res_path', type=str, help='Input file name')
parser.add_argument('times_path', type=str, help='Input file name')
args = parser.parse_args()
file_path = args.res_path
times_path = args.res_path

MAX_NUM = 1000000
MAX_DIM = 100000

calculation_times = dict()

for i in range(20):
    if i % 10 == 0:
        print(i)
    num_vec = random.randint(10, MAX_NUM)
    dim_vec = 10
    # dim_vec = random.randint(10, MAX_DIM)

    subprocess.run(["python", "./client/gen_data.py", 'data.npy', f"{num_vec}", f"{dim_vec}"])
    subprocess.run(["python", "./client/send_data.py", 'data.npy', file_path, '127.0.0.1', '8080'])

    lengths = np.load(file_path)
    calculation_times[(num_vec, dim_vec)] = lengths[-1]


with open('/home/tishran/CLionProjects/HST_LAB_1/times.pkl', 'wb') as fp:
    pickle.dump(calculation_times, fp)
    print('dictionary saved successfully to file')