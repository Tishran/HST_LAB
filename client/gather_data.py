import argparse
import numpy as np
import subprocess
import pickle
import matplotlib.pyplot as plt
import os
import re

from tqdm import tqdm

EXPERIMENT_RESULTS_PATH = '../experiment_results'


def extract_duration(output_string):
    match = re.search(r'Calculation_time:\s*([\d.]+)\s*microseconds', output_string)

    if match:
        return float(match.group(1))
    else:
        raise RuntimeError("No execution time found in output of client.py!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('start_num_vectors', type=str)
    parser.add_argument('dim_vectors', type=str)  # it is fixed
    parser.add_argument('step_num_vectors', type=str)
    parser.add_argument('num_experiments', type=str)
    parser.add_argument('plot_name', type=str)
    parser.add_argument('pickle_name', type=str)

    args = parser.parse_args()
    start_num_vectors = int(args.start_num_vectors)
    dim_vectors = int(args.dim_vectors)
    step_num_vectors = int(args.step_num_vectors)
    num_experiments = int(args.num_experiments)
    plot_name = args.plot_name
    pickle_name = args.pickle_name

    plot_name = os.path.join(EXPERIMENT_RESULTS_PATH, plot_name)
    pickle_name = os.path.join(EXPERIMENT_RESULTS_PATH, pickle_name)

    calculation_times = dict()  # num vectors to time

    for _ in tqdm(range(num_experiments)):
        result = subprocess.run(['python3', 'client.py', '127.0.0.1', '12345',
                        'data.npy', 'lengths.npy', '-n', str(start_num_vectors), '-d', str(dim_vectors)],
                       capture_output=True,
                       text=True)

        if result.returncode != 0:
            raise RuntimeError("Client finished with nonzero code!")

        calculation_times[start_num_vectors * dim_vectors * 4 / 1024 / 1024] = extract_duration(result.stdout)

        start_num_vectors += step_num_vectors

    print("Experiments finished!")
    print("Saving plot and duration records...")

    plt.plot(list(calculation_times.keys()), list(calculation_times.values()),
             label='time of size',
             color='blue',
             linestyle='-')
    plt.title(f'Calculation duration with dim={dim_vectors}')
    plt.ylabel('Duration, mcs')
    plt.xlabel('Data size, MB')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.savefig(f'{plot_name}.png', format='png', dpi=600)
    print(f'Plot saved successfully to {plot_name}.png')

    with open(f"{pickle_name}.pkl", 'wb') as fp:
        pickle.dump(calculation_times, fp)
        print(f'Durations records saved successfully to {pickle_name}.pkl')


if __name__ == "__main__":
    main()