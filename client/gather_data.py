import argparse
import subprocess
import pickle
import matplotlib.pyplot as plt

import utils
import os

from tqdm import tqdm

EXPERIMENT_RESULTS_PATH = '../experiment_results'
TCP_RUN = ['python3', '../client/client.py', '127.0.0.1', '12345', 'data.h5']
NO_TCP_RUN = ['python3', '../client/client.py', 'data.h5']

# change -np value as you wish
N_PROC_VAL = 20
N_ROUNDS = 5
MPI_RUN = f'/usr/local/bin/mpirun -np {N_PROC_VAL} /home/tishran/CLionProjects/HST_LAB/LAB_3/cmake-build-debug/LAB_3 data.h5'.split()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('start_num_vectors', type=str)
    parser.add_argument('dim_vectors', type=str)  # it is fixed
    parser.add_argument('step_num_vectors', type=str)
    parser.add_argument('num_experiments', type=str)
    parser.add_argument('plot_name', type=str)
    parser.add_argument('pickle_name', type=str)
    parser.add_argument("--mpi", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    start_num_vectors = int(args.start_num_vectors)
    dim_vectors = int(args.dim_vectors)
    step_num_vectors = int(args.step_num_vectors)
    num_experiments = int(args.num_experiments)
    plot_name = args.plot_name
    pickle_name = args.pickle_name
    mpi = bool(args.mpi)

    plot_name = os.path.join(EXPERIMENT_RESULTS_PATH, plot_name)
    pickle_name = os.path.join(EXPERIMENT_RESULTS_PATH, pickle_name)

    calculation_times = dict()  # num vectors to time

    for _ in tqdm(range(num_experiments)):
        curr_size = start_num_vectors * dim_vectors * 4 / 1024 / 1024
        calculation_times[curr_size] = 0

        for i in range(N_ROUNDS):
            result = subprocess.run(
                (NO_TCP_RUN if mpi else TCP_RUN) + ['-n', str(start_num_vectors), '-d', str(dim_vectors)],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(result.stderr)

            if mpi:
                result = subprocess.run(MPI_RUN,
                                        capture_output=True,
                                        text=True
                                        )

                if result.returncode != 0:
                    raise RuntimeError(result.stderr)

            h5file, dataset = utils.load_data('data.h5', verbose=False)
            calculation_times[curr_size] += h5file.attrs['execution_time']
            h5file.close()

        calculation_times[curr_size] /= 10
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

    png_name = f'{plot_name}_{N_PROC_VAL}.png'
    pkl_name = f"{pickle_name}_{N_PROC_VAL}.pkl"

    plt.savefig(png_name, format='png', dpi=600)
    print(f'Plot saved successfully to {png_name}')

    with open(pkl_name, 'wb') as fp:
        pickle.dump(calculation_times, fp)
        print(f'Durations records saved successfully to {pkl_name}')


if __name__ == "__main__":
    main()
