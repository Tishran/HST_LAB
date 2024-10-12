import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

EXPERIMENT_RESULTS_PATH = './experiment_results'

def main():
    exp_dir_list = [file_name for file_name in os.listdir(EXPERIMENT_RESULTS_PATH) if file_name.endswith('.pkl')]
    exp_dir_list.sort(key=lambda x: int(x.split('_')[1]))
    for file_name in exp_dir_list:
        with open(os.path.join(EXPERIMENT_RESULTS_PATH, file_name), 'rb') as f:
            calculation_times = pickle.load(f)
            plt.plot(np.array(list(calculation_times.keys())),
                     list(calculation_times.values()),
                     label=f"{file_name.split('_')[1]} threads",
                     linestyle='-')

    plt.title(f'Calculation duration comparison')
    plt.ylabel('Duration, mcs')
    plt.xlabel('Data size, MB')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.savefig(f'{os.path.join(EXPERIMENT_RESULTS_PATH, 'comparison_plot')}.png', format='png', dpi=600)
    print(f'Comparison plot saved successfully')


if __name__ == "__main__":
    main()
