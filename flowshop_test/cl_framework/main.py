import numpy as np

from flowshop_test.utils import *
from original import neh
from evolved import evolved_neh


def main(name: str, idx: int, use_evolved: bool = False):

    filename = f'{name}{idx}.txt'
    fs_data = load_datasets(f'/Users/cuiguangyuan/Documents/CityU/SemesterB/Artificial Intelligence/project/Funsearch_on_flowshop/data/{name}')[filename]
    fs_data = np.array(fs_data)

    if use_evolved:
        schedule = evolved_neh(fs_data)
    else:
        schedule = neh(fs_data)

    final_makespan = calc_makespan(schedule, fs_data)

    # plot_gantt_chart(schedule, fs_data)

    return schedule, final_makespan


if __name__ == '__main__':

    name = 'reeves'

    for i in range(1, 22):
        _, original_makespan = main(name, i)
        _, evolved_makespan = main(name, i, True)

        print(f"Original Makespan: {original_makespan} | Evolved Makespan: {evolved_makespan} | Is Better: {evolved_makespan <= original_makespan}")
