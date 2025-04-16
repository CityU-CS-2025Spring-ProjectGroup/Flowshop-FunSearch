from flowshop_test.utils import *

from original import original_priority
from evolved import evolved_priority, evolved_priority_2


def order_schedule(jobs, use_evolved=False):
    remaining_jobs = list(enumerate(jobs))
    schedule = []

    while remaining_jobs:
        # 计算每个作业的优先级
        priorities = []
        for idx, job in remaining_jobs:
            if use_evolved:
                priority = evolved_priority_2(job)
            else:
                priority = original_priority(job)
            priorities.append((priority, -job[0], idx, job))  # 总时间相同则选首机器时间大的

        # 按优先级降序排序
        priorities.sort(reverse=True, key=lambda x: (x[0], x[1]))

        # 选择优先级最高的作业
        selected = priorities[0]
        schedule.append(selected[2])
        remaining_jobs = [(idx, job) for idx, job in remaining_jobs if idx != selected[2]]

    return schedule


# 计算调度顺序的makespan
def calculate_makespan(order, proc_time):
    num_machines = len(proc_time[0])
    num_jobs = len(order)
    machine_times = [[0] * (num_jobs + 1) for _ in range(num_machines)]

    for i in range(num_jobs):
        job_id = order[i]
        times = proc_time[job_id]
        machine_times[0][i + 1] = machine_times[0][i] + times[0]
        for m in range(1, num_machines):
            machine_times[m][i + 1] = max(machine_times[m][i], machine_times[m - 1][i + 1]) + times[m]

    return machine_times[-1][-1]


if __name__ == '__main__':

    name = 'reeves'

    for i in range(1, 22):
        filename = f'{name}{i}.txt'
        fs_data = load_datasets(
            f'/Users/cuiguangyuan/Documents/CityU/SemesterB/Artificial Intelligence/project/Funsearch_on_flowshop/data/{name}')[
            filename]
        fs_data = np.array(fs_data)

        jobs = fs_data

        original_schedule = order_schedule(jobs)
        original_makespan = calculate_makespan(original_schedule, jobs)

        evolved_schedule = order_schedule(jobs, True)
        evolved_makespan = calculate_makespan(evolved_schedule, jobs)

        print(f"Original Makespan: {original_makespan} | Evolved Makespan: {evolved_makespan} | Is Better: {evolved_makespan <= original_makespan}")

        # plot_gantt_chart(schedule, jobs)
