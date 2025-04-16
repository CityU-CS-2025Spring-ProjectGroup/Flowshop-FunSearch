import numpy as np


# 计算调度顺序的makespan
def calculate_makespan(order: list[int], proc_time: np.ndarray) -> int:
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


def order_schedule(jobs: np.ndarray) -> list[int]:
    remaining_jobs = list(enumerate(jobs))
    schedule = []

    while remaining_jobs:
        # 计算每个作业的优先级
        priorities = []
        for idx, job in remaining_jobs:
            priority = calc_priority(job)
            priorities.append((priority, -job[0], idx, job))

        # 按优先级降序排序
        priorities.sort(reverse=True, key=lambda x: (x[0], x[1]))

        # 选择优先级最高的作业
        selected = priorities[0]
        schedule.append(selected[2])
        remaining_jobs = [(idx, job) for idx, job in remaining_jobs if idx != selected[2]]

    return schedule


@funsearch.run
def evaluate(instances: dict) -> float:
    """
    FunSearch evaluation function that computes the average makespan across multiple datasets.
    - instances: dict mapping instance names to 2D numpy arrays (processing time matrices).
    Returns the negative mean makespan (float) for optimization.
    """
    makespans = []
    for name in instances:
        processing_times = instances[name]
        if not isinstance(processing_times, np.ndarray):
            print(f"[ERROR] Instance {name} is not ndarray")
            continue
        if not np.issubdtype(processing_times.dtype, np.integer):
            processing_times = processing_times.astype(int)

        schedule = order_schedule(processing_times)
        ms = calculate_makespan(schedule, processing_times)
        makespans.append(ms)

    if not makespans:
        return 1e9
    return -float(np.mean(makespans))


@funsearch.evolve
def calc_priority(job: np.ndarray) -> float:
    priority = 0
    total_time = sum(job)
    first_machine_time = job[0]
    last_machine_time = job[-1]
    num_stages = len(job)

    # New Weights
    w1, w2, w3, w4, w5, w6 = 0.35, 0.2, 0.2, 0.15, 0.05, 0.05

    # Factor 1: First machine processing time
    priority += w1 * (first_machine_time ** 1.2)

    # Factor 2: Total job processing time
    total_time_efficiency = total_time / max(job)
    priority += w2 * total_time_efficiency

    # Factor 3: Utilization of stages
    utilization = sum([min(job) / stage_time for stage_time in job]) / num_stages
    priority += w3 * utilization

    # Factor 4: Variance between stages
    variance = np.var(job)
    priority += w4 * (1 / (variance + 1))  # More stable is better

    # Factor 5: First and Last machine time ratio
    if first_machine_time < last_machine_time:
        priority += w5 * (last_machine_time - first_machine_time)

    # Factor 6: Dynamic gaps calculation
    large_gap_penalty = 0
    gaps = [(job[i], abs(job[i] - job[i - 1])) for i in range(1, num_stages)]
    for stage, gap in gaps:
        if gap > np.median(job):
            large_gap_penalty += gap / stage
    priority -= w6 * large_gap_penalty

    # Further dynamic adjustment based on variance and extreme values
    extreme_count = sum(1 for time in job if time > 1.2 * total_time_efficiency)
    if extreme_count > num_stages / 3:
        priority *= 0.9  # Penalize with more extreme times

    for i in range(num_stages):
        stage_importance = i / num_stages
        if job[i] > total_time / num_stages:
            priority += stage_importance * 0.03 * job[i]  # Prioritize key stages more if above average

    # Introduce randomness for tie-breaking
    np.random.seed(int(total_time * 100) + 42)
    priority += np.random.normal(0, 0.7)

    return -priority

