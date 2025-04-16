from typing import List
import numpy as np


def compute_makespan(schedule: list[int], processing_times: np.ndarray) -> int:
    """
    Compute the makespan (total completion time) for a given job schedule in a PFSP.
    - schedule: list of job indices in the order they are processed.
    - processing_times: 2D numpy array of shape (num_jobs, num_machines) with processing times for each job on each machine.
    Returns the makespan (int) for the given order.
    """
    num_jobs = len(schedule)
    num_machines = processing_times.shape[1]
    if num_jobs == 0:
        return 0

    completion_times = np.zeros((num_jobs, num_machines), dtype=int)
    first_job = schedule[0]
    completion_times[0, 0] = processing_times[first_job, 0]
    for m in range(1, num_machines):
        completion_times[0, m] = completion_times[0, m - 1] + processing_times[first_job, m]

    for i in range(1, num_jobs):
        job = schedule[i]
        completion_times[i, 0] = completion_times[i - 1, 0] + processing_times[job, 0]
        for m in range(1, num_machines):
            completion_times[i, m] = max(completion_times[i, m - 1], completion_times[i - 1, m]) + processing_times[
                job, m]

    return int(completion_times[-1, -1])


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

        schedule = neh_heuristic(processing_times)
        ms = compute_makespan(schedule, processing_times)
        makespans.append(ms)

    if not makespans:
        return 1e9
    return -float(np.mean(makespans))


@funsearch.evolve
def neh_heuristic(processing_times: np.ndarray) -> list[int]:
    """
    An enhanced initial heuristic for the Permutation Flowshop Scheduling Problem (PFSP).

    This heuristic combines:
    - A weighted scoring for each job based on its total processing time and its maximum processing time.
      The weight parameter alpha balances these two criteria.
    - An iterative insertion procedure that builds an initial sequence.
    - A subsequent local search using pairwise swap improvements to further reduce the makespan.

    The resulting schedule (a list of job indices) is returned.
    """
    import random
    
    def calculate_priority(processing_times: np.ndarray, job: int, alpha: float, beta: float, gamma: float) -> float:
        total_time = processing_times[job].sum()
        max_time = processing_times[job].max()
        median_time = np.median(processing_times[job])
        score = alpha * total_time + beta * max_time + gamma * median_time
        return score

    num_jobs, num_machines = processing_times.shape
    alpha = 0.5
    beta = 0.3
    gamma = 0.2

    job_scores = [
        (job, calculate_priority(processing_times, job, alpha, beta, gamma))
        for job in range(num_jobs)
    ]
    
    # Sort jobs by descending score
    job_scores.sort(key=lambda x: x[1], reverse=True)
    
    sequence = [job_scores[0][0]]
    tabu_list = []
    max_tabu_size = 7

    for job, _ in job_scores[1:]:
        best_makespan = float('inf')
        insertion_options = []

        for pos in range(len(sequence) + 1):
            new_sequence = sequence[:pos] + [job] + sequence[pos:]
            ms = compute_makespan(new_sequence, processing_times)
            insertion_options.append((pos, ms))
        
        insertion_options.sort(key=lambda x: x[1])

        for pos, ms in insertion_options:
            if random.uniform(0, 1) < 0.8 or pos == insertion_options[0][0]:
                sequence.insert(pos, job)
                break

    best_sequence = sequence
    best_makespan = compute_makespan(sequence, processing_times)
    
    def compute_neighbourhood(seq):
        neighbours = []
        for i in range(len(seq) - 1):
            for j in range(i + 1, len(seq)):
                neighbour = seq.copy()
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                neighbours.append(neighbour)
        return neighbours

    for _ in range(200):
        neighbourhood = compute_neighbourhood(sequence)
        found_better = False

        for neighbour in neighbourhood:
            if tuple(neighbour) in tabu_list:
                continue

            current_makespan = compute_makespan(neighbour, processing_times)

            if current_makespan < best_makespan:
                sequence = neighbour
                best_makespan = current_makespan
                tabu_list.append(tuple(neighbour))

                if len(tabu_list) > max_tabu_size:
                    tabu_list.pop(0)

                found_better = True
                break

        if not found_better:
            alpha, beta, gamma = random.random(), random.random(), random.random()
            if alpha + beta + gamma != 1:
                total = alpha + beta + gamma
                alpha /= total
                beta /= total
                gamma /= total

    return best_sequence

