import numpy as np


def evolved_priority(job: np.ndarray) -> float:
    """Improved version of `calc_priority_v0` with more complex logic."""
    total_time = 0.0
    num_jobs = len(job)
    
    # Calculate average time to identify priority
    average_time = np.mean(job)
    
    # Initialize priority score
    priority_score = 0.0
    
    for i in range(num_jobs):
        if job[i] < average_time:
            priority_score += (average_time - job[i]) * 1.5  # Higher weight for jobs below average
        elif job[i] == average_time:
            priority_score += average_time  # Equal weight for average jobs
        else:
            priority_score += (job[i] - average_time) * 0.5  # Lower weight for jobs above average
    
    # Adding a penalty for jobs that exceed a certain threshold
    threshold = np.max(job) * 0.8
    for time in job:
        if time > threshold:
            priority_score -= (time - threshold) * 2  # Hefty penalty for exceeding threshold
    
    # Ensure priority score is non-negative
    if priority_score < 0:
        priority_score = 0
    
    total_time = sum(job) + priority_score
    return total_time

def evolved_priority_2(job: np.ndarray) -> float:
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
