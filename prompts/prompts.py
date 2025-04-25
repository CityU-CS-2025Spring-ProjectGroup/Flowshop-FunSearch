

# Base prompt given by the demonstration example
base_prompt = (
    "Complete a different and more complex Python function."
    "Be creative and you can insert multiple if-else and for-loop in the code logic."
    "Only output the Python code, no descriptions."
)


# Simple modification made on base prompt for NEH evolvement
flowshop_base_prompt_intruct_Open_ended = (
    "Improve the scheduling heuristic to minimize makespan."  
    "You can change how jobs are ordered or inserted,"
    "Be creative. Think beyond NEH logic."
    "Please only generate neh_heuristic(processing_times: np.ndarray) function"  
    "Use loops, conditionals, or clustering ideas. Only return valid Python code."  
    "Improve the scheduling heuristic to minimize makespan."
    "You can change how jobs are ordered or inserted,"
    "Be creative. Think beyond NEH logic."
    "Please only generate neh_heuristic(processing_times: np.ndarray) function"
    "Use loops, conditionals, or clustering ideas. Only return valid Python code."
)

flowshop_base_prompt_intruct_Orientation = (
    """Improve the NEH heuristic for PFSP to minimize makespan. Focus on:"
      1. **Scoring Strategy**: Modify how jobs are prioritized (e.g., dynamic alpha, machine load balancing).
      2. **Insertion Logic**: Optimize the position selection during insertion (e.g., early termination if no improvement).
      3. **Local Search**: Replace the swap-based search with more efficient methods (e.g., 3-opt, tabu search).
      4. **Constraints**:
        - Preserve job uniqueness (no duplicates).
        - Only use `compute_makespan` for evaluation.
        - Return a list of job indices (e.g., [0, 2, 1]).
      Generate *only* the `neh_heuristic` function body (no duplicate code from skeleton).
       Only return python code.
  """
)



flowshop_base_prompt_rag = ("""
    You are designing a scheduling heuristic for the Permutation Flowshop Scheduling Problem (PFSP) to minimize makespan.
    To assist you, here is background knowledge about a powerful heuristic known as **NEH** (Nawaz–Enscore–Ham):
    ----------------------------
    [NEH Algorithm Knowledge Base]
    1. **NEH Core Idea**:
    - For each job, compute its total processing time across all machines.
    - Sort jobs in descending order of total processing time.
    - Build the schedule iteratively by inserting each job into the current partial sequence at the position that minimizes makespan.
    2. **Example**:
    - If sorted job order is [2, 0, 1], we start with [2], then try inserting job 0 into all positions: [0,2], [2,0], and pick the one with the smallest makespan.
    - Repeat for job 1, trying all positions in the current sequence.
    3. **Possible Enhancements**:
    - Replace total processing time with weighted score (e.g., max(machine-wise time), early-machine dominance).
    - Prune insertion positions that are unlikely to improve makespan.
    - Use a local search after initial construction (e.g., 2-opt, job reordering).
    ----------------------------
    Now, imagine you are analyzing a **highly optimized black-box scheduler** whose logic is unknown.
    Your task is to **reverse-engineer its internal heuristic**, assuming it is based on some modification of NEH.
    Reconstruct the likely job scoring strategy, insertion policy, and local refinement steps it may be using.
    **Constraints**:
    - Use `compute_makespan(schedule, processing_times)` for evaluation only.
    - Ensure schedule is a valid permutation (no job repetition).
    - Only output the body of `neh_heuristic(processing_times: np.ndarray)` in valid Python code.
    """
)
