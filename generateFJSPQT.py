import numpy as np


def permute_rows(x):
    """
    :param x: np array 2-D
    :return:
    """
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low=1, high=99):
    min_machines = 1
    max_machines = n_m - min_machines

    times_array = np.zeros((n_j, n_m, n_m))
    machine_set = []
    for i in range(n_j):
        job_machines = []
        for j in range(n_m):
            while True:
                num_machines = np.random.randint(min_machines, max_machines + 1)
                machines = np.random.choice(np.arange(1, n_m + 1), num_machines, replace=False)
                machines_set = set(machines)
                if j == 0 or not any(machine in machines_set for machine in job_machines[-1]):
                    job_machines.append(machines_set)
                    break
            base_time = np.random.randint(low+25, high-15)
            for machine in machines:
                times_array[i, j, machine - 1] = base_time + round(np.random.randn() * 5)
        machine_set.append(job_machines)

    lag_time = np.random.randint(low=5, high=15, size=(n_j, n_m))
    queue_time = np.random.randint(low=int(high / 2), high=high * 3, size=(n_j, n_m))
    lag_time[:, -1] = queue_time[:, -1] = 0

    # Create operation-machine correspondence table
    # Rows: machines, Columns: operations (job-process pairs)
    op_machine_table = np.zeros((n_j * n_m, n_m))
    for i in range(n_j):
        for j in range(n_m):
            for machine in machine_set[i][j]:
                op_machine_table[i * n_m + j, machine - 1] = times_array[i, j, machine - 1]

    # Dimensions: [machine][job][operation]
    op_machine_table = times_array.transpose(2, 0, 1)
    # op_machine_table_2d = op_machine_table.transpose(1, 2, 0).reshape(-1, n_m)
    return  lag_time, queue_time, op_machine_table


# 示例用法
if __name__ == '__main__':
    import random
    import torch

    n_j = 5
    n_m = 4
    low = 1
    high = 99
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    lag_time, q_time, op_ma_table = uni_instance_gen(n_j, n_m, low, high)
    print(q_time.shape, lag_time.shape, op_ma_table.shape)

    print("\nQueue time：\n", q_time)
    print("\nLag time：\n", lag_time)
    print("\nOperation-Machine Correspondence Table (Rows: Machines, Columns: Jobs-Operations):")
    print("    ", end="")
    for i in range(n_j):
        for j in range(n_m):
            print(f"J{i + 1}P{j + 1} ", end="")
    print()
    for m in range(n_m):
        print(f"M{m + 1}: ", end="")
        for val in op_ma_table[m].reshape(-1):
            print(f"{val:4.0f} ", end="")
        print()