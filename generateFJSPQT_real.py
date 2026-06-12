

import numpy as np
import random
from typing import Tuple, List, Set, Dict
from math import ceil



PROCESS_TIME_RANGES_MIN: Dict[int, Tuple[int, int]] = {
    0: (12, 30),
    1: (8, 25),
    2: (5, 20),
    3: (90, 360),
    4: (20, 60),
    5: (30, 180),
    6: (8, 30),
    7: (5, 15),
}


DEFAULT_LAG_RANGE_MIN = (5, 15)


LAG_RANGE_MAP_MIN: Dict[Tuple[int, int], Tuple[int, int]] = {
    (0, 1): (8, 18),
    (1, 2): (5, 15),
    (2, 3): (8, 18),
    (3, 4): (8, 18),
    (3, 5): (8, 18),
    (2, 4): (6, 16),
    (4, 5): (8, 18),
    (0, 5): (8, 18),
    (5, 4): (8, 18),
    (4, 6): (6, 16),
    (1, 6): (6, 16),
    (5, 6): (6, 16),
    (6, 2): (5, 15),
    (7, 2): (5, 15),
}


ENFORCE_PROC_VS_LAG: bool = True
PROC_TO_LAG_MIN_RATIO: float = 1.0


FIRST_OP_PRIORS: Dict[int, float] = {
    2: 0.25,
    0: 0.20,
    1: 0.18,
    5: 0.15,
    4: 0.10,
    3: 0.06,
    6: 0.04,
    7: 0.02,
}


TRANSITION_WEIGHTS: Dict[int, Dict[int, float]] = {

    0: {1: 0.45, 2: 0.20, 5: 0.20, 7: 0.10, 4: 0.05},
    1: {2: 0.60, 6: 0.10, 4: 0.10, 7: 0.10, 5: 0.10},
    2: {3: 0.35, 4: 0.25, 5: 0.20, 0: 0.10, 2: 0.10},
    3: {4: 0.50, 5: 0.50},
    4: {5: 0.45, 6: 0.30, 1: 0.10, 2: 0.15},
    5: {4: 0.40, 6: 0.25, 2: 0.20, 0: 0.15},
    6: {2: 0.80, 4: 0.10, 5: 0.10},
    7: {2: 0.70, 4: 0.10, 0: 0.10, 5: 0.10},
}


def _sample_qtime_minutes(curr_idx: int, prev_idx: int, next_idx: int) -> int:





    if curr_idx == 6:
        return 10**4
    if curr_idx == 0:
        return 720 if random.random() < 0.8 else int(np.random.uniform(120, 1440))
    if curr_idx == 1:
        return 480 if random.random() < 0.7 else int(np.random.uniform(120, 480))
    if curr_idx == 3:
        if random.random() < 0.22:
            return 480 if random.random() < 0.7 else int(np.random.uniform(300, 720))
        return 10**4
    if curr_idx == 5:
        if random.random() < 0.5:
            r = random.random()
            if   r < 0.6: return 480
            elif r < 0.8: return 720
            else:         return int(np.random.uniform(120, 480))
        return 10**4
    if curr_idx == 2:
        if random.random() < 0.5:
            r = random.random()
            if   r < 0.5: return 180
            elif r < 0.8: return 480
            elif r < 0.9: return 720
            else:         return int(np.random.uniform(60, 720))
        return 10**4
    if curr_idx == 7 and prev_idx == 1:
        if random.random() < 0.1:
            return 150
        return 10**4
    return 10**4


def assign_capa_groups(
    n_capa: int,
    n_m: int,
    all_machines: List[int],
    process_types: Dict[int, Tuple[int, int]],
    min_machines: int = 1,
    balance_factor: float = 0.5
) -> List[Set[int]]:







    if n_capa < 0 or n_m < 0 or min_machines < 1 or not (0 <= balance_factor <= 1):
        raise ValueError("n_capa,n_m需非负；min_machines≥1；balance_factor∈[0,1]")


    groups: List[Set[int]] = [set() for _ in range(n_capa)]


    if n_m >= 2 * n_capa:

        avg_machines = ceil(n_m / max(n_capa, 1))
        max_cap = max(min_machines, int(avg_machines * (1 + balance_factor)))


        sizes = [min_machines for _ in range(n_capa)]
        left = n_m - min_machines * n_capa


        order = list(range(n_capa))
        random.shuffle(order)
        idx = 0
        while left > 0:
            g = order[idx % n_capa]
            if sizes[g] < max_cap:
                sizes[g] += 1
                left -= 1
            idx += 1

            if idx % (n_capa * 3) == 0:
                random.shuffle(order)


        start = 0
        for g in range(n_capa):
            take = sizes[g]
            groups[g] = set(all_machines[start:start + take])
            start += take
        return groups



    active = min(n_capa, int(n_m / 2))





    chosen_groups = random.sample(range(n_capa), k=active)


    sizes = [0 for _ in range(n_capa)]
    for g in chosen_groups:
        sizes[g] = min_machines


    left = n_m - active * min_machines
    while left > 0:
        g = random.choice(chosen_groups)
        sizes[g] += 1
        left -= 1


    start = 0
    for g in range(n_capa):
        take = sizes[g]
        if take > 0:
            groups[g] = set(all_machines[start:start + take])
            start += take

    return groups


def _choose_with_weights(candidates: List[int], weights_map: Dict[int, float]) -> int:






    if not candidates:
        raise ValueError("候选集合为空，无法采样")
    cands = list(candidates)

    w = np.array([weights_map.get(g, 0.0) for g in cands], dtype=float)
    if np.all(w <= 0):
        p = np.ones(len(cands), dtype=float) / len(cands)
    else:
        p = w / w.sum()
    return int(np.random.choice(cands, p=p))


def uni_instance_gen(n_j: int, n_m: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
















    process_time_ranges = PROCESS_TIME_RANGES_MIN
    all_machines: List[int] = list(range(1, n_m + 1))
    np.random.shuffle(all_machines)
    n_capa: int = 8


    capability_groups = assign_capa_groups(
        n_capa=n_capa, n_m=n_m, all_machines=all_machines,
        process_types=process_time_ranges,
        min_machines=1, balance_factor=0.5
    )


    total_assigned = sum(len(s) for s in capability_groups)
    if total_assigned != n_m:
        raise ValueError(f"能力组分配异常：应分配机台数={n_m}，实际合计={total_assigned}")

    union_all = set().union(*capability_groups) if capability_groups else set()
    if len(union_all) != n_m:

        raise ValueError("能力组分配异常：发现重复机台或缺失机台。")

    nonempty_groups = [g for g in range(n_capa) if len(capability_groups[g]) > 0]


    capability_assignments: np.ndarray = np.full((n_j, n_m), -1, dtype=int)

    for i in range(n_j):

        pool0 = random.sample(nonempty_groups, k=len(nonempty_groups))
        first_choice = _choose_with_weights(pool0, FIRST_OP_PRIORS)
        capability_assignments[i, 0] = first_choice


        for j in range(1, n_m):
            prev_c = capability_assignments[i, j - 1]
            pool = random.sample(nonempty_groups, k=len(nonempty_groups))
            candidates = [g for g in pool if g != prev_c] or pool
            trans_map = TRANSITION_WEIGHTS.get(prev_c, {})
            chosen = _choose_with_weights(candidates, trans_map)
            capability_assignments[i, j] = chosen


    times_array: np.ndarray = np.zeros((n_j, n_m, n_m), dtype=int)
    for i in range(n_j):
        prev_set = None
        for j in range(n_m):
            capa_idx = capability_assignments[i, j]
            avail = list(capability_groups[capa_idx])
            if not avail:

                raise ValueError(f"内部错误：能力组 {capa_idx} 无机台，但被选中。")
            random.shuffle(avail)


            max_attempts = 100
            for _ in range(max_attempts):
                k = np.random.randint(1, len(avail) + 1)
                chosen = set(np.random.choice(avail, k, replace=False))
                if prev_set is None or chosen != prev_set:
                    break
            else:
                chosen = {avail[0]}
            prev_set = chosen


            lo, hi = process_time_ranges[capa_idx]
            for m_id in chosen:
                proc = int(np.random.uniform(lo, hi))
                times_array[i, j, m_id - 1] = proc


    lag_time: np.ndarray = np.zeros((n_j, n_m), dtype=int)
    q_time:  np.ndarray = np.zeros((n_j, n_m), dtype=int)

    for i in range(n_j):
        for j in range(n_m):
            prev_idx = capability_assignments[i, j - 1] if j > 0 else -1
            curr_idx = capability_assignments[i, j]
            next_idx = capability_assignments[i, j + 1] if j < n_m - 1 else -1


            if prev_idx >= 0:
                lo, hi = LAG_RANGE_MAP_MIN.get((prev_idx, curr_idx), DEFAULT_LAG_RANGE_MIN)
            else:
                lo, hi = DEFAULT_LAG_RANGE_MIN
            lag_time[i, j] = np.random.randint(lo, hi + 1)


            q_time[i, j] = 0 if next_idx < 0 else _sample_qtime_minutes(curr_idx, prev_idx, next_idx)


        lag_time[i, -1] = 0
        q_time[i,  -1]  = 0


    if ENFORCE_PROC_VS_LAG and PROC_TO_LAG_MIN_RATIO > 0:
        for i in range(n_j):
            for j in range(n_m):
                need = int(np.ceil(PROC_TO_LAG_MIN_RATIO * lag_time[i, j]))
                if need <= 0:
                    continue

                nz = np.where(times_array[i, j, :] > 0)[0]
                for m_idx in nz:
                    if times_array[i, j, m_idx] < need:
                        times_array[i, j, m_idx] = need


    op_machine_table = times_array.transpose(2, 0, 1)


    return lag_time, q_time, op_machine_table



if __name__ == "__main__":

    n_j: int = 5
    n_m: int = 9


    random.seed(42)
    np.random.seed(42)


    lag_time, q_time, op_ma_table = uni_instance_gen(n_j, n_m)


    print("加工时间（不同机台，单位：分钟）：\n", op_ma_table)
    print("排队时间（Q-time，单位：分钟）：\n", q_time)
    print("滞后时间（Lag-time，单位：分钟）：\n", lag_time)
