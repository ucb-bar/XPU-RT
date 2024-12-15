import numpy as np
from workload import Workload

def overlap_fixer(workload: Workload, t: np.ndarray, alpha: np.ndarray):
    """
    Resolves overlaps by pushing them forward in time
    @return: updated t that is free of overlaps and respects the precedence constraints
    """
    transfer_times = workload.get_transfer_times()
    for i in range(len(t)):
        for j in range(i+1, len(t)):
            # check if j is predecessor of i and vice versa
            transfer_time = 0
            if workload.operations[i].predecessor == workload.operations[j]:
                machine_pred = np.argmax(alpha[j])
                machine_curr = np.argmax(alpha[i])
                transfer_time = transfer_times[machine_pred][machine_curr]
            elif workload.operations[j].predecessor == workload.operations[i]:
                machine_pred = np.argmax(alpha[i])
                machine_curr = np.argmax(alpha[j])
                transfer_time = transfer_times[machine_pred][machine_curr]

            if t[i] < t[j] and np.argmax(alpha[i]) == np.argmax(alpha[j]):
                if t[i] + workload.operations[i].get_durations()[np.argmax(alpha[i])] + transfer_time > t[j]:
                    t[j] = t[i] + workload.operations[i].get_durations()[np.argmax(alpha[i])] + transfer_time
            elif t[i] > t[j] and np.argmax(alpha[i]) == np.argmax(alpha[j]):
                if t[j] + workload.operations[j].get_durations()[np.argmax(alpha[j])] + transfer_time > t[i]:
                    t[i] = t[j] + workload.operations[j].get_durations()[np.argmax(alpha[j])] + transfer_time

    return t

def count_overlaps(workload: Workload, t: np.ndarray, alpha: np.ndarray):
    """
    @return: number of overlaps in the schedule
    """
    transfer_times = workload.get_transfer_times()
    count = 0

    for i in range(len(t)):
        for j in range(i+1, len(t)):
            # check if j is predecessor of i and vice versa
            transfer_time = 0
            if workload.operations[i].predecessor == workload.operations[j]:
                machine_pred = np.argmax(alpha[j])
                machine_curr = np.argmax(alpha[i])
                transfer_time = transfer_times[machine_pred][machine_curr]
            elif workload.operations[j].predecessor == workload.operations[i]:
                machine_pred = np.argmax(alpha[i])
                machine_curr = np.argmax(alpha[j])
                transfer_time = transfer_times[machine_pred][machine_curr]

            if t[i] < t[j] and np.argmax(alpha[i]) == np.argmax(alpha[j]):
                if t[i] + workload.operations[i].get_durations()[np.argmax(alpha[i])] + transfer_time > t[j]:
                    count += 1
            elif t[i] > t[j] and np.argmax(alpha[i]) == np.argmax(alpha[j]):
                if t[j] + workload.operations[j].get_durations()[np.argmax(alpha[j])] + transfer_time > t[i]:
                    count += 1
    return count