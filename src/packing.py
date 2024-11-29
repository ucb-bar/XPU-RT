from workload import Workload, Job, Operation, Window

def greedy_packing(workload: Workload) -> list[Window]:
    """
    Greedy packing algorithm that packs jobs into windows.
    """
    windows = []
    for job in workload.get_jobs():
        for operation in job.get_operations():
            for window in windows:
                if window.can_add_operation(operation):
                    window.add_operation(operation)
                    break
            else:
                window = Window()
                window.add_operation(operation)
                windows.append(window)
    return windows