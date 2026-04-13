import numpy as np

class WorkloadGenerator:
    """
    Poisson workload generator used in cloud simulations.
    Generates bursty task arrivals.
    """

    def __init__(self, arrival_rate=2):
        self.lambda_rate = arrival_rate

    def generate_tasks(self):
        num_tasks = np.random.poisson(self.lambda_rate)
        tasks = []

        for _ in range(num_tasks):
            cpu = np.random.uniform(0.1, 0.5)
            mem = np.random.uniform(0.1, 0.5)
            tasks.append((cpu, mem))

        return tasks
