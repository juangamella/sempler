import sempler
import sempler.generators
import numpy as np

rng = np.random.default_rng(42)
data = [rng.uniform(size=(100, 5)) for _ in range(2)]
graph = sempler.generators.dag_avg_deg(p=5, k=2, random_state=42)
network = sempler.DRFNet(graph, data)
print(network.graph)

sample = network.sample()
print(sample)
