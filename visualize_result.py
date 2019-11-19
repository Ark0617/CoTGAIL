import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np
results = pu.load_results('~/CoTGAIL/log/')
print(len(results))
pu.plot_results(results, average_group=True, split_fn=lambda _: '')
#print(np.cumsum(results[0].monitor.l))
#plt.plot(np.cumsum(results[0].monitor.l), pu.smooth(results[0].monitor.r, radius=10))
#plt.show()
