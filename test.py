import numpy as np
import math
import pandas as pd

data = [1, 3, 3, 5, 5, 5, 7]
bins = [0, 2, 4, 6, 8]
pdf, edges = np.histogram(data, bins=bins, density=True)
print(pdf)
print(edges)
