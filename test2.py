import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
from scaling_policy import ScalingPolicy

data = {
    'latency': ['low', 'low', 'medium', 'medium', 'medium', 'high', 'high'],
    'metric_var': [0.1, 0.2, 0.6, 0.4, 0.6, 0.8, 0.7],
    'scale': [1, 2, 3, 4, 5, 6, 7]
}
prediction_df = pd.DataFrame(data)

policy = ScalingPolicy(scale_domain=data["scale"])

for l in ["low", "medium", "high"]:
    for i in range(3):
        control_scale, _ = policy.max_unc_scale(prediction_df, "metric", latency=l, level=i+1)
        print(prediction_df)
        print(l, "highest ", i+1, control_scale)
