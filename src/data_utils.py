import pandas as pd
import numpy as np

def replace_abnormal_values(data:pd.Series, lower_bound, upper_bound, verbose=1):
    data = data.astype(float)
    mask_abnormal = (data < lower_bound) | (data > upper_bound)
    data[mask_abnormal] = np.nan
    if hasattr(data, 'name') and verbose > 0:
        print(f"{str(data.name):>20} | {mask_abnormal.sum()} ({mask_abnormal.sum()/len(data)*100:.2f}%)\t abnormal couunts")
    return data