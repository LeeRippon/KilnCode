import pandas as pd
import numpy as np
from time import time 

def load():
    print('Loading kiln data...')
    t1 = time()
    df = pd.read_csv('/content/drive/MyDrive/Invoice2_Deliverables/d_Eastover_B1_Aug31_Avg.csv') 
    l = len(df)
    t2 = time()
    print('Loading time is', t2-t1, 'seconds')
    return df, l