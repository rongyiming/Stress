import pandas as pd
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt


file = './dataset/CLAS_32Hz.pkl'

with open(file, 'rb') as f:
    data = pickle.load(f)

print(data)