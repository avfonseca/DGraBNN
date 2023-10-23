#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def createConfusionMatrix(y_true,y_pred):

    # Build confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 7))    
    return sns.heatmap(cm, annot=True).get_figure()
