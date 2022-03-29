# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:10:29 2022

@author: Javier
"""

"""
# Credit default prediction
Here's our first attempt at using data to create a table:
"""

import pandas as pd
import streamlit as st

train = pd.read_csv('/input/train.csv')
test = pd.read_csv('/input/test.csv')
train.head()



df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df