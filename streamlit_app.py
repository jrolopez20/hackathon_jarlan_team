# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

st.sidebar.write("""
	# *Smart Water Management*
	Project description goes here 
	""")
	
st.sidebar.title('Interactive options:')

st.sidebar.write("""
	Developed by **Jarlan Team** \n
	*© 2022 [Universidad Autónoma de Chihuahua](https://uach.mx)*
	""")

st.title('TODO')

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40],
  'third column': [100, 200, 300, 400],
})

df
