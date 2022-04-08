# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st
import pickle5 as pickle
import os
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
from helper import getData
from datetime import datetime

# Import datasets
# df_lack_water = pd.read_csv("./input/df_falta_agua.csv")
# df_water_leak = pd.read_csv("./input/df_fuga_agua.csv")

home = os.getcwd()

with open(os.path.join(home,'input','df_falta_agua.pickle'), 'rb') as handle:
    df_lack_water = pickle.load(handle)
with open(os.path.join(home,'input','df_fuga_agua.pickle'), 'rb') as handle:
    df_water_leak = pickle.load(handle)

st.sidebar.write("""
	# *Smart Water Management*
	Describes the current situation of the water supply in Chihuahua and how to contribute to beat this problem
	""")

st.sidebar.title('Interactive options:')

"""# Actual situation"""
col1, col2 = st.columns(2)

with col1:
    st.write('Some critical text goes here.')
    st.write('Lorem ipsum dolor sit amet consectetur adipiscing elit viverra mauris, taciti dapibus nec id at dictumst montes sem, praesent proin lacinia senectus aliquam et malesuada diam. In iaculis sociosqu urna conubia habitasse nam, habitant id nec dis vehicula proin, tempus nascetur varius volutpat dignissim. Vulputate volutpat erat venenatis nam augue conubia maecenas, nostra tempus donec montes pellentesque tincidunt, justo morbi egestas senectus eleifend iaculis')

with col2:
    st.image(
        'https://www.siliconrepublic.com/wp-content/uploads/2017/07/Leaking-pipe-718x523.jpg')

"""# So what is the problem?"""
"""## How to prevent water leaks efficiently"""


# TODO, just for testing purpose
c21, c22 = st.columns(2)
c21.dataframe(df_lack_water.columns)
c22.dataframe(df_water_leak.columns)

# st.dataframe(df_lack_water.head())
# st.dataframe(df_water_leak.head())

cf, ct = st.columns(2)
From_str = cf.date_input('From', datetime.strptime('2018-06-01', '%Y-%m-%d'), datetime.strptime('2018-06-01', '%Y-%m-%d'), datetime.strptime('2019-06-01', '%Y-%m-%d'))
To_str = ct.date_input('To', datetime.strptime('2019-06-01', '%Y-%m-%d'), datetime.strptime('2018-06-01', '%Y-%m-%d'), datetime.strptime('2019-06-01', '%Y-%m-%d'))

# Uncoment this line
hist = getData(pd.Timestamp(From_str), pd.Timestamp(To_str), pd.DateOffset(weeks=2), df_lack_water)
dict_hist = {'Fecha':[reg[0] for reg in hist]
            ,'Total':[reg[1] for reg in hist]
            ,'Zona 0':[reg[2][0] for reg in hist]
            ,'Zona 1':[reg[2][1] for reg in hist]
            ,'Zona 2':[reg[2][2] for reg in hist]
            ,'Zona 3':[reg[2][3] for reg in hist]
            ,'Zona 4':[reg[2][4] for reg in hist]
            ,'Zona 5':[reg[2][5] for reg in hist]
            }
df_hist = pd.DataFrame(dict_hist)

fig = px.line(df_hist, x='Fecha', y="Total")
st.plotly_chart(fig, use_container_width=True)



"""# Solution"""
"""With this work we offer a possible solution to efficiently mitigate and prevent all water leaks and contribute to its conservation."""


"""TODO Here we will start displaying chart as part of our solution"""

# Sidebar footer
st.sidebar.write("""
	Developed by **Jarlan Team** \n
	*© 2022 [Universidad Autónoma de Chihuahua](https://uach.mx)*
	""")