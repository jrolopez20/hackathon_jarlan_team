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

# Reading datasets
home = os.getcwd()
with open(os.path.join(home,'input','df_falta_agua.pickle'), 'rb') as handle:
    df_lack_water = pickle.load(handle)
with open(os.path.join(home,'input','df_fuga_agua.pickle'), 'rb') as handle:
    df_water_leak = pickle.load(handle)

# Sidebar
st.sidebar.write("""
	# *Smart Water Management*
	Describes the current situation of the water supply in Chihuahua and how to contribute to beat this problem
	""")

st.sidebar.title('Interactive options:')

st.write("""# Actual situation""")  
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


hist = getData(pd.Timestamp('2018-06-01'), pd.Timestamp('2019-06-01'), pd.DateOffset(days=1), df_lack_water)
dict_hist = {'Date':[reg[0] for reg in hist]
            ,'Total calls':[reg[1] for reg in hist]
            ,'Zone 0':[reg[2][0] for reg in hist]
            ,'Zone 1':[reg[2][1] for reg in hist]
            ,'Zone 2':[reg[2][2] for reg in hist]
            ,'Zone 3':[reg[2][3] for reg in hist]
            ,'Zone 4':[reg[2][4] for reg in hist]
            ,'Zone 5':[reg[2][5] for reg in hist]
            }
df_hist = pd.DataFrame(dict_hist)

fig = px.line(df_hist, x='Date', y=df_hist.columns)

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=14, label="2w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(step="all")
        ])
        ,bgcolor="#555"
        ,activecolor="#777"
    )

)

# fig = px.line(df_hist, x='Date', y="Total calls")
# fig = px.area(df_hist, facet_col="company", facet_col_wrap=2)
# fig = px.area(df, facet_col="company", facet_col_wrap=2)
st.plotly_chart(fig, use_container_width=True)



"""# Solution"""
"""With this work we offer a possible solution to efficiently mitigate and prevent all water leaks and contribute to its conservation."""


"""TODO Here we will start displaying chart as part of our solution"""

# Sidebar footer
st.sidebar.write("""
	Developed by **Jarlan Team** \n
	*© 2022 [Universidad Autónoma de Chihuahua](https://uach.mx)*
	""")