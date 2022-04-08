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
<<<<<<< HEAD

with open(os.path.join(home, 'input', 'df_falta_agua.pickle'), 'rb') as handle:
=======
with open(os.path.join(home,'input','df_falta_agua.pickle'), 'rb') as handle:
>>>>>>> f0aea084ad7b898c28848e1c44f4e53142617334
    df_lack_water = pickle.load(handle)
with open(os.path.join(home, 'input', 'df_fuga_agua.pickle'), 'rb') as handle:
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
    """
    As tends to be the case in large, developing countries, Mexico is a nation of notable meteorological, hydrographic and social contrasts throughout its territory, which impact the various population strata in different ways. The public administration in Mexico is divided into federal (nationwide), state and municipal levels. In this sense, it is desirable to have water security metrics not only for the country as a whole but also for each state.\n
    The present work seeks to show this problem but focused on Chihuahua and how these contrasts create different water-security scenarios using pertinent indices.
    """

with col2:
    st.image(
        'https://www.siliconrepublic.com/wp-content/uploads/2017/07/Leaking-pipe-718x523.jpg'
    )

    st.image(
        'https://www.theyucatantimes.com/wp-content/uploads/2020/10/agua-chihuahua.jpg'
    )

"""# So what is the problem?"""

"""## How to prevent water leaks efficiently"""

"""Based on the data collected by JMAS (Junta Municipal de Agua y Saneamiento de Chihuahua) on leak reports as well as the history of breakage records in the city of Chihuahua during the period from June 2018 to May 2019, a study can be done to analyze this data and be able to identify possible solutions."""

"""In order to go into the matter here we present you some records of lack of water in the period mentioned above."""
hist = getData(pd.Timestamp('2018-06-01'),
               pd.Timestamp('2019-06-01'), pd.DateOffset(days=1), df_lack_water)
dict_hist = {'Date': [reg[0] for reg in hist], 'Total calls': [reg[1] for reg in hist], 'Zone 0': [reg[2][0] for reg in hist], 'Zone 1': [reg[2][1] for reg in hist], 'Zone 2': [reg[2][2] for reg in hist], 'Zone 3': [reg[2][3] for reg in hist], 'Zone 4': [reg[2][4] for reg in hist], 'Zone 5': [reg[2][5] for reg in hist]
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
        ]), bgcolor="#555", activecolor="#777"
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
