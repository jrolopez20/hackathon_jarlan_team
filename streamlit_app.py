# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import pickle5 as pickle
import os
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import json
from helper import getData
from datetime import datetime

# Reading datasets
home = os.getcwd()

with open(os.path.join(home, 'input', 'df_falta_agua.pickle'), 'rb') as handle:
    df_lack_water = pickle.load(handle)
with open(os.path.join(home, 'input', 'df_fuga_agua.pickle'), 'rb') as handle:
    df_water_leak = pickle.load(handle)

# Sidebar
st.sidebar.write("""# *Smart Water Management* """)

language = st.sidebar.selectbox('', ('EN', 'ES'))

with open('./languages/%s.json' % language, 'r', encoding="utf-8") as translation_file:    
    translation = json.load(translation_file)

st.sidebar.write(translation['project_description'])

# Sidebar footer
st.sidebar.write("""
	%s **Jarlan Team** \n
	*© 2022 [Universidad Autónoma de Chihuahua](https://uach.mx)*
	""" % translation['developed_by'])


st.write("""# %s""" % (translation['actual_situation']))
col1, col2 = st.columns(2)
with col1:
    """%s""" % translation['introduction1']

with col2:
    st.image(
        'https://www.siliconrepublic.com/wp-content/uploads/2017/07/Leaking-pipe-718x523.jpg'
    )

    st.image(
        'https://www.theyucatantimes.com/wp-content/uploads/2020/10/agua-chihuahua.jpg'
    )

    """%s""" % translation['introduction2']
    
"""# %s""" % translation['what_is_the_problem']

"""## %s""" % translation['problem_to_solve']

"""%s""" % translation['problem_description1']

"""%s""" % translation['problem_description2']

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

"""# %s""" % translation['solution']

"""%s""" % translation['solution1']

"""TODO Here we will start displaying chart as part of our solution"""


