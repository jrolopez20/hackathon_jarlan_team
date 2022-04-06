# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st
from helper import getData

# Import datasets
df_lack_water = pd.read_csv("./input/df_falta_agua.csv")
df_water_leak = pd.read_csv("./input/df_fuga_agua.csv")

st.sidebar.write("""
	# *Smart Water Management*
	Describes the current situation of the water supply in Chihuahua and how to contribute to beat this problem
	""")

st.sidebar.title('Interactive options:')

st.sidebar.write("""
	Developed by **Jarlan Team** \n
	*© 2022 [Universidad Autónoma de Chihuahua](https://uach.mx)*
	""")

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

From = pd.Timestamp('2018-06-01')
To = pd.Timestamp('2019-06-01')

# TODO, just for testing purpose
c21, c22 = st.columns(2)
c21.dataframe(df_lack_water.head())
# c22.write(df_water_leak.columns)

# Uncoment this line
# hist = getData(From, To, pd.DateOffset(weeks=2), df_lack_water)



"""# Solution"""
"""With this work we offer a possible solution to efficiently mitigate and prevent all water leaks and contribute to its conservation."""


"""TODO Here we will start displaying chart as part of our solution"""
