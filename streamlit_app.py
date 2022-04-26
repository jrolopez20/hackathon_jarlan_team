# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
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
from sklearn import svm, datasets

# Reading datasets
home = os.getcwd()

st.set_page_config(
     page_title="Smart Water Management in Chihuahua",
    #  page_icon="🧊",
    #  layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'About': "This project tend to describes the current situation of the water supply in Chihuahua and how to contribute to beat this problem"
     }
 )

with open(os.path.join(home, 'input', 'df_falta_agua_v2.pickle'), 'rb') as handle:
    df_lack_water = pickle.load(handle)
with open(os.path.join(home, 'input', 'df_fuga_agua_v2.pickle'), 'rb') as handle:
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
        'https://www.siliconrepublic.com/wp-content/uploads/2017/07/Leaking-pipe-718x523.jpg',
         caption='Fig 1. Avería'
    )

    st.image(
        'https://www.theyucatantimes.com/wp-content/uploads/2020/10/agua-chihuahua.jpg',
         caption='Fig 2. Persona manifestándose'
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

# Mapping data
df_lack_water.rename(columns = {'lng':'lon'}, inplace = True)
df_water_leak.rename(columns = {'lng':'lon'}, inplace = True)
st.map(df_lack_water.sample(300))

"""# %s""" % translation['solution']

"""%s""" % translation['solution1']

"""A partir de la información recogida por los sensores en cada una de las estaciones de bombeo que se 
encuentran distribuidas en la ciudad como se puede ver en la Fig 4, es posible obtener un grafo como el
de la Fig 4 donde se muestra la relación entre cada una de estas estaciones"""

cols1, cols2 = st.columns(2)
with cols1:
    st.image(
        './resources/images/tanks.png',
        caption='Fig 3. Tanks'
    )

with cols2:
    st.image(
        './resources/images/graph.png',
        caption='Fig 4. Graph'
    )

"""Luego teniendo el cuenta el nivel de llenado de cada uno de los tanques así como los reportes de llamdas
que se producen en la ciudad como se pueden ver en el mapa y haciendo uso de herramientas de aprendizaje de 
maquina es posible hacer predicciones con anterioridad de posibles averías incluso antes de que se produzca 
una llamada de reporte."""

"""### Machine Learning:"""

st.image(
        './resources/images/big-picture.png',
        caption='Fig 5. Representación gráfica de la solución haciendo uso de técnicas de Machine learning'
    )

"""Y a partir de este herramienta que sea capás de identificar además la zona donde se produjo dicha avería 
a fin de que se puedan poner en práctica los procedimientos para darle solución en tiempo."""

st.image(
        './resources/images/zones.png',
        caption='Fig 6. Distribución por zonas de abastecimiento de agua'
    )

"""TODO Here we will start displaying chart as part of our solution"""

# Machine learning models
# modelSVM = svm.SVR(kernel='rbf', gamma=0.7, C=5.0, epsilon=0.6)
modelSVM = svm.SVC(kernel='rbf', gamma=0.6, C=2.0)

# Testing with iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

modelSVM.fit(X,y)
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
fig = plt.axes()
plot_contours(fig, modelSVM, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
fig.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
fig.set_xlim(xx.min(), xx.max())
fig.set_ylim(yy.min(), yy.max())
fig.set_xlabel("Sepal length")
fig.set_ylabel("Sepal width")
fig.set_xticks(())
fig.set_yticks(())
fig.set_title('title')

st.pyplot(plt)
