# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import pandas as pd
import sklearn
import streamlit as st
import pickle5 as pickle
import os
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import json
from helper import getData
from sklearn import svm, datasets
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import random

import base64

# Reading datasets
home = os.getcwd()

st.set_page_config(
     page_title="Smart Water Management in Chihuahua",
    #  page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'About': "This project tend to describes the current situation of the water supply in Chihuahua and how to contribute to beat this problem"
     }
 )

with open(os.path.join(home, 'input', 'df_falta_agua_v2.pickle'), 'rb') as handle:
    df_lack_water = pickle.load(handle)
with open(os.path.join(home, 'input', 'df_fuga_agua_v2.pickle'), 'rb') as handle:
    df_water_leak = pickle.load(handle)
with open(os.path.join(home, 'input', 'df_tanks.pickle'), 'rb') as handle:
    df_tanks = pickle.load(handle)

df_target5zone = pd.read_csv("./input/target1Zone.csv")
df_target5zone['timestamp'] = df_target5zone['timestamp'].apply(lambda x: pd.Timestamp(x))
# df_target5zone['callSum'] = df_target5zone['callSum'].apply(lambda x: x.strip(']['))
# df_target5zone['callSum'] = df_target5zone['callSum'].apply(lambda x: x.split(', '))
# df_target5zone['callSum'] = df_target5zone['callSum'].apply(lambda x: list(map(int, x)))

# df_tanks = df_tanks.sample(5)

# st.dataframe(df_target5zone)
# st.dataframe(df_tanks)

# Sidebar
st.sidebar.write("""# *Smart Water Management* """)

language = st.sidebar.selectbox('Select language', ('EN', 'ES'))

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
    
    """%s""" % translation['introduction2']

with col2:
    st.image(
        'https://www.siliconrepublic.com/wp-content/uploads/2017/07/Leaking-pipe-718x523.jpg',
         caption='Fig 1. Avería'
    )

    # st.image(
    #     'https://www.theyucatantimes.com/wp-content/uploads/2020/10/agua-chihuahua.jpg',
    #      caption='Fig 2. Persona manifestándose'
    # )

    
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
        ]), bgcolor="#222", activecolor="#222"
    )
)


# fig = px.line(df_hist, x='Date', y="Total calls")
# fig = px.area(df_hist, facet_col="company", facet_col_wrap=2)
# fig = px.area(df, facet_col="company", facet_col_wrap=2)
st.plotly_chart(fig, use_container_width=True)

"""%s""" % translation['map_text']

# Mapping data
df_lack_water.rename(columns = {'lng':'lon'}, inplace = True)
# df_water_leak.rename(columns = {'lng':'lon'}, inplace = True)
st.map(df_lack_water.sample(300))

"""# %s""" % translation['solution']

"""%s""" % translation['solution1']

"""%s""" % translation['solution2']

cols1, cols2 = st.columns(2)
with cols1:
    st.image(
        './resources/images/tanks.png',
        caption='Fig 2. Tanks distribution in the city'
    )

with cols2:
    st.image(
        './resources/images/graph.png',
        caption='Fig 3. Relation between each tanks'
    )

"""%s""" % translation['solution3']

"""### Machine Learning:"""
s1, s2 = st.columns(2)
with s1:
    st.image(
        './resources/images/big-picture.png',
        caption='Fig 4. Graphic representation of the solution using Machine learning techniques'
    )

with s2:
    """%s""" % translation['solution4']

# Machine learning models
# modelSVM = svm.SVR(kernel='rbf', gamma=0.7, C=5.0, epsilon=0.6)


X = df_tanks[df_tanks['datetime'] >= pd.Timestamp('2018-06-07')]
X = X[X['datetime'] <= pd.Timestamp('2019-01-08')]
index = pd.DatetimeIndex(X['datetime'])
X = X.iloc[index.indexer_between_time('8:00','21:00')]
X = X.drop(columns = ['datetime'])
X = X.reset_index(drop = True)
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)
# st.write(X)

y = df_target5zone[df_target5zone['timestamp'] >= pd.Timestamp('2018-06-07')]
y = y[y['timestamp'] <= pd.Timestamp('2019-01-08')]
y = y.iloc[index.indexer_between_time('8:00','21:00')]
y = y.drop(columns = ['timestamp'])
y = y.reset_index(drop = True)
# st.write(y)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.3)

# st.write(X_train.columns)
# st.write(y_train)

model = ExtraTreesClassifier(criterion='entropy')
multi_target_model = MultiOutputClassifier(model)
multi_target_model.fit(X_train, y_train)

test_pred_model = multi_target_model.predict(X_test)

# st.write(sklearn.metrics.classification_report(y_test, test_pred_model))

# st.pyplot(ConfusionMatrixDisplay(confusion_matrix(y_test,test_pred_model)).plot())

# st.write(sklearn.metrics.accuracy_score(y_test, test_pred_model))
# st.write(y_train)


# Testing with iris dataset
# iris = datasets.load_iris()
# X = iris.data[:, :2]
# y = iris.target

# def make_meshgrid(x, y, h=0.02):
#     """Create a mesh of points to plot in

#     Parameters
#     ----------
#     x: data to base x-axis meshgrid on
#     y: data to base y-axis meshgrid on
#     h: stepsize for meshgrid, optional

#     Returns
#     -------
#     xx, yy : ndarray
#     """
#     x_min, x_max = x.min() - 1, x.max() + 1
#     y_min, y_max = y.min() - 1, y.max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     return xx, yy

# def plot_contours(ax, clf, xx, yy, **params):
#     """Plot the decision boundaries for a classifier.

#     Parameters
#     ----------
#     ax: matplotlib axes object
#     clf: a classifier
#     xx: meshgrid ndarray
#     yy: meshgrid ndarray
#     params: dictionary of params to pass to contourf, optional
#     """
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out

# modelSVM.fit(X,y)
# X0, X1 = X[:, 0], X[:, 1]
# xx, yy = make_meshgrid(X0, X1)
# fig = plt.axes()
# plot_contours(fig, modelSVM, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# fig.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
# fig.set_xlim(xx.min(), xx.max())
# fig.set_ylim(yy.min(), yy.max())
# fig.set_xlabel("Sepal length")
# fig.set_ylabel("Sepal width")
# fig.set_xticks(())
# fig.set_yticks(())
# fig.set_title('title')

# st.pyplot(plt)

"""%s""" % translation['solution5']

with st.expander("""%s""" % translation['solution7']):
    """%s""" % translation['solution6']

    i = 0
    dict = {}
    e1, e2 = st.columns(2)

    for c in X_train.columns:
        if i % 2 == 0:
            dict[c] = e1.slider(c, 0, 30, 7)
        else:
            dict[c] = e2.slider(c, 0, 30, 5)
        i += 1

    if st.button("""%s""" % translation['evaluate']):
        new_set = pd.DataFrame(dict, index=[0])

        new_set = new_set.apply(lambda x: x/30)
        new_prediction = multi_target_model.predict(new_set)
        
        """%s""" % translation['error_msg']
        st.write(new_prediction)
        st.markdown(
            """
            <style>
            .container {
                display: flex;
            }
            
            .zone-active {
                float: left;
                position: absolute;
                height: 100% !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="container">
                <img src="data:image/png;base64,{base64.b64encode(open('./resources/images/zones.png', "rb").read()).decode()}">
                <img class="zone-active" style="display: {'block' if new_prediction[0][0][1] == '1' else 'none'};" src="data:image/png;base64,{base64.b64encode(open('./resources/images/z1.png', "rb").read()).decode()}">
                <img class="zone-active" style="display: {'block' if new_prediction[0][0][4] == '1' else 'none'};" src="data:image/png;base64,{base64.b64encode(open('./resources/images/z2.png', "rb").read()).decode()}">
                <img class="zone-active" style="display: {'block' if new_prediction[0][0][7] == '1' else 'none'};" src="data:image/png;base64,{base64.b64encode(open('./resources/images/z3.png', "rb").read()).decode()}">
                <img class="zone-active" style="display: {'block' if new_prediction[0][0][10] == '1' else 'none'};" src="data:image/png;base64,{base64.b64encode(open('./resources/images/z4.png', "rb").read()).decode()}">
                <img class="zone-active" style="display: {'block' if new_prediction[0][0][13] == '1' else 'none'};" src="data:image/png;base64,{base64.b64encode(open('./resources/images/z5.png', "rb").read()).decode()}">
                <img class="zone-active" style="display: {'block' if new_prediction[0][0][16] == '1' else 'none'};" src="data:image/png;base64,{base64.b64encode(open('./resources/images/z6.png', "rb").read()).decode()}">
            </div>
            """,
            unsafe_allow_html=True
        )
