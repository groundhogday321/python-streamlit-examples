# streamlit run python_streamlit_dashboard.py

# *** IMPORTS ***
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from vega_datasets import data as vds
from scipy import stats

# *** DATAFRAME ***
st.title('Cars DataFrame')
cars = vds.cars().dropna()
st.write(cars)

measures = ['Miles_per_Gallon', 'Cylinders', 'Displacement', 'Horsepower', 'Weight_in_lbs', 'Acceleration']

# *** SCATTER PLOT WITH DROPDOWN SELECTBOXES ***
st.title('Scatter Plot')
scatter_x = st.selectbox('x', measures)
scatter_y = st.selectbox('y', measures)
sns.regplot(x=scatter_x, y=scatter_y, data=cars)
st.pyplot()

# prediction model with sklearn (i.e.-mpg at 3000 pounds)
# estimate based on regression line
st.title('Predict MPG')
weight = st.number_input('enter vehicle weight example 1', min_value=1500, max_value=6000)
X = np.array(cars.Weight_in_lbs).reshape(-1,1)
y = np.array(cars.Miles_per_Gallon)
lr = LinearRegression().fit(X,y)
prediction = lr.predict([[weight]])
st.text('predicted mpg using sklearn model')
st.write(f'{prediction[0]:.2f}')
# add in train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr_split = LinearRegression().fit(X_train, y_train)
prediction2 = lr_split.predict([[weight]])
st.text('predicted mpg using sklearn model with train test split')
st.write(f'{prediction2[0]:.2f}')

# prediction model with statsmodels
statsmodels_Y = cars.Miles_per_Gallon
sm_X = cars.Weight_in_lbs
statsmodels_X = sm.add_constant(sm_X)
mod = sm.OLS(statsmodels_Y, statsmodels_X)
fit = mod.fit()
# predictions
weights_list = list(range(0,6001))
weights_df = pd.DataFrame({'weights': weights_list})
weights = weights_df.weights
predict = sm.add_constant(weights)
ypred = fit.predict(predict)
weight2 = st.number_input('enter vehicle weight example 2', min_value=1500, max_value=6000)
st.text('predicted mpg using statsmodels model')
st.write(f'{ypred[weight2]:.2f}')
# st.write(f'{ypred.loc[weight2]:.2f}')

# prediction model with y=mx+b function
cars_x = cars.Weight_in_lbs
cars_y = cars.Miles_per_Gallon
slope, intercept, r_value, p_value, std_err = stats.linregress(cars_x, cars_y)

# m=slope, x=speed, b=y-intercept
def predict(m,x,b):
    y = m*x+b
    return y

weight_text_input = st.text_input('enter vehicle weight example 3')
if st.button('click button'):
    prediction = f'{predict(slope, int(weight_text_input), intercept):.2f}'
    st.text('predicted mpg using y=mx+b function')
    st.write(prediction)

# *** BOXPLOT WITH RADIO BUTTONS ***
st.title('Radio Buttons')
radio_button_options = st.radio('Choose measure for boxplot:', measures)

def create_boxplot(measure):
    swarmplot = st.checkbox('overlay swarmplot')
    # if swarmplot and radio_button_options == measure:
    if swarmplot:
        sns.boxplot(data=cars[measure])
        # sns.stripplot(data=cars[measure], color='lightgrey')
        sns.swarmplot(data=cars[measure], color='lightgrey')
        st.pyplot()
    else:
        sns.boxplot(data=cars[measure])
        st.pyplot()

create_boxplot(radio_button_options)
