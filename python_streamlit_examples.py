



# streamlit run python_streamlit_examples.py

# *** IMPORTS ***
import streamlit as st
import pandas as pd
import numpy as np
from vega_datasets import data as vds
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import altair as alt
import datetime

# *** TITLE ***
st.title('Streamlit Application')
st.text('text')
"""
### Subtitle
"""

# *** DATAFRAME ***
# st.write() is not always needed
st.title('DataFrame')
cars = vds.cars()
st.write(cars.head())
st.dataframe(cars)

# *** TABLE ***
st.title('Table')
st.table(cars.head())

# *** LINE CHART ***
st.title('Line Chart')
line_chart_data = pd.DataFrame({'A': [2,1,3,4,2], 'B': [4,3,2,5,3]})
st.line_chart(line_chart_data)

# display matplotlib plot
st.title('Line Chart 2')
plt.plot([3,4,3,5,4])
st.pyplot()

# *** BAR CHART ***
# example using matplotlib
st.title('Bar Chart')
fig, ax = plt.subplots()
ax.bar([1,2,3,4,5],[2,1,3,4,1])
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(list('ABDCE'))
st.write(fig)
# can also use st.bar_chart()
# st.bar_chart()

# *** SCATTER PLOT ***
# example using altair
cars = vds.cars()
st.write(cars.head())
scatter = alt.Chart(cars).mark_circle().encode(x='Weight_in_lbs', y='Miles_per_Gallon').interactive()
st.altair_chart(scatter)

# *** OTHER CHARTS AND PLOTS ***
# streamlit can display a number of other charts, images, etc.

# *** MAP ***
st.title('Map')
airports = vds.airports()[['latitude', 'longitude']][0:100]
st.map(airports)

# *** SLIDER ***
st.title('Slider')
slider = st.slider(label='slider', min_value=0, max_value=10, value=5)
st.write(slider, 'cubed is', slider * slider * slider)

# *** CHECKBOX ***
st.title('Checkbox')
fig_map = plt.figure(figsize=(12, 8))
ax_map = fig_map.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax_map.set_global()
if st.checkbox('land'): ax_map.add_feature(cfeature.LAND)
if st.checkbox('ocean'): ax_map.add_feature(cfeature.OCEAN)
if st.checkbox('coastline'): ax_map.add_feature(cfeature.COASTLINE)
if st.checkbox('borders'): ax_map.add_feature(cfeature.BORDERS, linestyle=':')
if st.checkbox('lakes'): ax_map.add_feature(cfeature.LAKES, alpha=0.5)
if st.checkbox('rivers'): ax_map.add_feature(cfeature.RIVERS)
st.write(fig_map)

# *** SELECTBOX ***
# widgets can be put in sidebar
# not all dashboard elements can be added to sidebar
st.title('Selectbox')
st.subheader('See Sidebar for Selectbox Widget')
option = st.sidebar.selectbox('Choose', ['a','b','c','d','e'])
st.write('You selected: ', option)

# *** TEXT INPUT AND BUTTON ***
st.title('Text Input & Button')
text = st.text_input('Enter Name')
if st.button('click button'):
    st.write(f'Hello {text}')

# *** RADIO BUTTONS ***
st.title('Radio Buttons')
radio_button_options = st.radio('options:', ('A', 'B', 'C'))
if radio_button_options == 'A':
    st.write('option A chosen')
elif radio_button_options == 'B':
    st.write('option B chosen')
elif radio_button_options == 'C':
    st.write('option C chosen')

# *** MULTISELECT ***
st.title('Multiselect')
options = st.multiselect('What are your favorite colors',
                         ['green', 'orange', 'red', 'blue', 'purple'])

st.write('You selected:')
for i in options:
    st.write(i)

# *** DATES ***
st.title('Select Dates')
d = st.date_input('select date:', datetime.date(2016, 1, 1))
st.write(d)

# *** COLOR PICKER ***
st.title('Color Picker')
color = st.beta_color_picker('pick a color')
st.write(color)

# *** BALLOONS ***
# st.balloons()

# *** SEE DOCUMENTATION FOR MORE WIDGETS ***
