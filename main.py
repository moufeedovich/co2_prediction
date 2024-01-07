import base64
import streamlit as st
import pandas as pd
import altair as alt
import warnings
import plotly.express as px
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle

warnings.filterwarnings('ignore')

st.set_page_config(page_title="CO2 Emission", page_icon=":bar_chart:", layout="wide")

###############################################--- Data ---###############################################

data = pd.read_csv("final_data.csv")
st.markdown("<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)

###############################################--- Side bar filters ---###############################################

st.sidebar.header("My filters")
countries = st.sidebar.multiselect("Choose a country", sorted(data["country_name"].unique()))
years = st.sidebar.multiselect("Choose a year", sorted(data["year"].unique(), reverse=True))
filtered_data = data[(data['country_name'].isin(countries)) & (data['year'].isin(years))]

###############################################--- data analytics dashboard ---###############################################

dashboard_url = "https://app.powerbi.com/view?r=eyJrIjoiNTNjMGY4YzYtMjBlNS00ZDE2LWFhZTYtNzBjOTE4MjJmNGQ2IiwidCI6IjY2NDU3NWE2LTY5OTUtNDVkNi1iZTE1LTY1YjgzYWI4MDBkZCIsImMiOjh9"
button_code = f'<a href="{dashboard_url}" target="_blank" style="text-decoration: none; color: white;"><button style="background-color: #FFFFFF; padding: 10px 15px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer;">Explore Dashboard</button></a>'
st.sidebar.header("Data Analysis Section")
st.sidebar.markdown(button_code, unsafe_allow_html=True)

###############################################--- jumping to data science section ---###############################################

st.sidebar.header("Data Science Section")
st.sidebar.markdown("[Jump to Choose a Country Section](#choose-a-country)")





###############################################--- percentage growth of co2 ---###############################################

def percentage_growth(data, countries, years):
    # Check if there is filtered data
    if len(countries) > 0 or len(years) > 0:
        data_ = pd.read_csv("final_data.csv")
        if len(countries) > 0:
            data_ = data_[(data_['country_name'].isin(countries))]
        if len(years) > 0:
            data_ = data_[(data_['year'].isin(years))]

        min_year = data_['year'].min()
        max_year = data_['year'].max()

        sum_min = data_[data_['year'] == min_year]['value'].sum()
        sum_max = data_[data_['year'] == max_year]['value'].sum()

        growth = ((sum_max - sum_min) / abs(sum_min)) * 100 if sum_min != 0 else 0

        if growth > 0:
            color = 'red'
            growth = (f"↑ {round(growth,2)}")
        else:
            color = 'green'
            growth = (f"↓ {round(growth,2)}")

        st.markdown(f"<p style='color:{color}; font-size:{'60px'};'><strong>{growth}</strong>%</p><p style='font-size:{'20px'};'> Last year's CO2 level: {sum_max:,.2f}</p>", unsafe_allow_html=True)

    else:
        data = pd.read_csv("final_data.csv")
        data_new = data.groupby(['year'])['value'].sum().reset_index()

        min_year = data_new["year"].min()
        max_year = data_new["year"].max()

        value_min_year = data[data['year'] == min_year]['value'].sum()
        value_max_year = data[data['year'] == max_year]['value'].sum()

        growth = ((value_max_year - value_min_year) / abs(value_min_year)) * 100 if value_min_year != 0 else 0

        if growth > 0:
            color = 'red'
            growth = (f"↑ {round(growth,2)}")
        else:
            color = 'green'
            growth = (f"↓ {round(growth,2)}")

        st.markdown(f"<p style='color:{color}; font-size:{'60px'};'><strong>{growth}</strong>%</p><p style='font-size:{'20px'};'>Last year's CO2 level: {value_max_year:,.2f}</p>", unsafe_allow_html=True)

###############################################--- percentage growth of population ---###############################################

def population_growth(data, countries, years):
    # Check if there is filtered data
    if len(countries) > 0 or len(years) > 0:
        data_ = pd.read_csv("final_data.csv")
        if len(countries) > 0:
            data_ = data_[(data_['country_name'].isin(countries))]

        if len(years) > 0:
            data_ = data_[(data_['year'].isin(years))]

        min_year = data_['year'].min()
        max_year = data_['year'].max()


        sum_min = data_[data_['year'] == min_year]['Count'].sum()
        sum_max = data_[data_['year'] == max_year]['Count'].sum()

        growth = ((sum_max - sum_min) / abs(sum_min)) * 100 if sum_min != 0 else 0

        if growth > 0:
            color = 'red'
            growth = (f"↑ {round(growth,2)}")
        else:
            color = 'green'
            growth = (f"↓ {round(growth,2)}")

        st.markdown(f"<p style='color:{color}; font-size:{'60px'};'><strong>{growth}</strong>%</p><p style='font-size:{'20px'};'> Last year's population: {sum_max:,.2f}</p>", unsafe_allow_html=True)

    else:
        data = pd.read_csv("final_data.csv")
        data_new = data.groupby(['year'])['Count'].sum().reset_index()

        min_year = data_new["year"].min()
        max_year = data_new["year"].max()

        value_min_year = data[data['year'] == min_year]['Count'].sum()
        value_max_year = data[data['year'] == max_year]['Count'].sum()

        growth = ((value_max_year - value_min_year) / abs(value_min_year)) * 100 if value_min_year != 0 else 0

        if growth > 0:
            color = 'red'
            growth = (f"↑ {round(growth,2)}")
        else:
            color = 'green'
            growth = (f"↓ {round(growth,2)}")

        # Use Streamlit's markdown to display text with custom styling
        st.markdown(f"<p style='color:{color}; font-size:{'60px'};'><strong>{growth}</strong>%</p><p style='font-size:{'20px'};'> Last year's population: {value_max_year:,.2f}</p>", unsafe_allow_html=True)

###############################################--- displaying the co2 and population percantages ---###############################################

def display_growth_metrics(data, countries, years):
    col1, col2 = st.columns(2)

    with col1:
        st.write("# CO2")
        percentage_growth(data, countries, years)

    with col2:
        st.write("# Population")
        population_growth(data, countries, years)

result_df = display_growth_metrics(filtered_data, countries, years)

st.markdown('<br>', unsafe_allow_html=True)

###############################################--- line chart of co2 and population ---###############################################

def first_chart():
    st.markdown(f"<span style='color:#0000FF;'>CO2 Value</span> and <span style='color:#FF0000;'>Population</span> Over Time", unsafe_allow_html=True)
    filtered_df = data
    if countries:
        filtered_df = filtered_df[filtered_df["country_name"].isin(countries)]
    if years:
        filtered_df = filtered_df[filtered_df["year"].isin(years)]

    # Group by year and sum the values
    grouped_data_value = filtered_df.groupby('year')['value'].sum().reset_index()
    grouped_data_count = filtered_df.groupby('year')['Count'].sum().reset_index()

    # Line chart with two lines
    line_chart_value = alt.Chart(grouped_data_value).mark_line(color='blue').encode(
        x='year',
        y=alt.Y('value', axis=alt.Axis(title='CO2 Value')),
        tooltip=['year', 'value']
    ).properties(
        width=600,
        height=400
    )

    line_chart_count = alt.Chart(grouped_data_count).mark_line(color='red').encode(
        x='year',
        y=alt.Y('Count', axis=alt.Axis(title='Population')),
        tooltip=['year', 'Count']
    )

    # Combine the charts with dual axes
    dual_axis_chart = alt.layer(line_chart_value, line_chart_count).resolve_scale(
        y='independent'  # Separate y-axes for the two lines
    )


    # Streamlit display
    st.altair_chart(dual_axis_chart, use_container_width=True)

first_chart()

###############################################--- taking the filtered data ---###############################################

filtered_df = data
if countries:
    filtered_df = filtered_df[filtered_df["country_name"].isin(countries)]
if years:
    filtered_df = filtered_df[filtered_df["year"].isin(years)]

###############################################--- button to download the filtered data after shoing it ---###############################################
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    # Add styling to the download link with hover effect
    href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv" style="text-decoration: none; padding: 10px; background-color: #4CAF50; color: white; border-radius: 5px; cursor: pointer; margin-top: 10px; transition: background-color 0.3s;">Download CSV</a>'
    return href

###############################################--- showing the filtered data in a table ---###############################################
if st.button("Show Filtered Data Table"):
    st.dataframe(filtered_df)
    st.markdown(get_table_download_link(filtered_df), unsafe_allow_html=True)

###############################################--- map chart for both co2 and population ---###############################################

def map():

    filtered_data = data.copy()
    if countries:
        filtered_data = filtered_data[filtered_data["country_name"].isin(countries)]
    if years:
        filtered_data = filtered_data[filtered_data["year"].isin(years)]

    # Create CO2 Emission Heatmap
    fig_co2 = px.density_mapbox(
        filtered_data,
        lat="Latitude",
        lon="Longitude",
        z="value",
        radius=13,
        center=dict(lat=filtered_data["Latitude"].mean(), lon=filtered_data["Longitude"].mean()),
        zoom=2,
        mapbox_style="carto-positron",
        title="CO2 Emission Heatmap",
    )

    # Create Population Heatmap
    fig_population = px.density_mapbox(
        filtered_data,
        lat="Latitude",
        lon="Longitude",
        z="Count",
        radius=13,
        center=dict(lat=filtered_data["Latitude"].mean(), lon=filtered_data["Longitude"].mean()),
        zoom=2,
        mapbox_style="carto-positron",
        title="Population Heatmap"
    )

    # Display the maps side by side
    st.plotly_chart(fig_co2, use_container_width=True)
    st.plotly_chart(fig_population, use_container_width=True)

map()

###############################################--- listing the countries ---###############################################

country = data['country_name'].unique()
country_list = list(country)

###############################################--- listing the years ---###############################################

years_list = list(range(1960, 2051))

###############################################--- Encoding ---###############################################

label_encoder1 = preprocessing.LabelEncoder()
data["country_name"] = label_encoder1.fit_transform(data['country_name'])

###############################################--- User selections ---###############################################

col1, col2 = st.columns(2)
with col1:
     x=st.write(" ## Choose a country: ")
     selected_country = st.selectbox("Select a country: ", country_list)
with col2:
     st.write("## Choose a year:")
     selected_year = st.selectbox("Select a year:", years_list)

col1, col2 = st.columns(2)

###############################################--- chart for the actual vs predicted and the mae and accuracy ---###############################################

def errors():
    # Plotting actual vs predicted values
    fig, ax = plt.subplots()

    # Plot actual values in blue
    ax.scatter(y_test, y_test, color='blue', label='Actual', alpha=0.5)

    # Plot predicted values in red
    ax.scatter(y_test, y_pred_2, color='red', label='Predicted', alpha=0.5)

    # Add a diagonal line for reference
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)

    ax.set_title('Actual vs Predicted Values')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()

    # Show the plot in Streamlit
    st.pyplot(fig)

    mae = mean_absolute_error(y_test, y_pred_2)
    st.write(f" ### Mean Absolute Error (MAE): {mae:.4f}")


    r2 = r2_score(y_test, y_pred_2)
    st.write(f" ### R-squared (Accuracy): {r2:.4f}")

###############################################--- population prediction model ---###############################################

with col1:
        st.write("<br>", unsafe_allow_html=True)  # HTML <br> tag for a line break

        X = data[["country_name","year"]]  # Features
        y = data['Count']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the XGBoost model
        model = RandomForestRegressor(random_state=42)
        # Train the model on the training set
        model.fit(X_train, y_train)
        ###############################################--- Saving the population model ---###############################################

        # Predict on the test set
        # with open("population.pickle","wb") as f:
        #      pickle.dump(model,f)

        pickle_in = open("population.pickle","rb")
        model = pickle.load(pickle_in)


        y_pred = model.predict(X_test)


        new_val1 = label_encoder1.transform([selected_country])
        new_val2 = selected_year
        test = np.array([new_val1[0],new_val2])

        y_pred = model.predict(test.reshape((1,2)))

        y_pred = int(y_pred[0])
        y_pred = round(y_pred, 2)

        formatted_price = "{:,.0f}".format(y_pred)
        st.write(" ### Estimated Population: ", formatted_price)

###############################################--- user population input ---###############################################

with col2:
    new_val3 = st.number_input("Enter population number:", value=y_pred, step=1)

X = data[["country_name","year","Count"]]  # Features
y = data['value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)

###############################################--- Saving the co2 model ---###############################################

# # Train the model on the training set
# model.fit(X_train, y_train)
# # Predict on the test set
# with open("co2_model.pickle","wb") as f:
#     pickle.dump(model,f)

pickle_in = open("co2_model.pickle","rb")
model = pickle.load(pickle_in)

y_pred = model.predict(X_test)
y_pred_2 = y_pred

new_val1 = label_encoder1.transform([selected_country])
new_val2 = selected_year
test = np.array([new_val1[0],new_val2, new_val3])

y_pred = model.predict(test.reshape((1,3)))

y_pred = int(y_pred[0])
y_pred = round(y_pred, 2)
formatted_price = "{:,.0f}".format(y_pred)
st.write("# CO2 Predicted Value: ")
styled_price = f'<span style="color:#4CAF50; font-weight:bold; font-size:3.5em;">{formatted_price}</span>'
st.markdown(styled_price, unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)  # HTML <br> tag for a line break
st.write("<br>", unsafe_allow_html=True)  # HTML <br> tag for a line break
st.write("<br>", unsafe_allow_html=True)  # HTML <br> tag for a line break


errors()




