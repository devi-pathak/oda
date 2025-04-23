import streamlit as st
import pandas as pd
import preprocessor, helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import pickle
import numpy as np

from preprocess_and_save import region_df

# Cache data loading
@st.cache_data
def load_data():
    return pd.read_csv('agg_df.csv')

# Load pre-aggregated data
agg_df = load_data()

# Load pre-trained models (exclude Ensemble, as we'll define it locally)
models = {
    'Random Forest': pickle.load(open('random_forest_model.pkl', 'rb')),
    'XGBoost': pickle.load(open('xgboost_model.pkl', 'rb'))

}

# Define ensemble_predict function locally
def ensemble_predict(X, rf_model=models['Random Forest'], xgb_model=models['XGBoost']):
    y_pred_rf = rf_model.predict(X)
    y_pred_xgb = xgb_model.predict(X)
    return 0.5 * y_pred_rf + 0.5 * y_pred_xgb

# Add Ensemble RF+XGB to models dictionary
models['Ensemble RF+XGB'] = ensemble_predict

# Load RMSE scores
rmse_scores = pickle.load(open('rmse_scores.pkl', 'rb'))

# Cache original data loading
@st.cache_data
def load_original_data():
    df = pd.read_csv('athlete_events.csv', dtype={
        'ID': 'int32', 'Name': 'category', 'Sex': 'category', 'Age': 'float32',
        'Height': 'float32', 'Weight': 'float32', 'Team': 'category', 'NOC': 'category',
        'Games': 'category', 'Year': 'int32', 'Season': 'category', 'City': 'category',
        'Sport': 'category', 'Event': 'category', 'Medal': 'category'
    })
    return preprocessor.preprocess(df, region_df)

# Load original data for other sections
df = load_original_data()

st.sidebar.title("Olympics Analysis")
st.sidebar.image('https://shop.flagfactory.bg/image/cache/catalog/products/flags/sports/olympics-600x360h.png')
user_menu = st.sidebar.radio(
    'Select Option:',
    ('Medal Tally', 'Overall Analysis', 'Country-wise Analysis', 'Athlete-wise Analysis', 'Medal Prediction')
)

if user_menu == 'Medal Tally':
    st.sidebar.header("Medal Tally")
    years, country = helper.country_year_list(df)
    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_country = st.sidebar.selectbox("Select Country", country)

    medal_tally = helper.fetch_medal_tally(df, selected_year, selected_country)

    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title('Overall Performance')
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title('Performance in ' + str(selected_year) + ' Olympics')
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + ' performance in Olympics')
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(selected_country + ' performance in ' + str(selected_year))

    st.table(medal_tally)

if user_menu == 'Overall Analysis':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.title('Top Statistics')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Nations")
        st.title(nations)
    with col2:
        st.header("Events")
        st.title(events)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    nations_over_time = helper.data_over_time(df, 'region')
    fig = px.line(nations_over_time, x="Edition", y="region")
    st.title("Participating Nations over the Years")
    st.plotly_chart(fig)

    events_over_time = helper.data_over_time(df, 'Event')
    fig = px.line(events_over_time, x="Edition", y="Event")
    st.title("Events over the Years")
    st.plotly_chart(fig)

    athlete_over_time = helper.data_over_time(df, 'Name')
    fig = px.line(athlete_over_time, x="Edition", y="Name")
    st.title("Athletes over the Years")
    st.plotly_chart(fig)

    st.title('No. of Events over time (Every Sport)')
    fig, ax = plt.subplots(figsize=(15, 15))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
                     annot=True, cmap="cubehelix_r")
    st.pyplot(fig)

    st.title("Most Successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    selected_sport = st.selectbox('Select a Sport', sport_list)
    x = helper.most_successful(df, selected_sport)
    if x.empty:
        st.write("No medal-winning athletes found for the selected sport.")
    else:
        st.table(x)

if user_menu == 'Country-wise Analysis':
    st.title('Country-wise Analysis')
    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.sidebar.selectbox('Select a Country', country_list)

    medal_data_overall = df.dropna(subset=['Medal'])
    country_medal_data = medal_data_overall[medal_data_overall['region'] == selected_country]

    if country_medal_data.empty:
        st.warning(f"No medal data available for {selected_country}.")
    else:
        country_df = helper.yearwise_medal_tally(df, selected_country)
        fig = px.line(country_df, x="Year", y="Medal", title=f"{selected_country} Medal Tally over the Years")
        st.plotly_chart(fig)

        st.title(f"{selected_country} excels in the following Sports")
        pt = helper.country_event_heatmap(df, selected_country)
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(pt, annot=True, cmap="cubehelix_r", linewidths=0.5, annot_kws={"size": 9})
        st.pyplot(fig)

        st.title(f"TOP 10 Athletes of {selected_country}")
        top_athletes = helper.most_successful_countrywise(df, selected_country)
        st.table(top_athletes)

if user_menu == 'Athlete-wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()
    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],
                             show_hist=False, show_rug=False)

    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=400)
    st.title("Distribution of Age w.r.t. Sports (Gold Medalist)")
    st.plotly_chart(fig)

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    st.title('Height Vs Weight')
    selected_sport = st.selectbox('Select a Sport', sport_list)
    temp_df = helper.weight_v_height(df, selected_sport)
    fig, ax = plt.subplots()
    ax = sns.scatterplot(x=temp_df['Weight'], y=temp_df['Height'],
                         hue=temp_df['Medal'], style=temp_df['Sex'], s=20)
    st.pyplot(fig)

    st.title("Men Vs Women Participation Over the Years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)

if user_menu == 'Medal Prediction':
    # Calculate target standard deviation for display
    target_std = agg_df['Total_Medals'].std()


    # Country Overall Prediction
    region_mapping = dict(zip(region_df['NOC'], region_df['region']))
    agg_df['region'] = agg_df['NOC'].map(region_mapping)
    country_list = agg_df['region'].dropna().unique().tolist()
    selected_country = st.sidebar.selectbox("Select Country", country_list)
    future_years = list(range(2020, 2032, 4))
    future_year = st.sidebar.selectbox("Select Year", future_years, index=0)

    st.title(f"{selected_country} Medal Prediction for {future_year} Olympics")

    country_data = agg_df[agg_df['region'] == selected_country].iloc[-1]
    future_data = pd.DataFrame({
        'Year': [future_year],
        'Age': [country_data['Age']],
        'Height': [country_data['Height']],
        'Weight': [country_data['Weight']],
        'ID': [country_data['ID']],
        'Prev_Medals': [country_data['Total_Medals']]
    })

    if future_data.isnull().values.any():
        st.write("No available data for prediction.")
    else:
        st.subheader("Overall Medals Likely to Receive")
        predictions = {}
        for name, model in models.items():
            pred = model(future_data) if name == 'Ensemble RF+XGB' else model.predict(future_data)
            medal_count = max(0, int(round(pred[0])))
            predictions[name] = medal_count
        pred_df = pd.DataFrame({
            'Model': list(predictions.keys()),
            'Predicted Medals': list(predictions.values())
        })
        st.table(pred_df)

    # Sport-Specific Prediction
    st.markdown("---")
    st.title('Sport-Specific Prediction')
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    selected_sport = st.selectbox("Select Sport", sport_list, key="select_sport")

    sport_agg_df = df.groupby(['NOC', 'Sport', 'Year']).agg({
        'Gold': 'sum',
        'Silver': 'sum',
        'Bronze': 'sum',
        'Age': 'mean',
        'Height': 'mean',
        'Weight': 'mean',
        'ID': 'count'
    }).reset_index()
    sport_agg_df['Total_Medals'] = sport_agg_df['Gold'] + sport_agg_df['Silver'] + sport_agg_df['Bronze']
    sport_agg_df['Prev_Medals'] = sport_agg_df.groupby(['NOC', 'Sport'])['Total_Medals'].shift(1).fillna(0)
    sport_agg_df['region'] = sport_agg_df['NOC'].map(region_mapping)

    selected_data = sport_agg_df[(sport_agg_df['region'] == selected_country) &
                                (sport_agg_df['Sport'] == selected_sport)]

    if selected_data.empty:
        st.write(f"No historical data available for {selected_country} in {selected_sport}. Cannot predict.")
    else:
        country_sport_data = selected_data.iloc[-1]
        future_data_sport = pd.DataFrame({
            'Year': [future_year],
            'Age': [country_sport_data['Age']],
            'Height': [country_sport_data['Height']],
            'Weight': [country_sport_data['Weight']],
            'ID': [country_sport_data['ID']],
            'Prev_Medals': [country_sport_data['Total_Medals']]
        })
        future_data_sport = future_data_sport.fillna(0)

        st.subheader(f"Medals Likely to Receive in {selected_sport}")
        predictions_sport = {}
        for name, model in models.items():
            pred = model(future_data_sport) if name == 'Ensemble RF+XGB' else model.predict(future_data_sport)
            medal_count = max(0, int(round(pred[0])))
            predictions_sport[name] = medal_count
        pred_sport_df = pd.DataFrame({
            'Model': list(predictions_sport.keys()),
            'Predicted Medals': list(predictions_sport.values())
        })
        st.table(pred_sport_df)