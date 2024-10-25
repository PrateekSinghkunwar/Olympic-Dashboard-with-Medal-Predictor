import streamlit as st
import pandas as pd
import preprocessor, helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import time
import pickle

# Load Data and Models
@st.cache_data
def load_data():
    df = pd.read_csv('D:/Olympic Dashboard/Olympic Data.csv')
    df = preprocessor.preprocess(df)
    modelrfc = pickle.load(open("modelrfc.pkl", "rb"))
    modellr = pickle.load(open("modellr.pkl", "rb"))
    transformer = pickle.load(open("transformer.pkl", "rb"))
    return df, modelrfc, modellr, transformer

df, modelrfc, modellr, transformer = load_data()

# Sidebar Options
st.sidebar.title("Olympics Dashboard")
st.sidebar.image('D:/Olympic Dashboard/Olympic Symbol.png')

user_menu = st.sidebar.radio(
    'Select an Option',
    ('Medal Tally', 'Medal Predictor', 'Overall Analysis', 'Country-wise Analysis', 'Athlete-wise Analysis')
)

# Reusable functions for visualizations
def show_medal_tally(df, selected_year, selected_country):
    medal_tally = helper.fetch_medal_tally(df, selected_year, selected_country)
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Tally")
    elif selected_year != 'Overall' and selected_country == 'Overall':
        st.title(f"Medal Tally in {selected_year} Olympics")
    elif selected_year == 'Overall' and selected_country != 'Overall':
        st.title(f"{selected_country}'s Overall Performance")
    else:
        st.title(f"{selected_country}'s Performance in {selected_year} Olympics")
    st.table(medal_tally)

def show_overall_analysis(df):
    st.title("Top Statistics")
    editions = df['Year'].nunique() + 2
    cities, sports, events, athletes, nations = (df['City'].nunique(), df['Sport'].nunique(), 
                                                 df['Event'].nunique(), df['Name'].nunique(), 
                                                 df['region'].nunique())
    
    # Display metrics in a grid format
    cols = st.columns(3)
    cols[0].metric("Editions", editions)
    cols[1].metric("Host Cities", cities)
    cols[2].metric("Sports", sports)

    cols = st.columns(3)
    cols[0].metric("Events", events)
    cols[1].metric("Nations", nations)
    cols[2].metric("Athletes", athletes)

    # Add more charts and visualizations
    nations_over_time = helper.participating_nations_over_time(df)
    st.title("Participating Nations Over the Years")
    st.plotly_chart(px.line(nations_over_time, x="Edition", y="No. of Countries"))

    st.title("Events Over the Years")
    events_over_time = helper.events_over_time(df)
    st.plotly_chart(px.line(events_over_time, x="Edition", y="Events"))

    st.title("No. of Events Over Time (All Sports)")
    fig, ax = plt.subplots(figsize=(20, 17))
    event_heatmap = df.drop_duplicates(['Year', 'Sport', 'Event']).pivot_table(
        index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0)
    sns.heatmap(event_heatmap.astype(int), annot=True, ax=ax)
    st.pyplot(fig)

# Medal Tally Section
if user_menu == 'Medal Tally':
    st.sidebar.header("Medal Tally")
    years, country = helper.country_year_list(df)
    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_country = st.sidebar.selectbox("Select Country", country)
    show_medal_tally(df, selected_year, selected_country)

# Overall Analysis Section
elif user_menu == 'Overall Analysis':
    show_overall_analysis(df)

# Add new country-wise analysis section with more interactive visualizations
elif user_menu == 'Country-wise Analysis':
    st.sidebar.title('Country-wise Analysis')
    country_list = sorted(df['region'].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox('Select a Country', country_list)

    st.title(f"{selected_country}'s Performance Over the Years")
    country_df = helper.year_wise_medal_tally(df, selected_country)
    st.plotly_chart(px.line(country_df, x="Year", y="Medal"))

    st.title(f"{selected_country} Excels in the Following Sports")
    fig, ax = plt.subplots(figsize=(20, 20))
    sport_heatmap = helper.country_event_heatmap(df, selected_country)
    sns.heatmap(sport_heatmap, annot=True, ax=ax)
    st.pyplot(fig)

    st.title(f"Top 10 Athletes of {selected_country}")
    top_athletes_country = helper.most_successful_countrywise(df, selected_country)
    st.table(top_athletes_country)

# Athlete-wise Analysis Section remains similar
elif user_menu == 'Athlete-wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    st.title("Age Distribution of Athletes")
    x1, x2, x3, x4 = (athlete_df['Age'].dropna(),
                      athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna(),
                      athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna(),
                      athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna())
    
    fig_age_dist = ff.create_distplot([x1, x2, x3, x4], 
                                      ['Overall Age', 'Gold', 'Silver', 'Bronze'], 
                                      show_hist=False, show_rug=False)
    fig_age_dist.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig_age_dist)



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
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age wrt Sports(Gold Medalist)")
    st.plotly_chart(fig)




    st.title("Men Vs Women Participation Over the Years")
    final = helper.men_vs_women(df)
    final_filtered = final[~final['Year'].isin([2020, 2024])]
    fig = px.line(final_filtered, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)


# Medal Predictor Section
elif user_menu == 'Medal Predictor':
    st.title("Olympics Medal Predictor")
    selected_col = ["Sex" , "region" ,"Sport","Height" , "Weight" , "Age" ]
    sport = ['Aeronautics', 'Alpine Skiing', 'Alpinism', 'Archery', 'Art Competitions', 'Athletics', 'Badminton', 'Baseball', 'Basketball', 'Basque Pelota', 'Beach Volleyball', 'Biathlon', 'Bobsleigh', 'Boxing', 'Canoeing', 'Cricket', 'Croquet', 'Cross Country Skiing', 'Curling', 'Cycling', 'Diving', 'Equestrianism', 'Fencing', 'Figure Skating', 'Football', 'Freestyle Skiing', 'Golf', 'Gymnastics', 'Handball', 'Hockey', 'Ice Hockey', 'Jeu De Paume', 'Judo', 'Lacrosse', 'Luge', 'Military Ski Patrol', 'Modern Pentathlon', 'Motorboating', 'Nordic Combined', 'Polo', 'Racquets', 'Rhythmic Gymnastics', 'Roque', 'Rowing', 'Rugby', 'Rugby Sevens', 'Sailing', 'Shooting', 'Short Track Speed Skating', 'Skeleton', 'Ski Jumping', 'Snowboarding', 'Softball', 'Speed Skating', 'Swimming', 'Synchronized Swimming', 'Table Tennis', 'Taekwondo', 'Tennis', 'Trampolining', 'Triathlon', 'Tug-Of-War', 'Volleyball', 'Water Polo', 'Weightlifting', 'Wrestling']
    country = ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Antigua', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Boliva', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic', 'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guam', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Individual Olympic Athletes', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Republic of Congo', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts', 'Saint Lucia', 'Saint Vincent', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad', 'Tunisia', 'Turkey', 'Turkmenistan', 'UK', 'USA', 'Uganda', 'Ukraine', 'United Arab Emirates', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Virgin Islands, British', 'Virgin Islands, US', 'Yemen', 'Zambia', 'Zimbabwe']
    with st.form("my_form"):
        Sex = st.selectbox("Select Sex",["M","F"])
        Age = st.slider("Select Age",10,60)
        Height = st.slider("Select Height(In centimeters)",127,220)
        Weight = st.slider("Select Weight(In kilograms)",25,130)
        region = st.selectbox("Select Country",country)
        Sport = st.selectbox("Select Sport",sport)
        input_model = st.selectbox("Select Prediction Model",["Random Forest Classifier","Logistic Regression","Neutral Network"])


        
        submitted = st.form_submit_button("Submit")
        if submitted:
            inputs = [Sex,region,Sport,Height,Weight,Age]
            inputs = pd.DataFrame([inputs],columns=selected_col)
            inputs = transformer.transform(inputs)
            if input_model == "Random Forest Classifier":
                model = modelrfc
            if input_model == "Logistic Regression":
                model = modellr
            if input_model == "Neutral Network":
                model = modelrfc
            prediction = model.predict(inputs)
            
            with st.spinner('Predicting output...'):
                time.sleep(1)
                if prediction[0] == 0 :
                    ans = "Low"
                    st.warning("Medal winning probability is {}".format(ans),icon="⚠️")
                else :
                    ans = "High"
                    st.success("Medal winning probability is {}".format(ans),icon="✅")