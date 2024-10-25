import numpy as np
import pandas as pd

def fetch_medal_tally(df, year, country):
    # Remove duplicates based on the specified columns
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])

    # Initialize flag to determine grouping logic
    flag = 0

    # Filter the data based on the selected year and country
    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    elif year == 'Overall' and country != 'Overall':
        flag = 1  # Set flag to group by year when country is selected but year is not
        temp_df = medal_df[medal_df['region'] == country]
    elif year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    elif year != 'Overall' and country != 'Overall':
        temp_df = medal_df[(medal_df['Year'] == int(year)) & (medal_df['region'] == country)]

    # If country is selected but year is 'Overall', group by year and sum medals
    if flag == 1:
        x = temp_df.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Year').reset_index()
        
        # If no data is available, return 0 medals for the selected country
        if x.empty:
            x = pd.DataFrame({'Year': [int(year)], 'Gold': [0], 'Silver': [0], 'Bronze': [0], 'total': [0]})

    # Otherwise, group by region (countries) and sum medals
    else:
        x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending=False).reset_index()

        # Add total medals column (Gold + Silver + Bronze)
        x['total'] = x['Gold'] + x['Silver'] + x['Bronze']

        # If no data is available, fill with 0 for medals for the selected filter
        if x.empty:
            regions = df['region'].unique() if year == 'Overall' else [country]
            x = pd.DataFrame({'region': regions, 'Gold': [0] * len(regions), 'Silver': [0] * len(regions), 'Bronze': [0] * len(regions), 'total': [0] * len(regions)})

    return x

def country_year_list(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    country = np.unique(df['region'].dropna().values).tolist()
    country.sort()
    country.insert(0, 'Overall')

    return years, country


def participating_nations_over_time(df):
    nations_over_time=df.drop_duplicates(['Year','region'])['Year'].value_counts().reset_index().sort_values('Year')

    nations_over_time.loc[nations_over_time['Year'] == 2024, 'count'] = 204
    nations_over_time.loc[nations_over_time['Year'] == 2020, 'count'] = 206
    nations_over_time.rename(columns={'Year':'Edition','count':'No. of Countries'},inplace = True)
    return nations_over_time

def events_over_time(df):
    event_over_year=df.drop_duplicates(['Year','Event'])['Year'].value_counts().reset_index().sort_values('Year')
    event_over_year.loc[event_over_year['Year'] == 2024, 'count'] = 329
    event_over_year.loc[event_over_year['Year'] == 2020, 'count'] = 339
    event_over_year.rename(columns={'Year':'Edition','count':'Events'},inplace = True)
    return event_over_year

def most_successful(df,sport):
  temp_df=df.dropna(subset=['Medal'])

  if sport !='Overall':
    temp_df=temp_df[temp_df['Sport']==sport]
  x= temp_df['Name'].value_counts().reset_index().head(15).merge(df,left_on='Name',right_on='Name',how='left')[['Name','count','Sport','region']].drop_duplicates('Name') 
  x.rename(columns={'count': 'Medals'},inplace=True)
  return x

def year_wise_medal_tally(df,country):
    temp_df= df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'],inplace= True)
    new_df = temp_df[temp_df['region']==country]
    final_df=new_df.groupby('Year').count()['Medal'].reset_index()

    return final_df

def country_event_heatmap(df,country):
    temp_df= df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'],inplace= True)
    new_df = temp_df[temp_df['region']==country]

    pt= new_df.pivot_table(index='Sport',columns='Year',values='Medal',aggfunc='count').fillna(0)

    return pt

def most_successful_countrywise(df,country):
  temp_df=df.dropna(subset=['Medal'])

  
  temp_df=temp_df[temp_df['region']==country]
  x= temp_df['Name'].value_counts().reset_index().head(10).merge(df,left_on='Name',right_on='Name',how='left')[['Name','count','Sport']].drop_duplicates('Name') 
  x.rename(columns={'count': 'Medals'},inplace=True)
  return x


def men_vs_women(df):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    athlete_df['Sex'].replace('W', 'F', inplace=True)

    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)

    final.fillna(0, inplace=True)

    return final