import streamlit as st
from streamlit_option_menu import option_menu

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')

import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import nearest_points
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
import re
import json
import networkx as nx
import nx_altair as nxa

#Packages for plotting a wordcloud
from ipywidgets import widgets, interact, interactive, fixed, Button, Layout
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from IPython.display import display
from nltk.tokenize import word_tokenize

#PAGE LAYOUT
st.set_page_config(layout="wide")

#Set color palette 
eurovision_palette = ["#D2B4DE", '#15CAB6', '#F6B53D', '#EF8A5A', '#E85E76', '#696CB5', '#BABABA', '#156082', '#F4F6FA']
eurovision_cont_palette = ['#007d79','#15CAB6','#d9fbfb']

#Add dashboard menu
logo_url = 'https://raw.githubusercontent.com/asolbas/eurovision-dataviz/main/Figures/Eurovision_Song_Contest_logo.png'
st.sidebar.image(logo_url)

with st.sidebar:
    selected = option_menu(
        menu_title = 'Menu',
        options = ['Overview', 'Geopolitics', 'Music', 'Voting'],
        #Boostrap icons codes
        icons = ['graph-up', 'globe-americas', 'music-note-beamed', 'trophy-fill'],
        menu_icon = 'chat-right-heart',
        default_index=0,

    )


#IMPORT DATA & PREPROCESSING -------------------------------

    #Contestants ---------------------
    contestants_df = pd.read_csv('./Data/contestants_preprocessed.csv')
    #Filter irrelevant variables
    columns_filt = ['sf_num', 'running_sf', 'place_sf', 'points_sf', 'points_tele_sf', 
                'points_jury_sf', 'composers', 'lyricists', 'youtube_url']
    contestants_df.drop(columns_filt, axis=1, inplace=True)
    #Data Cleaning
    contestants_df.loc[contestants_df['to_country']=='Andorra','to_country_id'] = 'ad'
    contestants_df.loc[contestants_df['to_country_id']=='mk','to_country'] = 'North Macedonia'
    contestants_df.loc[contestants_df['to_country']=='Czechia','to_country'] = 'Czech Republic'
    contestants_df.loc[contestants_df['to_country']=='Bosnia & Herzegovina','to_country'] = 'Bosnia and Herzegovina'
    contestants_df.loc[contestants_df['to_country']=='Serbia & Montenegro','to_country'] = 'Serbia and Montenegro'
    #Set points_final to 0 if country did not qualify
    contestants_df['points_final'].fillna(0, inplace=True)
    #Drop 2020
    contestants_df = contestants_df[contestants_df['year']!=2020]
    #Add song language
    results_df = pd.read_csv('./Data/Every_Eurovision_Result_Ever.csv')
    results_df['Song'] = results_df['Song'].str.title() #Capitalize each song title 
    contestants_df = contestants_df.merge(results_df[['Country', 'Year', 'Language', 'Song']], left_on=['year', 'to_country', 'song'],
                                        right_on=['Year', 'Country', 'Song'], how='left')
    #Add column to determine if a contestant classified for the final
    contestants_df['finalist'] = 'No'
    contestants_df.loc[~contestants_df['place_final'].isna(), 'finalist'] = 'Yes'
    contestants_df['country'] = contestants_df['to_country']
    #Create a classification group based on their position in the grand final
    def classification_group(final_position):
        if final_position == 1:
            classification = 'Winner'
        elif final_position <= 5: 
            classification = 'Top5'
        elif final_position <= 10: 
            classification = 'Top10'
        elif math.isnan(final_position) or final_position == '_':
            classification = 'Semifinalist'
        else:
            classification = 'Finalist'
        return classification 
    contestants_df['classification'] = contestants_df['place_final'].apply(classification_group)
    #Standarize country code
    country_df = pd.read_csv('./Data/countries.csv')
    country_df = country_df.rename(columns={'country':'country_id'})
    country_df.loc[country_df['country_name']=='Bosnia & Herzegovina','country_name'] = 'Bosnia and Herzegovina'
    country_df.loc[country_df['country_name']=='Serbia & Montenegro','country_name'] = 'Serbia and Montenegro'
    country_df.loc[country_df['country_name']=='Czechia','country_name'] = 'Czech Republic'
    contestants_df = pd.merge(contestants_df, country_df[['country_id','country_name']], left_on='to_country',
                     right_on='country_name', how='left')
    contestants_df = contestants_df.drop(columns=['to_country_id','country_name']).rename(columns={'country_id': 'to_country_id'})

    #Votes ------------------------
    country_votes_df = pd.read_csv('./Data/votes.csv')
    #Transform IDs to upper case
    country_votes_df['from_country_id'] = country_votes_df['from_country_id'].str.upper()
    country_votes_df['to_country_id'] = country_votes_df['to_country_id'].str.upper()
    #Get country name associated with ID
    country_votes_df = pd.merge(country_votes_df, country_df[['country_id','country_name']], left_on='from_country_id',
                        right_on='country_id', how='left').drop(['country_id','from_country'], axis=1).rename(
                            columns={'country_name': 'from_country'})
    country_votes_df = pd.merge(country_votes_df, country_df[['country_id','country_name']], left_on='to_country_id',
                        right_on='country_id', how='left').drop(['country_id','to_country'], axis=1).rename(
                            columns={'country_name': 'to_country'})
    #Select relevant variables/instances
    country_votes_df = country_votes_df[country_votes_df['from_country_id']!='WLD']
    country_votes_df = country_votes_df[country_votes_df['year']!=2020]
    country_votes_df.drop(['tele_points', 'jury_points'], axis=1, inplace=True)
    # Add geographical group 
    country_group_mapping = dict(zip(country_df['country_id'], country_df['region']))
    country_votes_df['from_country_group'] = country_votes_df['from_country_id'].replace(country_group_mapping)
    country_votes_df['to_country_group'] = country_votes_df['to_country_id'].replace(country_group_mapping)

    #Calculate pertentage of points given and received
    country_votes_filter_df = country_votes_df[country_votes_df['round']=='final'][['year', 'from_country_id', 'to_country_id', 'from_country', 
                                                                                    'to_country', 'total_points']].dropna()
    country_votes_filter_df = country_votes_filter_df.groupby(['from_country', 'to_country', 'from_country_id', 'to_country_id' ], as_index=False)['total_points'].sum()

    country_votes_filter_df.dropna(inplace=True)
    #Calculate percentage points a country has given to other countries trough its history
    country_total_points = country_votes_filter_df.groupby('from_country',as_index=False)['total_points'].sum()
    country_total_points.columns = ['from_country', 'overall_points_from']
    country_votes_filter_df = country_votes_filter_df.merge(country_total_points)
    country_votes_filter_df['norm_points_from'] = country_votes_filter_df['total_points'] / country_votes_filter_df['overall_points_from']
    country_votes_filter_df['pct_points_from'] = country_votes_filter_df['norm_points_from'] * 100

    #Calculate percentage points a country has receiven by others trough its history
    country_total_points = country_votes_filter_df.groupby('to_country',as_index=False)['total_points'].sum()
    country_total_points.columns = ['to_country', 'overall_points_to']
    country_votes_filter_df = country_votes_filter_df.merge(country_total_points)
    country_votes_filter_df['norm_points_to'] = country_votes_filter_df['total_points'] / country_votes_filter_df['overall_points_to']
    country_votes_filter_df['pct_points_to'] = country_votes_filter_df['norm_points_to'] * 100

    #Total number of points received/given historically (metadata)
    from_metadata = country_votes_filter_df[['from_country','from_country_id','overall_points_from']].drop_duplicates()
    to_metadata = country_votes_filter_df[['to_country','to_country_id', 'overall_points_to']].drop_duplicates()
    country_points_metadata = from_metadata.merge(to_metadata, how='left', 
        left_on=['from_country', 'from_country_id'], right_on=['to_country','to_country_id',]).drop(['to_country_id','to_country'], axis=1)
    country_points_metadata.fillna(0, inplace=True) 
    country_points_metadata.columns = ['country','country_id','Total points given', 'Total points received']
    country_points_metadata[['Total points given', 'Total points received']] = country_points_metadata[
        ['Total points given', 'Total points received']].astype('int')

    #Songs ----------------------
    songs_df = pd.read_csv('./Data/song_data.csv', na_values=['-', 'unknown'])
    #Drop 2020 data
    songs_df = songs_df[songs_df['year']!=2020]
    #Select relevant variables
    col_sel = ['year', 'country', 'final_draw_position', 'energy', 'danceability', 'happiness', 'acousticness', 
                                'liveness', 'speechiness', 'gender', 'style', 'backing_dancers', 'backing_singers', 'backing_instruments']
    songs_df = songs_df[col_sel]
    #Create a classification column to group songs according to their position
    songs_df['classification'] = songs_df['final_draw_position'].apply(classification_group)
    songs_df.drop('final_draw_position',axis=1, inplace=True)

    #Betting scores ----------------
    bets_df = pd.read_csv('./Data/betting_offices.csv')
    #Get country IDs
    bets_df = pd.merge(bets_df, country_df[['country_id','country_name']], on='country_name',how='left').drop(
        ['country_code'], axis=1).rename(columns={'country_id': 'country_code'})
    #Drop 2020 data
    bets_df = bets_df[bets_df['year']!=2020]
    #Find missing country names and codes
    aux_df = bets_df[bets_df['country_name'].isnull()].merge(contestants_df[['year', 'song', 'to_country', 'to_country_id']
        ], on=['year', 'song']).set_index(bets_df[bets_df['country_name'].isnull()].isnull().index)
    bets_df['country_name'].fillna(aux_df['to_country'], inplace=True)
    bets_df['country_code'].fillna(aux_df['to_country_id'], inplace=True)
    #Drop entries with missing betting scores
    bets_df.dropna(inplace=True)
    #Select relevant variables
    col_sel = ['year', 'country_name', 'country_code', 'song', 'performer', 'contest_round', 'betting_name', 'betting_score']
    bets_df = bets_df[col_sel]    

#EUROVISION IN A NUTSHELL-----------------------
if selected == 'Overview':
    st.title('Eurovision in a Nutshell (1956-2023)')

    #General information about the contest --------
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(
        label='Number of editions',
        value=len(contestants_df['year'].unique()),
        delta=None
    )

    kpi2.metric(
        label='Number of countries',
        value=len(contestants_df['to_country'].unique()),
        delta=None
    )

    kpi3.metric(
        label='Number of songs',
        value=len(contestants_df['song'].unique()),
        delta=None
    )

    kpi4.metric(
        label='Number of languages',
        value=len(contestants_df['Language'].unique()),
        delta=None
    )


    #Plots
    col1, col2 = st.columns(2)

    # Evolución del número de participantes y puntos otorgados-----------
    #Number of participants
    participants_year = contestants_df.groupby(['year', 'finalist'], as_index=False)['to_country_id'].nunique()
    participants_year.columns = ['year', 'finalist', 'n_participants']

    #Total points
    points_year = contestants_df.pivot_table(index='year', 
                                        values=['points_final'], 
                                        aggfunc='sum').stack().reset_index()

    points_year.columns = ['year', 'source', 'points']
    points_year.loc[points_year['points'] == 0, 'points'] = np.nan

    #Combine participants and points
    contestants_year = participants_year.merge(points_year, on='year')
    contestants_year['Total points'] = len(contestants_year)*['Total points awarded']
    contestants_year['year'] = pd.to_datetime(contestants_year['year'], format='%Y')

    #Plot
    base = alt.Chart(contestants_year).encode(alt.X("year:T", axis=alt.Axis(format='%Y')).title('Years'))
    bar = base.mark_bar().encode(
        alt.Y('n_participants').title('Number of participant countries'),
        color=alt.Color("finalist:N", scale=alt.Scale(range=eurovision_palette)).title('Classified for the final')
        )


    line = base.mark_line(size=4).encode(
        alt.Y('points').title('Total points'),
            color = alt.Color('Total points', title=' ', scale=alt.Scale(range=[eurovision_palette[2]]))
                )

    points_plot = alt.layer(bar, line).resolve_scale(
            y='independent', 
            color = 'independent'
        ).properties(
            width=600,
            height=300
        ).configure_axis(
            labelFontSize=16,
            titleFontSize=16
        ).configure_legend(
        titleFontSize=15,
        labelFontSize=15        
        ) 

    col1.subheader('Evolution of the contest')
    col1.altair_chart(points_plot, use_container_width=True, theme=None)

    #Languages -----------------------------------

    #Find entries with multiple languages being English one of them
    def contains_english(language_str):
        return bool(re.search(r'\bEnglish\b', language_str)) and ',' in language_str
    # Identify rows where 'English' is in the list of languages along with other languages
    bilingual_entries = contestants_df['Language'].apply(lambda x: contains_english(str(x)))
    contestants_df.loc[bilingual_entries, 'Language'] = 'English + other language'
    #Select common languages
    common_languages_ls = ['English', 'French', 'German','German','Italian',
                       'Spanish', 'Portuguese', 'English + other language']
    minoritary_languages_ls = contestants_df[~contestants_df['Language'].isin(common_languages_ls)]['Language'].unique()
    # Replace minoritary languages with "Other Languages"
    contestants_df['Language_red'] = contestants_df['Language'].copy()
    contestants_df.loc[contestants_df['Language'].isin(minoritary_languages_ls), 'Language_red'] = 'Other Languages'
    contestants_language_df = contestants_df.groupby(['year', 'Language_red'], as_index=False).size()
    contestants_language_df['Year'] = contestants_language_df['year']
    contestants_language_df['Year'] = pd.to_datetime(contestants_language_df['Year'], format='%Y')

    #Plot languages
    languages_plot =  alt.Chart(contestants_language_df).mark_area().encode(
                x=alt.X("Year:T", axis=alt.Axis(format='%Y')).title('Year'),
                y=alt.Y('size:Q').title('Number of entries'),
                color=alt.Color("Language_red:N", scale=alt.Scale(range=eurovision_palette)
                ).title('Language'),
                tooltip=[alt.Tooltip('year:Q', title='Year'), alt.Tooltip('Language_red:N', title='Language'),
            alt.Tooltip('size:Q', title='Number of songs')]).properties(
                    width=500,
                    height=325
            ).configure_axis(
            labelFontSize=16,
            titleFontSize=16
        ).configure_legend(
        titleFontSize=15,
        labelFontSize=15        
        ) 
    col1.subheader('Evolution of the songs languages')
    col1.altair_chart(languages_plot, use_container_width=True, theme=None)


    # Trayectoria de cada país en el concurso ------------------------
    contestants_traj_df = contestants_df.copy()
    contestants_traj_df['year'] = pd.to_datetime(contestants_traj_df['year'], format='%Y')
    click_state = alt.selection_point(fields=['country'])
    color_scale = alt.Scale(domain=contestants_traj_df['country'].unique(), range=eurovision_palette)
    color = alt.condition(
        click_state,
        alt.Color('country:N', scale=color_scale).legend(None),
        alt.value('rgba(211, 211, 211, 0)')
    )

    points_plot = alt.Chart(contestants_traj_df).mark_line(point=True).encode(
        x=alt.X("year:T", axis=alt.Axis(format='%Y')).title('Years'),
        y=alt.Y('place_contest',scale=alt.Scale(domain=(45, 1))).title('Place in contest'),
        color=color,
        tooltip=[alt.Tooltip('country:N', title='Country'), alt.Tooltip('year:Q', title='Year'),
        alt.Tooltip('song:N', title='Song title'), alt.Tooltip('performer:N', title='Performer'),
        alt.Tooltip('points_final:Q', title='Total points')]
    ).properties(
        width=500,
        height=250
    )

    legend = alt.Chart(contestants_traj_df).mark_point().encode(
        alt.Y('country:N', title=None).axis(orient='right'),
        color=color
    ).transform_filter(
        click_state
    ).properties(
        height=750
    )


    ## Number of entries per country map
    entries_country = contestants_df.groupby(['to_country'], as_index=False)['to_country'].value_counts()
    entries_country.columns = ['country',  'n_entries']
    
    #Create map of Europe
    europe_url = "https://raw.githubusercontent.com/asolbas/eurovision-dataviz/main/Data/europe_map.geojson"
    worldmap = alt.Data(url=europe_url, format=alt.DataFormat(property='features',type='json'))

    choropleth = (
        alt.Chart(worldmap)
        .mark_geoshape()
        .transform_lookup(
            lookup="properties.country", from_=alt.LookupData(entries_country, "country", ["country", "n_entries"])
        )
        .encode(
            color=alt.Color("n_entries:Q").scale(range=eurovision_cont_palette, reverse=True).title('Number of entries'),
            opacity=alt.condition(click_state, alt.value(1), alt.value(0.2)),
            tooltip=[alt.Tooltip('country:N', title='Country'), alt.Tooltip('n_entries:Q', title='Number of entries')],
        )
        .project(
            type= 'mercator',
        scale= 200,                          # Magnify
        center= [-60,70],                     # [lon, lat]
        clipExtent= [[-100, 0], [500, 300]],    # [[left, top], [right, bottom]]
        ).properties(
            width=400,
            height=300
    )
    )

    #Add Australia to the map
    australia_url = "https://raw.githubusercontent.com/asolbas/eurovision-dataviz/main/Data/australia_map.geojson"
    worldmap_australia = alt.Data(url=australia_url, format=alt.DataFormat(property='features',type='json'))
    # create a choropleth map for Australia
    choropleth_australia = (
        alt.Chart(worldmap_australia).mark_geoshape(
        ).transform_lookup(
            lookup="properties.country", from_=alt.LookupData(entries_country, "country", ["country", "n_entries"])
        ).encode(
            color=alt.Color("n_entries:Q").scale(range=['#15CAB6','#d9fbfb'], reverse=True).title('Number of entries'),
            opacity=alt.condition(click_state, alt.value(1), alt.value(0.2)),
            tooltip=[alt.Tooltip('country:N', title='Country'), alt.Tooltip('n_entries:Q', title='Number of entries')],
        ).project(
            type='mercator',
            scale= 100,                          # Magnify
            center=[100, 70],                     # [lon, lat]
            clipExtent= [[-100, 0], [500, 300]],    # [[left, top], [right, bottom]]
        ).properties(
            width=400,
            height=300
    )
    )

    trajectory_plot = (alt.hconcat(alt.vconcat(alt.layer(choropleth,choropleth_australia), points_plot), legend)).configure_axis(
            labelFontSize=16,
            titleFontSize=16
        ).configure_legend(
        titleFontSize=15,
        labelFontSize=15        
        ).add_params(click_state)

    col2.subheader('Trajectory of each country')
    col2.info('Click on one or multiple countries to study its trajectory in the contest.', icon="ℹ️")
    col2.altair_chart(trajectory_plot, use_container_width=False, theme=None)

#GEOPOLITICS IN EUROVISION -----------------------
if selected == 'Geopolitics':
    st.title('Geopolitics in Eurovision (1956-2023)')

    #Plots
    col1, col2 = st.columns(2)

    #Who is the most popular? ----------------------

    ## Number of wins per country
    wins_country = contestants_df[contestants_df['place_contest']==1.0].groupby(['country'], as_index=False)['country'].value_counts()
    wins_country.columns = ['country',  'n_wins']
    wins_country = wins_country.merge(contestants_df['country'], on='country', how='right').fillna(0).drop_duplicates()
    wins_country = wins_country.merge(country_points_metadata, on='country')

    # define a pointer selection
    click_state = alt.selection_point(fields=["country"], toggle=False)
    europe_url = "https://raw.githubusercontent.com/asolbas/eurovision-dataviz/main/Data/europe_map.geojson"
    worldmap = alt.Data(url=europe_url, format=alt.DataFormat(property='features',type='json'))
    choropleth = (
        alt.Chart(worldmap)
        .mark_geoshape()
        .transform_lookup(
            lookup="properties.country", from_=alt.LookupData(wins_country, "country", ["country", "n_wins", 
            "Total points given", "Total points received"])
        )
        .encode(
            color=alt.Color("n_wins:Q").scale(range=eurovision_cont_palette, reverse=True).title('Number of wins'),
            opacity=alt.condition(click_state, alt.value(1), alt.value(0.2)),
            tooltip=[alt.Tooltip('country:N', title='Country'), alt.Tooltip('n_wins:N', title='Number of wins'), 
            "Total points received:Q", "Total points given:Q"],
        )
        .project(
            type= 'mercator',
        scale= 200,                          # Magnify
        center= [-60,70],                     # [lon, lat]
        clipExtent= [[-100, 0], [500, 300]],    # [[left, top], [right, bottom]]
        ).properties(
            width=400,
            height=300
    )
    )

    #Add Australia to the map
    australia_url = "https://raw.githubusercontent.com/asolbas/eurovision-dataviz/main/Data/australia_map.geojson"
    worldmap_australia = alt.Data(url=australia_url, format=alt.DataFormat(property='features',type='json'))
    # create a choropleth map for Australia
    choropleth_australia = (
        alt.Chart(worldmap_australia).mark_geoshape(
        ).transform_lookup(
            lookup="properties.country", from_=alt.LookupData(wins_country, "country", ["country", "n_wins",
            "Total points given", "Total points received"])
        ).encode(
            color=alt.Color("n_wins:Q").scale(range=eurovision_cont_palette, reverse=True).title('Number of wins'),
            opacity=alt.condition(click_state, alt.value(1), alt.value(0.2)),
            tooltip=[alt.Tooltip('country:N', title='Country'), alt.Tooltip('n_wins:N', title='Number of wins'), 
            "Total points received:Q", "Total points given:Q"],
        ).project(
            type='mercator',
            scale= 100,                          # Magnify
            center=[100, 70],                     # [lon, lat]
            clipExtent= [[-100, 0], [500, 300]],    # [[left, top], [right, bottom]]
        ).properties(
            width=400,
            height=300
    )
    )

    # create a bar chart with a ranking
    aux_df = country_votes_filter_df.copy()
    aux_df['country'] = country_votes_filter_df['to_country']
    most_votes_from = (
        alt.Chart(
            aux_df).mark_bar(color = eurovision_palette[0]).encode(
            x=alt.X("pct_points_to").title('Points received (percentage)'),
            y=alt.Y("from_country").title('Country').sort("-x"),
            tooltip=[alt.Tooltip('from_country:N', title='From'), alt.Tooltip('country:N', title='To'), 
            alt.Tooltip("pct_points_to:Q", title='Percentage of points', format='.2f')],
        ).transform_filter(
        click_state
        ).transform_window(
        rank='rank(pct_points_to)',
        sort=[alt.SortField('pct_points_to', order='descending')]
    ).transform_filter(
        alt.datum.rank <= 5).properties(
            width=100,
            height=100,
            title="Most voted by"
        )
    )


    aux_df = country_votes_filter_df.copy()
    aux_df['country'] = country_votes_filter_df['from_country']
    most_votes_to = (
        alt.Chart(
            aux_df).mark_bar(color = eurovision_palette[0]).encode(
            x=alt.X("pct_points_from").title('Points given (percentage)'),
            y=alt.Y("to_country").title('Country').sort("-x"),
            tooltip=[alt.Tooltip('country:N', title='From'), alt.Tooltip('to_country:N', title='To'), 
            alt.Tooltip("pct_points_from:Q", title='Percentage of points', format='.2f')],
        ).transform_filter(
        click_state,
        ).transform_window(
        rank='rank(pct_points_from)',
        sort=[alt.SortField('pct_points_from', order='descending')]
    ).transform_filter(
        alt.datum.rank <= 5).properties(
            width=100,
            height=100,
            title="Most votes to"
        )
    )

    popularity_plot = alt.vconcat(alt.layer(choropleth,choropleth_australia), 
                    alt.hconcat(most_votes_to, most_votes_from)
                    ).add_params(click_state).configure_axis(
            labelFontSize=16,
            titleFontSize=16
        ).configure_legend(
        titleFontSize=15,
        labelFontSize=15,
        orient='top'        
        )
    
    col1.subheader('Who is the most popular?')
    col1.info('Click on one country to study its voting pattern.', icon="ℹ️")
    col1.altair_chart(popularity_plot, use_container_width=False, theme=None)

    #Friendzone plot ----------------------------
    col2.subheader('Friendzone graph')
    col2.info('Each node represents a country whereas the edges represent the relationships between \
              them on the basis of the percentage of point that a country has given to another. \
              The edge width is proportional to this percentage', icon="ℹ️")

    #Percentage threshold 
    #Add an interactive bar to select percentage threshold
    th = col2.select_slider('**Select a percentage threshold**', 
                               options = np.arange(0,20.5,0.5),
                               value=6)

    votes_graph = nx.from_pandas_edgelist(country_votes_filter_df[country_votes_filter_df['pct_points_from']>th],
                                        'from_country_id', 'to_country_id',
                                        edge_attr=['pct_points_from','total_points'],
                                        )

    #Create a dictionary mapping each country to its country_id
    country_name_mapping = dict(zip(country_votes_filter_df['from_country_id'], country_votes_filter_df['from_country']))
    country_id_mapping = dict(zip(country_votes_filter_df['from_country_id'], country_votes_filter_df['from_country']))
    country_points_to_mapping = dict(zip(country_points_metadata['country_id'], country_points_metadata['Total points received']))
    country_points_from_mapping = dict(zip(country_points_metadata['country_id'], country_points_metadata['Total points given']))


    # Add country_id and name as node attributes
    nx.set_node_attributes(votes_graph, country_name_mapping, 'country_name')
    nx.set_node_attributes(votes_graph, country_id_mapping, 'country_id')
    nx.set_node_attributes(votes_graph, country_id_mapping, 'label')
    nx.set_node_attributes(votes_graph, country_group_mapping, 'Region')
    nx.set_node_attributes(votes_graph, country_points_to_mapping, 'Total points received')
    nx.set_node_attributes(votes_graph, country_points_from_mapping, 'Total points given')

    # Compute positions for viz.
    pos = nx.spring_layout(votes_graph, seed=1999)

    #Get edges weights 
    weights = list(nx.get_edge_attributes(votes_graph,'pct_points_from').values())

    # Draw the graph using Altair
    geo_graph = nxa.draw_networkx(
            votes_graph, pos=pos,
            width='pct_points_from',
            node_color = 'Region',
            cmap='set3',
            edge_color='#E6E6FA', 
            node_tooltip=[alt.Tooltip('country_name:N', title='Country'), alt.Tooltip('Total points received:Q'),
            alt.Tooltip('Total points given:Q')]
        ).properties(
            width=600,
            height=400
        ).configure_view(
                stroke=None  # Remove the stroke
            ).configure_axis(
                domain=False  # Remove all axes
            ).interactive().configure_legend(
            titleFontSize=15,
            labelFontSize=15) 
    
    col2.altair_chart(geo_graph, use_container_width=True, theme=None)

    #Neighbors voting plot ------------------------
    st.subheader('Analysis by geographical regions')

    def distance_btw_countries(country_a, country_b):
        """Calculate shortest distance beween two countries"""
        if (country_a in gdf_ne["country"].values) and (country_b in gdf_ne["country"].values):
            #Get country geometry
            country_a_geom = gdf_ne[gdf_ne["country"] == country_a]["geometry"].iloc[0]
            country_b_geom = gdf_ne[gdf_ne["country"] == country_b]["geometry"].iloc[0]
            #Find nearest points between borders 
            dot_a, dot_b = nearest_points(country_a_geom, country_b_geom)
            #Calculate distance between points
            distance = dot_a.distance(dot_b)
        else:
            distance = np.nan
        return distance

    region_color_map = {
        'Balcans': eurovision_palette[1],
        'Eastern Europe': eurovision_palette[2],
        'Mediterranean': eurovision_palette[0],
        'Middle East-Caucasus': eurovision_palette[4],
        'Northern Europe': eurovision_palette[5],
        'Western Europe': eurovision_palette[3]
    }

    #Calculate distance between countries
    url = "https://raw.githubusercontent.com/asolbas/eurovision-dataviz/main/Data/world.geojson"
    gdf_ne = gpd.read_file(url)  # zipped shapefile
    gdf_ne = gdf_ne[["name", 'geometry', 'continent', 'type']]
    gdf_ne.columns = ["country", 'geometry', 'continent', 'type']
    country_votes_dist_df = country_votes_filter_df[country_votes_filter_df['from_country'] != country_votes_filter_df['to_country']]
    country_votes_dist_df.loc[:,'distance'] = country_votes_dist_df.apply(lambda row: distance_btw_countries(row['from_country'], row['to_country']), axis=1)
    country_votes_dist_df = country_votes_dist_df.dropna(subset=['distance'])

    #Filter Australia
    country_votes_dist_df = country_votes_dist_df[country_votes_dist_df['from_country']!='Australia']
    country_votes_dist_df = country_votes_dist_df[country_votes_dist_df['to_country']!='Australia']

    #Select country group
    country_votes_dist_df['from_country_group'] = country_votes_dist_df['from_country_id'].replace(country_group_mapping)
    country_votes_dist_df = country_votes_dist_df.rename(columns={'from_country_group':'Region'})
    selection = alt.selection_point(fields=['Region'], empty=True)
    color = alt.condition(
        selection,
        alt.Color('Region:N', scale=alt.Scale(domain=list(region_color_map.keys()),
                range=list(region_color_map.values()))).legend(None),
        alt.value('rgba(211, 211, 211, 0)')
    )

    #Scatter plot
    distance_plot = alt.Chart(country_votes_dist_df).mark_circle(size=60).encode(
            x=alt.X('distance:Q').title('Distance (degrees)'),
            y=alt.Y('pct_points_from:Q').title('Percentage of points given'),
            color = color,
            tooltip=[alt.Tooltip('from_country:N', title='From'), alt.Tooltip('to_country:N', title='To'), 
            alt.Tooltip('pct_points_from:Q', title='Percentage of points', format='.2f'), 
            alt.Tooltip('distance:Q', title='Distance', format='.2f')]
    ).properties(
                width=500,
                height=250,
                title=alt.TitleParams(
            text='Do neighbors vote each other?',
            fontSize=20  # Increase title font size
        ))

    legend = alt.Chart(country_votes_dist_df).mark_point().encode(
        alt.Y('Region:N', title='Geographical region').axis(orient='right'),
        color=color
    ).add_params(
        selection
    )

    reg_line = distance_plot.transform_regression('distance', 'pct_points_from').mark_line(
        color='white', size=5)

    #Average points recieved by country
    country_votes_scaled = country_votes_df.copy()
    country_votes_scaled.loc[country_votes_scaled['year'] >= 2015, 'total_points'] /= 2
    points_country = country_votes_scaled.groupby(['to_country', 'to_country_group'])['total_points'].mean().sort_values(ascending=False)
    points_country = points_country.reset_index()
    points_country.columns = ['Country','Region', 'Avg_points']

    # Calculate median points by region
    median_points_by_region = points_country.groupby('Region')['Avg_points'].median().reset_index()
    median_points_by_region = median_points_by_region.sort_values('Avg_points', ascending=False)

    # Order regions by median points
    region_order = median_points_by_region['Region'].tolist()

    boxplot = alt.Chart(points_country).mark_boxplot(ticks={'color':'white'}, size=50, color='white').encode(
            x=alt.X("Region:N", sort=region_order, axis=alt.Axis(labelAngle=-45)), 
            y=alt.Y("Avg_points:Q").title('Average points received per country'), 
            color = alt.Color("Region:N").scale(domain=list(region_color_map.keys()),
                range=list(region_color_map.values())).legend(None),
        ).properties(
            width=400, 
            height=300,
            title=alt.TitleParams(
            text='Popularity ranking (regions)',
            fontSize=20  # Increase title font size
        )
        )

    #Country ranking
    selection = alt.selection_point(fields=['Region'], bind='legend')
    country_ranking = alt.Chart(points_country).mark_bar().encode(
        x=alt.X("Country:N", sort='-y', axis=alt.Axis(labelAngle=-45)), 
        y=alt.Y("Avg_points:Q", scale=alt.Scale(domain=[0, 7])).title('Average points received per country'), 
        color=alt.Color("Region:N", scale=alt.Scale(domain=list(region_color_map.keys()),
                range=list(region_color_map.values()))).title('Region')
    ).add_params(
        selection
    ).transform_filter(
        selection
    ).transform_window(
        rank='rank(Avg_points)',
        sort=[alt.SortField('Avg_points', order='descending')]
    ).transform_filter(
    alt.datum.rank <= 15
    ).properties(
        width=400,
        height=250,
        title=alt.TitleParams(
            text='Popularity ranking (countries)',
            fontSize=20  # Increase title font size
        )
    )

    ranking_selector = st.radio(
        "Select the level of detail",
        ["Regions", "Countries"],
        horizontal=True)

    if ranking_selector == "Regions":
        st.info('Australia is not displayed in the distance plot for being the only country that is not part of Eurasia.', icon="ℹ️")
        st.altair_chart((boxplot | distance_plot + reg_line| legend).resolve_scale(
            color='independent'
        ).configure_axis(
            labelFontSize=16,
            titleFontSize=16
        ).configure_legend(
        titleFontSize=15,
        labelFontSize=15, 
        orient='bottom'),  theme=None, use_container_width=True)
    else:
        st.info('Australia is not displayed in the distance plot for being the only country that is not part of Eurasia.', icon="ℹ️")
        st.altair_chart((country_ranking | distance_plot + reg_line| legend).resolve_scale(
            color='independent'
        ).configure_axis(
            labelFontSize=16,
            titleFontSize=16
        ).configure_legend(
        titleFontSize=15,
        labelFontSize=15, 
        orient='right'),  theme=None, use_container_width=True)


#WHAT MAKES A SONG SO GOOD-----------------------
if selected == 'Music':
    st.title('What makes a song SO good? (2009-2023)')

    group_selection = st.selectbox("Select classification group", 
                                   options=['All', 'Winner', 'Top5', 'Top10', 
                                            'Finalist', 'Semifinalist'])


    if group_selection == 'Winner':
        songs_filt_df = songs_df[songs_df['classification']=='Winner']
        contestants_filt_df = contestants_df[contestants_df['classification']=='Winner']
    elif group_selection == 'Top5':
        songs_filt_df = songs_df[songs_df['classification'].isin(['Winner', 'Top5'])]
        contestants_filt_df = contestants_df[contestants_df['classification'].isin(['Winner', 'Top5'])]
    elif group_selection == 'Top10':
        songs_filt_df = songs_df[songs_df['classification'].isin(['Winner', 'Top5', 'Top10'])]
        contestants_filt_df = contestants_df[contestants_df['classification'].isin(['Winner', 'Top5', 'Top10'])]
    elif group_selection == 'Finalist':
        songs_filt_df = songs_df[songs_df['classification'].isin(['Winner', 'Top5', 'Top10', 'Finalist'])]
        contestants_filt_df = contestants_df[contestants_df['classification'].isin(['Winner', 'Top5', 'Top10', 'Finalist'])]
    elif group_selection == 'Semifinalist':
        songs_filt_df = songs_df[songs_df['classification'].isin(['Semifinalist'])]
        contestants_filt_df = contestants_df[contestants_df['classification'].isin(['Semifinalist'])]
    else:
        songs_filt_df = songs_df
        contestants_filt_df = contestants_df

    col1, col2 = st.columns(2)

    #Features radar plot ----------------------
    col1.header('Mean song features values')

    #Calculate the mean for each characteristic
    features_df = songs_filt_df[['energy', 'danceability', 'happiness', 'acousticness', 
                            'liveness', 'speechiness']].melt(
                                var_name='Feature', value_name='Value')
    features_df = features_df.groupby('Feature',as_index=False)['Value'].mean()

    #Plot mean values
    fig = px.line_polar(features_df, r='Value', theta='Feature', line_close=True,
                    template = 'plotly_dark', color_discrete_sequence = eurovision_palette)
    fig.update_traces(fill='toself')
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 80],
                tickfont=dict(size=15)  # Aumentar tamaño de fuente del título del eje radial
            ),
            angularaxis=dict(
                tickfont=dict(size=15)  # Aumentar tamaño de fuente del título del eje angular
            )
        ),
        showlegend=False,
        width=500,
        height=500,
        font=dict(
            size=18  # Aumentar tamaño de fuente general
        )
    )

    col1.plotly_chart(fig, use_container_width=True, theme=None)

    #Lyrics wordcloud -----------------------
    col2.header('Most frequent words in lyrics')
    maximum = col2.select_slider('**Select maximum number of words to display**', 
                               options = [1,25,50,75,100,125,150],
                               value=100)

    # Define stopwords for multiple languages 
    stop_words = set(stopwords.words())
    # Function to remove stopwords
    def remove_stopwords(text):
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered_tokens)

    # Function to lemmatize text
    def lemmatize(text):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatized_text = [lemmatizer.lemmatize(word) for word in text.split()]
        return ' '.join(lemmatized_text)

    # Function to clean sentences
    def clean_sent(text):
        return ' '.join([word for word in text.split() if len(word) > 1])
    
    def full_cleaning(frame,col):
        newframe=frame.copy()  
        newframe[col] = newframe[col].str.replace('\d+', '',regex=True).str.replace('?', '', regex=True
            ).str.replace('\W', ' ', regex=True).str.replace("\n", '', regex=True
            ).str.replace('\b(n\w+)\b', '', regex=True).str.lower().str.strip()
        newframe = newframe.astype('str')
        text = ' '.join(newframe[col][:])
        cleantext = remove_stopwords(text)
        words = set(nltk.corpus.words.words())
        cleantext=lemmatize(cleantext) 
        cleantext = clean_sent(cleantext)
        return cleantext
    
    def color_wordcloud(word, font_size, position,orientation,random_state=None, **kwargs):
        random_color = np.random.choice(eurovision_palette[:-4])
        return random_color

    #cloudtext is going to be imported from .txt since NLTK package
    #is not working properly in Streamlit Cloud.
    #Thus, all the preprocessing was done locally and save as .txt
    with open("./Data/wordcloud.txt", 'r',  encoding="utf8") as file:
        cloudtext = file.read().rstrip()
    #cloudtext=full_cleaning(contestants_filt_df[['lyrics']],col='lyrics')
    wordcloud = WordCloud(max_font_size=50, max_words=maximum, background_color="black",collocations=False
        ).generate(cloudtext)
    #wordcloud.generate_from_frequencies
    plt.style.use("seaborn-white")
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(wordcloud.recolor(color_func=color_wordcloud), interpolation="bilinear")
    ax.axis("off")

    col2.pyplot(fig, use_container_width=True)

    
    #Piecharts ------------------------
    st.header('Some numbers...')
    col1, col2, col3, col4, col5 = st.columns(5)

    #Set color palettes
    style_color_map = {
        'Pop': eurovision_palette[0],
        'Dance': eurovision_palette[1],
        'Ballad': eurovision_palette[2],
        'Traditional': eurovision_palette[3],
        'Rock': eurovision_palette[4],
        'Opera': eurovision_palette[5]
    }

    gender_color_map = {
        'Female': eurovision_palette[0],
        'Male': eurovision_palette[1],
        'Mix': eurovision_palette[2],
    }


    #Gender
    counts_df = songs_filt_df.groupby('gender').size().reset_index()
    counts_df.columns = [counts_df.columns[0], 'count']
    total_count = counts_df['count'].sum()
    counts_df['percentage'] = (counts_df['count'] / total_count) * 100
    gender_plot = alt.Chart(counts_df).mark_arc().encode(
            theta="count",
            color=alt.Color("gender",scale=alt.Scale(domain=list(gender_color_map.keys()),
            range=list(gender_color_map.values()))).title('Gender'),
            tooltip=[alt.Tooltip('gender:N', title='Gender'), alt.Tooltip('count:Q', title='Number of songs'), 
                alt.Tooltip('percentage', title='Percentage', format='.1f')]
        ).properties(
                width=600,
                height=400,
            title='Main singer(s) gender'
            ).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
        titleFontSize=15,
        labelFontSize=15
        ).configure_title(fontSize=24) 
    #Music style
    counts_df = songs_filt_df.groupby('style').size().sort_values().reset_index()
    counts_df.columns = [counts_df.columns[0], 'count']
    total_count = counts_df['count'].sum()
    counts_df['percentage'] = (counts_df['count'] / total_count) * 100
    style_plot = alt.Chart(counts_df).mark_arc().encode(
            theta=alt.Theta("count:Q", sort='ascending'),
            color=alt.Color("style",scale=alt.Scale(domain=list(style_color_map.keys()),
            range=list(style_color_map.values()))).title('Style'),
           tooltip=[alt.Tooltip('style:N', title='Style'), alt.Tooltip('count:Q', title='Number of songs'), 
                alt.Tooltip('percentage', title='Percentage', format='.1f')]
            ).properties(
                width=600,
                height=400,
            title='Music style'
        ).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
        titleFontSize=15,
        labelFontSize=15
        ).configure_title(fontSize=24) 
        
    #Dancers
    counts_df = songs_filt_df.groupby('backing_dancers').size().sort_values().reset_index()
    counts_df.columns = [counts_df.columns[0], 'count']
    total_count = counts_df['count'].sum()
    counts_df['percentage'] = (counts_df['count'] / total_count) * 100
    dancers_plot = alt.Chart(counts_df).mark_arc().encode(
            theta=alt.Theta("count:Q", sort='ascending'),
            color=alt.Color("backing_dancers:Q").scale(domain=[0, 5], range=eurovision_cont_palette, reverse=True).title('Number'),
            tooltip=[alt.Tooltip('backing_dancers:Q', title='Number of backing dancers'), alt.Tooltip('count:Q', title='Number of songs'), 
                alt.Tooltip('percentage', title='Percentage', format='.1f')]
            ).properties(
                width=500,
                height=400,
                title='Backing dancers'
            ).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
        titleFontSize=15,
        labelFontSize=15
        ).configure_title(fontSize=24) 
    
    #Singers
    counts_df = songs_filt_df.groupby('backing_singers').size().sort_values().reset_index()
    counts_df.columns = [counts_df.columns[0], 'count']
    total_count = counts_df['count'].sum()
    counts_df['percentage'] = (counts_df['count'] / total_count) * 100
    singers_plot = alt.Chart(counts_df).mark_arc().encode(
            theta=alt.Theta("count:Q", sort='ascending'),
            color=alt.Color("backing_singers:Q").scale(domain=[0, 5], range=eurovision_cont_palette, reverse=True).title('Number'),
            tooltip=[alt.Tooltip('backing_singers:Q', title='Number of backing singers'), alt.Tooltip('count:Q', title='Number of songs'), 
                alt.Tooltip('percentage', title='Percentage', format='.1f')]
            ).properties(
                width=500,
                height=400,
                title='Backing singers'
            ).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
        titleFontSize=15,
        labelFontSize=15
        ).configure_title(fontSize=24) 
    
    #Instruments
    counts_df = songs_filt_df.groupby('backing_instruments').size().sort_values().reset_index()
    counts_df.columns = [counts_df.columns[0], 'count']
    total_count = counts_df['count'].sum()
    counts_df['percentage'] = (counts_df['count'] / total_count) * 100
    instruments_plot = alt.Chart(counts_df).mark_arc().encode(
            theta=alt.Theta("count:Q", sort='ascending'),
            color=alt.Color("backing_instruments:Q").scale(domain=[0, 5], range=eurovision_cont_palette, reverse=True).title('Number'),
           tooltip=[alt.Tooltip('backing_instruments:Q', title='Number of backing instruments'), alt.Tooltip('count:Q', title='Number of songs'), 
                alt.Tooltip('percentage', title='Percentage', format='.1f')]
            ).properties(
                width=500,
                height=400,
                title='Backing instruments'
            ).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
        titleFontSize=15,
        labelFontSize=15
        ).configure_title(fontSize=24) 
    col1.altair_chart(gender_plot, use_container_width=True, theme=None)
    col2.altair_chart(style_plot, use_container_width=True, theme=None)
    col3.altair_chart(singers_plot, use_container_width=True, theme=None)
    col4.altair_chart(dancers_plot, use_container_width=True, theme=None)
    col5.altair_chart(instruments_plot, use_container_width=True, theme=None)

#IS EUROVISION PREDICTABLE? ---------------------
if selected == 'Voting':
    st.title('Is the ESC predictable?')
    col1, col2 = st.columns(2)

    #Betting houses ---------------------
    #Get only songs that got into the final
    bets_rank_df = bets_df[bets_df['contest_round']=='final'].merge(contestants_df[['year','to_country', 'place_contest', 'points_final', 'points_tele_final', 'points_jury_final']], how='left',
    left_on=['year','country_name'], right_on=['year','to_country'])
    bets_rank_df = bets_rank_df.rename(columns={'points_tele_final':'Televote', 'points_jury_final':'Jury'})
    bets_rank_df = bets_rank_df[bets_rank_df['place_contest'].notnull()]
    bets_rank_df = bets_rank_df[bets_rank_df['Televote'].notnull()]
    bets_rank_df['country_group'] = bets_rank_df['country_code'].replace(country_group_mapping)
    bets_points_df = pd.melt(bets_rank_df, id_vars=['country_name','betting_name', 'betting_score', 'year'], value_vars=['Televote', 'Jury'],
        var_name='type', value_name='points')

    mean_odds_df = bets_points_df.groupby(['year', 'country_name', 'type', 'points']).agg({'betting_score': 'mean'}).reset_index()

    # Define the selection
    selection = alt.selection_point(fields=['type'], bind='legend')
    voting_color_map = {
        'Jury': eurovision_palette[0],
        'Televote': eurovision_palette[1]
    }

    # Create the scatter plot using Altair
    scatter_plot = alt.Chart(mean_odds_df).mark_circle(size=60).encode(
        x=alt.X('betting_score:Q').title('Mean betting score'),
        y=alt.Y('points:Q', scale=alt.Scale(domain=[0, 450])).title('Points in the final'),
        color=alt.Color('type:N',scale=alt.Scale(domain=list(voting_color_map.keys()),
            range=list(voting_color_map.values()))).title('Type of votes'),
        tooltip=[alt.Tooltip('year:Q', title='Year'), alt.Tooltip('country_name:N', title='Country'), 
            alt.Tooltip('betting_score:Q', format='.2f', title='Betting score')]
    ).properties(
        title='Mean Betting Odds vs Points received (by Country and Year)'
    ).properties(
        width=600,
        height=400
    ).add_params(
        selection
    ).transform_filter(
        selection
    ).configure_axis(
            labelFontSize=16,
            titleFontSize=16
        ).configure_legend(
        titleFontSize=15,
        labelFontSize=15,
        )
    # Show the plot
    col1.header('Betting houses predictive power')
    col1.altair_chart(scatter_plot, use_container_width=True, theme=None) 

    #Running order effect-----------------------
    #Drops songs that didn't participate in the final
    finalists_df = contestants_df[~contestants_df['running_final'].isna()]
    #Select data from 2016 to 2023
    finalists_df = finalists_df[finalists_df['year'] >= 2016]
    #Determine if they performed in the first or second half
    finalists_df['final_half'] = pd.cut(finalists_df['running_final'], bins=[0, 14, 27], labels=['1st', '2nd'], right=False)
    points_df = finalists_df[['final_half','points_tele_final', 'points_jury_final']]
    points_df.columns = ['final_half', 'Televote', 'Jury']
    points_half = pd.melt(points_df, 
                id_vars='final_half', value_vars=['Televote', 'Jury'],
                var_name='voting_type', value_name='points')
    #Distribution of points according to the running order 
    boxplot_running_order = alt.Chart(points_half).mark_boxplot(ticks={'color':'white'}, size=50, color='white').encode(
            x=alt.X("voting_type:O", title=None, axis=alt.Axis(labels=False, ticks=False), scale=alt.Scale(padding=1)), 
            y=alt.Y("points:Q").title('Points'), 
            color=alt.Color("voting_type:N", scale=alt.Scale(range=eurovision_palette)).title('Voting type'),
            column=alt.Column('final_half:N', header=alt.Header(orient='bottom')).title('Final half')
        ).properties(
            width=250, 
            height=300, 
            title='Distribution of points by running order'
        ).configure_facet(
            spacing=0
        ).configure_view(
            stroke=None
        ).configure_axis(
            labelFontSize=16,
            titleFontSize=16
        ).configure_legend(
        titleFontSize=15,
        labelFontSize=15,
        )
    # Show the plot
    col2.header('Running order effect on votes')
    col2.altair_chart(boxplot_running_order, use_container_width=False, theme=None) 

    #Jury vs Televote -----------------

    votes_df = contestants_df[
    contestants_df.year >= 2016][['year', 'to_country', 'points_jury_final', 'points_tele_final', 'place_contest']].dropna()
    votes_df.columns = ['year', 'country', 'points_jury_final', 'points_tele_final', 'place_contest']
    votes_df['year'] = votes_df['year'].astype('str')

    #Scatter plot
    jury_vs_tele = alt.Chart(votes_df).mark_circle(size=60).encode(
            x=alt.X('points_jury_final').title('Points Jury'),
            y=alt.Y('points_tele_final').title('Points Televote'),
            color=alt.Color('place_contest',
                    scale=alt.Scale(range=eurovision_cont_palette, reverse=True),
                    title='Place in final',
                    legend=alt.Legend(orient='right', titleFontSize=12, labelFontSize=10)  # Orientación y tamaño de la leyenda
                   ),
            tooltip=[alt.Tooltip('year:Q', title='Year'), alt.Tooltip('country:N', title='Country'), 
                alt.Tooltip('points_jury_final:Q', title='Points Jury'), alt.Tooltip('points_tele_final:Q', title='Points Televote'), 
                alt.Tooltip('place_contest:Q', title='Place in contest')]
        ).properties(
        width=300,
        height=250
    )

    #Ranking Jury and Televote -------------------------
    total_points_country = votes_df.groupby(['country'], as_index=False)[['points_jury_final', 'points_tele_final']].sum()
    # create a bar chart with a ranking
    ranking_jury = (
        alt.Chart(total_points_country).mark_bar(color = eurovision_palette[0]).encode(
            x=alt.X("points_jury_final").title('Points'),
            y=alt.Y("country").title('Country').sort("-x"),
        ).transform_window(
        rank='rank(points_jury_final)',
        sort=[alt.SortField('points_jury_final', order='descending')]
    ).transform_filter(
        alt.datum.rank <= 10).properties(
            width=300,
            height=250,
            title="Most voted by Jury"
        )
    )

    ranking_tele = (
        alt.Chart(total_points_country).mark_bar(color = eurovision_palette[1]).encode(
            x=alt.X("points_tele_final").title('Points'),
            y=alt.Y("country").title('Country').sort("-x"),
        ).transform_window(
        rank='rank(points_tele_final)',
        sort=[alt.SortField('points_tele_final', order='descending')]
    ).transform_filter(
        alt.datum.rank <= 10).properties(
            width=300,
            height=250,
            title="Most voted by Televote"
        )
    )

    st.header('Do Jury and Televote agree?')
    st.altair_chart(alt.hconcat(jury_vs_tele,  ranking_jury | ranking_tele).resolve_legend(color='independent')
        , use_container_width=True, theme=None) 
