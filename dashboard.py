import streamlit as st
from streamlit_option_menu import option_menu

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

#PAGE LAYOUT
st.set_page_config(layout="wide")
#st.title("Analysis of the Eurovision Song Contest")

#Add menu
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
    contestants_df.drop_duplicates(inplace=True)
    contestants_df.loc[contestants_df['to_country']=='Andorra','to_country_id'] = 'ad'
    contestants_df.loc[contestants_df['to_country_id']=='mk','to_country'] = 'North Macedonia'
    contestants_df.loc[contestants_df['to_country']=='Czechia','to_country'] = 'Czech Republic'

    #Add song language
    results_df = pd.read_csv('./Data/Every_Eurovision_Result_Ever.csv')
    results_df['Song'] = results_df['Song'].str.title()
    contestants_df = contestants_df.merge(results_df[['Country', 'Year', 'Language', 'Song']], left_on=['year', 'to_country', 'song'],
                                        right_on=['Year', 'Country', 'Song'], how='left')
    contestants_df['finalist'] = 0
    contestants_df.loc[~contestants_df['place_final'].isna(), 'finalist'] = 1
    contestants_df['country'] = contestants_df['to_country']

    #Drop 2020
    contestants_df = contestants_df[contestants_df['year']!=2020]
    contestants_df['year'] = pd.to_datetime(contestants_df['year'], format='%Y')

    #Votes ------------------------
    country_votes_df = pd.read_csv('./Data/votes.csv')
    # Import country-geographical group mapping
    with open('./Data/country_groups_mapping.json', 'r') as file:
        country_group_mapping = json.load(file)

    # Add geographical group 
    country_votes_df['from_country_group'] = country_votes_df['from_country'].replace(country_group_mapping)
    country_votes_df['to_country_group'] = country_votes_df['to_country'].replace(country_group_mapping)

    #Calculate pertentage of points given and received
    country_votes_filter_df = country_votes_df[country_votes_df['round']=='final'][['year', 'from_country', 'to_country', 'total_points']].dropna()
    country_votes_filter_df = country_votes_filter_df.groupby(['from_country', 'to_country'], as_index=False)['total_points'].sum()
    country_votes_filter_df = country_votes_filter_df.merge(contestants_df[['to_country_id', 'to_country']].drop_duplicates(), how='left',
                                                            left_on='from_country', right_on='to_country_id').drop('to_country_id',axis=1)
    country_votes_filter_df.columns = ['from_country', 'to_country', 'total_points', 'from_country_name']
    country_votes_filter_df = country_votes_filter_df.merge(contestants_df[['to_country_id', 'to_country']].drop_duplicates(), how='left',
                                                            left_on='to_country', right_on='to_country_id').drop('to_country_id',axis=1)
    country_votes_filter_df.columns = ['from_country', 'to_country', 'total_points', 'from_country_name', 'to_country_name']
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

    #Songs ----------------------
    songs_df = pd.read_csv('./Data/song_data.csv', na_values=['-', 'unknown'])
    songs_df = songs_df[songs_df['year']!=2020]

    #Create a classification column to group songs according to their position
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

    songs_df['classification'] = songs_df['final_draw_position'].apply(classification_group)

#EUROVISION IN A NUTSHELL-----------------------
if selected == 'Overview':
    #st.header('Eurovision in a Nutshell')
    st.title('Eurovision in a Nutshell')

    #General information about the contest
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

    # Evolución del número de participantes y puntos otorgados
    #Number of participants
    participants_year = contestants_df.groupby(['year', 'finalist'], as_index=False)['to_country_id'].nunique()
    participants_year.columns = ['year', 'finalist', 'n_participants']
    participants_year.loc[participants_year['finalist']==0,'finalist'] = 'No'
    participants_year.loc[participants_year['finalist']==1,'finalist'] = 'Yes'

    #Total points
    points_year = contestants_df.pivot_table(index='year', 
                                        values=['points_final'], 
                                        aggfunc='sum').stack().reset_index()

    points_year.columns = ['year', 'source', 'points']
    points_year.loc[points_year['points'] == 0, 'points'] = np.nan

    #Combine
    contestants_year = participants_year.merge(points_year, on='year')

    base = alt.Chart(contestants_year).encode(alt.X("year:T", axis=alt.Axis(format='%Y')).title('Years'))

    bar = base.mark_bar().encode(
        alt.Y('n_participants').title('Number of participant countries'),
        color=alt.Color("finalist:N").title('Classified for the final')
        )

    line = base.mark_line(color='white').encode(
        alt.Y('points').title('Total points')    
        )

    points_plot = alt.layer(bar, line).resolve_scale(
            y='independent'
        ).properties(
            width=600,
            height=300
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
                        'Spanish', 'Portuguese', 'Greek', 'Swedish', 'English + other language']
    minoritary_languages_ls = contestants_df[~contestants_df['Language'].isin(common_languages_ls)]['Language'].unique()
    # Replace minoritary languages with "Other Languages"
    contestants_df['Language_red'] = contestants_df['Language'].copy()
    contestants_df.loc[contestants_df['Language'].isin(minoritary_languages_ls), 'Language_red'] = 'Other Languages'
    contestants_language_df = contestants_df.groupby(['year', 'Language_red'], as_index=False).size()
    #Plot languages
    languages_plot =  alt.Chart(contestants_language_df).mark_area().encode(
                x=alt.X("year:T", axis=alt.Axis(format='%Y')).title('Year'),
                y=alt.Y('size:Q').title('Number of entries'),
                color=alt.Color("Language_red:N").title('Language')).properties(
                    width=600,
                    height=300
            )
    col1.subheader('Evolution of the songs languages')
    col1.altair_chart(languages_plot, use_container_width=True, theme=None)


    # Trayectoria de cada país en el concurso ------------------------
    click_state = alt.selection_point(fields=['country'])
    color = alt.condition(
        click_state,
        alt.Color('country:N').legend(None),
        alt.value('rgba(211, 211, 211, 0)')
    )

    points_plot = alt.Chart(contestants_df).mark_line(point=True).encode(
        x=alt.X("year:T", axis=alt.Axis(format='%Y')).title('Years'),
        y=alt.Y('place_contest',scale=alt.Scale(domain=(45, 1))).title('Place in contest'),
        color=color,
        tooltip=['points_final:N', 'song:N', 'performer:N']
    ).properties(
        width=500,
        height=300
    )

    legend = alt.Chart(contestants_df).mark_point().encode(
        alt.Y('country:N', title='Country').axis(orient='right'),
        color=color
    ).add_params(
        click_state
    )


    ## Number of entries per country map
    entries_country = contestants_df.groupby(['to_country'], as_index=False)['to_country'].value_counts()
    entries_country.columns = ['country',  'n_entries']
    entries_country = entries_country.sort_values(by='n_entries', ascending=False)
    europe_url = "https://raw.githubusercontent.com/asolbas/eurovision-dataviz/main/Data/europe_map.geojson"
    worldmap = alt.Data(url=europe_url, format=alt.DataFormat(property='features',type='json'))

    choropleth = (
        alt.Chart(worldmap)
        .mark_geoshape()
        .transform_lookup(
            lookup="properties.country", from_=alt.LookupData(entries_country, "country", ["country", "n_entries"])
        )
        .encode(
            color=alt.Color("n_entries:Q").title('Number of entries'),
            opacity=alt.condition(click_state, alt.value(1), alt.value(0.2)),
            tooltip=["country:N", "n_entries:Q"],
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
            color=alt.Color("n_entries:Q").title('Number of entries'),
            opacity=alt.condition(click_state, alt.value(1), alt.value(0.2)),
            tooltip=["country:N", "n_entries:Q"],
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

    trajectory_plot = (alt.layer(choropleth,choropleth_australia) & points_plot).add_params(click_state)

    col2.subheader('Trajectory of each country')
    col2.altair_chart(trajectory_plot, use_container_width=True, theme=None)

#GEOPOLITICS IN EUROVISION -----------------------
if selected == 'Geopolitics':
    st.title('Geopolitics in Eurovision')

    #Plots
    col1, col2 = st.columns(2)

    #Who is the most popular? ----------------------

    ## Number of wins per country
    wins_country = contestants_df[contestants_df['place_contest']==1.0].groupby(['country'], as_index=False)['country'].value_counts()
    wins_country.columns = ['country',  'n_wins']
    wins_country = wins_country.merge(contestants_df['country'], on='country', how='right').fillna(0).drop_duplicates()
    wins_country['n_wins'].unique()

    # define a pointer selection
    click_state = alt.selection_point(fields=["country"], toggle=False)

    europe_url = "https://raw.githubusercontent.com/asolbas/eurovision-dataviz/main/Data/europe_map.geojson"
    worldmap = alt.Data(url=europe_url, format=alt.DataFormat(property='features',type='json'))
    choropleth = (
        alt.Chart(worldmap)
        .mark_geoshape()
        .transform_lookup(
            lookup="properties.country", from_=alt.LookupData(wins_country, "country", ["country", "n_wins"])
        )
        .encode(
            color=alt.Color("n_wins:Q").title('Number of wins'),
            opacity=alt.condition(click_state, alt.value(1), alt.value(0.2)),
            tooltip=["country:N", "n_wins:Q"],
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
            lookup="properties.country", from_=alt.LookupData(wins_country, "country", ["country", "n_wins"])
        ).encode(
            color=alt.Color("n_wins:Q").title('Number of wins'),
            opacity=alt.condition(click_state, alt.value(1), alt.value(0.2)),
            tooltip=["country:N", "n_wins:Q"],
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
    aux_df['country'] = country_votes_filter_df['to_country_name']
    most_votes_from = (
        alt.Chart(
            aux_df).mark_bar().encode(
            x=alt.X("pct_points_to").title('Points received (percentage)'),
            y=alt.Y("from_country_name").title('Country').sort("-x"),
        ).transform_filter(
        click_state
        ).transform_window(
        rank='rank(pct_points_to)',
        sort=[alt.SortField('pct_points_to', order='descending')]
    ).transform_filter(
        alt.datum.rank <= 5).properties(
            width=150,
            title="Most voted by"
        )
    )

    aux_df = country_votes_filter_df.copy()
    aux_df['country'] = country_votes_filter_df['from_country_name']
    most_votes_to = (
        alt.Chart(
            aux_df).mark_bar().encode(
            x=alt.X("pct_points_from").title('Points given (percentage)'),
            y=alt.Y("to_country_name").title('Country').sort("-x"),
        ).transform_filter(
        click_state,
        ).transform_window(
        rank='rank(pct_points_from)',
        sort=[alt.SortField('pct_points_from', order='descending')]
    ).transform_filter(
        alt.datum.rank <= 5).properties(
            width=150,
            title="Most votes to"
        )
    )

    popularity_plot = alt.vconcat(alt.layer(choropleth,choropleth_australia), 
                    alt.hconcat(most_votes_to, most_votes_from)
                    ).add_params(click_state)
    
    col1.subheader('Who is the most popular?')
    col1.altair_chart(popularity_plot, use_container_width=True, theme=None)

    #Friendzone plot ----------------------------
    col2.subheader('Friendzone graph')

    #Percentage threshold 
    #Add an interactive bar to select percentage threshold
    th = col2.select_slider('**Select a percentage threshold**', 
                               options = np.arange(0,20.5,0.5),
                               value=6)
    votes_graph = nx.from_pandas_edgelist(country_votes_filter_df[country_votes_filter_df['pct_points_from']>th],
                                        'from_country', 'to_country',
                                        edge_attr=['pct_points_from','total_points'])

    #Create a dictionary mapping each country to its country_id
    country_name_mapping = dict(zip(country_votes_filter_df['from_country'], country_votes_filter_df['from_country_name']))
    country_id_mapping = dict(zip(country_votes_filter_df['from_country'], country_votes_filter_df['from_country']))

    # Add country_id and name as node attributes
    nx.set_node_attributes(votes_graph, country_name_mapping, 'country_name')
    nx.set_node_attributes(votes_graph, country_id_mapping, 'country_id')
    nx.set_node_attributes(votes_graph, country_id_mapping, 'label')
    nx.set_node_attributes(votes_graph, country_group_mapping, 'country_group')

    # Compute positions for viz.
    pos = nx.spring_layout(votes_graph, seed=1999)

    #Get edges weights 
    weights = list(nx.get_edge_attributes(votes_graph,'pct_points_from').values())

    # Draw the graph using Altair
    geo_graph = nxa.draw_networkx(
            votes_graph, pos=pos,
            width='pct_points_from',
            node_color = 'country_group',
            edge_tooltip=['pct_points_from', 'total_points'],
            node_tooltip=['country_name'],
            edge_color='#E6E6FA',
        ).properties(
            width=600,
            height=400
        ).configure_view(
            stroke=None  # Remove the stroke
        ).configure_axis(
            domain=False  # Remove all axes
        ).interactive()
    
    col2.altair_chart(geo_graph, use_container_width=True, theme=None)

    #Neighbors voting plot ------------------------
    st.subheader('Do neighbors vote each other?')

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

    #Calculate distance between countries
    url = "./Data/world.geo.json"
    gdf_ne = gpd.read_file(url)  # zipped shapefile
    gdf_ne = gdf_ne[["name", 'geometry', 'continent', 'type']]
    gdf_ne.columns = ["country", 'geometry', 'continent', 'type']
    country_votes_dist_df = country_votes_filter_df[country_votes_filter_df['from_country'] != country_votes_filter_df['to_country']]
    country_votes_dist_df.loc[:,'distance'] = country_votes_dist_df.apply(lambda row: distance_btw_countries(row['from_country_name'], row['to_country_name']), axis=1)
    country_votes_dist_df = country_votes_dist_df.dropna(subset=['distance'])
    #Filter Australia
    country_votes_dist_df = country_votes_dist_df[country_votes_dist_df['from_country_name']!='Australia']
    country_votes_dist_df = country_votes_dist_df[country_votes_dist_df['to_country_name']!='Australia']
    #Scatter plot
    distance_plot = alt.Chart(country_votes_dist_df).mark_circle(size=60).encode(
            x=alt.X('distance:Q').title('Distance (degrees)'),
            y=alt.Y('pct_points_from:Q').title('Percentage of points given'),
            tooltip=['from_country_name', 'to_country_name', 'pct_points_from', 'distance']
        ).properties(
                width=800,
                height=400
        )

    reg_line = distance_plot.transform_regression('distance', 'pct_points_from').mark_line(color='red')

    st.altair_chart(distance_plot + reg_line,  theme=None)

#WHAT MAKES A SONG SO GOOD-----------------------
if selected == 'Music':
    st.title('What makes a song SO good?')

    group_selection = st.selectbox("Select classification group", 
                                   options=['All', 'Winner', 'Top5', 'Top10', 
                                            'Finalist', 'Semifinalist'])

    col1, col2 = st.columns(2)

    #Features radar plot ----------------------
    col1.header('Mean song features values')

    if group_selection == 'Winner':
        features_df = songs_df[songs_df['classification']=='Winner']
    elif group_selection == 'Top5':
        features_df = songs_df[songs_df['classification'].isin(['Winner', 'Top5'])]
    elif group_selection == 'Top10':
        features_df = songs_df[songs_df['classification'].isin(['Winner', 'Top5', 'Top10'])]
    elif group_selection == 'Finalist':
        features_df = songs_df[songs_df['classification'].isin(['Winner', 'Top5', 'Top10', 'Finalist'])]
    elif group_selection == 'Semifinalist':
        features_df = songs_df[songs_df['classification'].isin(['Semifinalist'])]
    else:
        features_df = songs_df
    #Calculate the medium for each characteristic
    features_df = features_df[['energy', 'danceability', 'happiness', 'acousticness', 
                            'liveness', 'speechiness']].melt(
                                var_name='Feature', value_name='Value')
    features_df = features_df.groupby('Feature',as_index=False)['Value'].mean()

    #Plot mean values
    fig = px.line_polar(features_df, r='Value', theta='Feature', line_close=True,
                    template = 'plotly_dark')
    fig.update_traces(fill='toself')
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 80]
        )),
    showlegend=False,
    width=600,
    height=400
    )

    col1.plotly_chart(fig, use_container_width=True, theme=None)