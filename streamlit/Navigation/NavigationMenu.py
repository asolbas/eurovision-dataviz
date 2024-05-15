import streamlit as st
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title = 'Menu',
        options = ['Overview', 'Geopolitics', 'Music', 'Voting'],
        #Boostrap icons codes
        icons = ['graph-up', 'globe-americas', 'music-note-beamed', 'trophy-fill'],
        #menu_icon = uwu,
        default_index=0,

    )

    if selected == 'Overview':
        st.header('Eurovision in a Nutshell')