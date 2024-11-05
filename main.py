import streamlit as st
from streamlit_navigation_bar import st_navbar
import os
import pagess




    
pages = ["Home", "User Guide", "API", "Examples", "GitHub","Account"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "logo.svg")
urls = {"GitHub": "https://github.com/MohamedAliJmal/Matrix_Manipulator"}
styles = {
     
    "nav": {
        "background-color": "royalblue",
        
        
    },
    "img": {
        "padding-right": "14px",
    },
    "span": {
        "color": "white",
        "padding": "14px",
        
    },
    "active": {
        "background-color": "white",
        "color": "black",
        "font-weight": "normal",
        "padding": "14px",
    }
}



page = st_navbar(
    pages,
    logo_path=logo_path,
    urls=urls,
    styles=styles, 
)

functions = {
    "Home": pagess.show_home,
    "User Guide": pagess.show_user_guide,
    "API": pagess.show_api,
    "Examples": pagess.show_examples,
    "Account": pagess.show_account,
    
}


go_to = functions.get(page)
if go_to:
    go_to()



