import pagess
import streamlit as st
from streamlit_navigation_bar import st_navbar
import os

st.set_page_config(page_title="MatManipulator", page_icon="logo.svg", layout="wide")
def navbar():

    pages = ["Home", "User Guide",  "About Us", "GitHub", "Account"]
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(parent_dir, "assets/logo/logo.svg")
    urls = {"GitHub": "https://github.com/MohamedAliJmal/Matrix_Manipulator"}
    styles = {
        "nav": {
            "background-color": "#4169e1",
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
        },
    }

    return st_navbar(
        pages,
        logo_path=logo_path,
        urls=urls,
        styles=styles,
    )


page = navbar()
functions = {
    "Home": pagess.show_home,
    "User Guide": pagess.show_user_guide,
    "About Us": pagess.show_examples,
    "Account": pagess.show_account,
}


go_to = functions.get(page)
if go_to:
    go_to()
