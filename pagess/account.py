import streamlit as st
from widgets.widgets import __login__


def show_account():

    __login__obj = __login__(auth_token="your api key")

    LOGGED_IN = __login__obj.build_login_ui()

    if LOGGED_IN:
        side_bar, selected_option = __login__obj.account_bar()
        if selected_option=="Profile":
            pass
        
        elif selected_option=="Settings":
            pass

        elif selected_option == "Log out":
            __login__obj.logout_widget()
