import streamlit as st
from widgets.widgets import __login__
import pandas as pd
import json
import io




def retrieve_data(File:str)->bytes:
    with open(f"assets/token/matrices/{File}",'r') as file:
        res=pd.read_csv(file)
        buffer = io.StringIO()
        res.to_csv(buffer, index=False)
        buffer.seek(0) 
    return buffer.getvalue().encode()
        



def show_account()->None:

    __login__obj = __login__(auth_token="dk_prod_MK46VYMVP6M3VSM22AFF6JAX7R4F")

    LOGGED_IN = __login__obj.build_login_ui()

    if LOGGED_IN:
        
        side_bar, selected_option,user = __login__obj.account_bar()
        if selected_option=="Profile":
            __login__obj.show_history()
                
        elif selected_option=="Settings":
            __login__obj.reset_password_from_settings()

        elif selected_option == "Log out":
            __login__obj.logout_widget()
