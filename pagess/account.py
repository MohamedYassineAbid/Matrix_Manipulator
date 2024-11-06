import streamlit as st
from widgets.widgets import __login__



def show_account():
    

    __login__obj = __login__(auth_token = "courier_auth_token",
                    width = 200, height = 250,
                    logout_button_name = 'Logout', hide_menu_bool = False,
                    hide_footer_bool = False,
                    )

    LOGGED_IN= __login__obj.build_login_ui()
    
    username= __login__obj.get_username()
    

    if LOGGED_IN == True:
        side_bar,selected_option=__login__obj.account_bar()
        if(selected_option=="Log out"):
            __login__obj.logout_widget()
        

    

    
        
   
  


