from widgets.widgets import __login__
import creds








def show_account()->None:

    __login__obj = __login__(auth_token=creds.api_key1)

    LOGGED_IN = __login__obj.build_login_ui()

    if LOGGED_IN:
        
        side_bar, selected_option,user = __login__obj.account_bar()
        if selected_option=="Profile":
            __login__obj.show_history()
                
        elif selected_option=="Settings":
            __login__obj.reset_password_from_settings()

        elif selected_option == "Log out":
            __login__obj.logout_widget()
