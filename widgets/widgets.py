import streamlit as st
import json
import os
from streamlit_option_menu import option_menu
from streamlit_cookies_manager import EncryptedCookieManager
from utils.utils import check_usr_pass
from utils.utils import check_valid_name
from utils.utils import check_valid_email
from utils.utils import check_unique_email
from utils.utils import check_unique_usr
from utils.utils import register_new_usr
from utils.utils import check_email_exists
from utils.utils import generate_random_passwd
from utils.utils import send_passwd_in_email
from utils.utils import change_passwd
from utils.utils import check_current_passwd


class __login__:
    """
    Builds the UI for the Login/ Sign Up page.
    """

    def __init__(self, auth_token: str):
        """
        Arguments:
        -----------
        1. self
        2. auth_token : The unique authorization token received from - https://www.courier.com/
        """
        self.auth_token = auth_token
        # with open(self.lottie_url,'r') as file:
        #     self.lottie_json = json.load(file)
        self.cookies = EncryptedCookieManager(
            prefix="streamlit_login_ui_yummy_cookies",
            password="9d68d6f2-4258-45c9-96eb-2d6bc74ddbb5-d8f49cab-edbb-404a-94d0-b25b1d4a564b",
        )

        if not self.cookies.ready():
            st.stop()

    def check_auth_json_file_exists(self, auth_filename: str) -> bool:
        """
        Checks if the auth file (where the user info is stored) already exists.
        """
        file_names = []
        for path in os.listdir("assets/token"):
            if os.path.isfile(os.path.join("./assets/token", path)):
                file_names.append(path)

        present_files = []
        for file_name in file_names:
            if auth_filename in file_name:
                present_files.append(file_name)

            present_files = sorted(present_files)
            if len(present_files) > 0:
                return True
        return False

    def get_username(self):
        if st.session_state["LOGOUT_BUTTON_HIT"] == False:
            fetched_cookies = self.cookies
            if "__streamlit_login_signup_ui_username__" in fetched_cookies.keys():
                username = fetched_cookies["__streamlit_login_signup_ui_username__"]
                return username

    def login_widget(self) -> None:
        """
        Creates the login widget, checks and sets cookies, authenticates the users.
        """

        # Checks if cookie exists.
        if st.session_state["LOGGED_IN"] == False:
            if st.session_state["LOGOUT_BUTTON_HIT"] == False:
                fetched_cookies = self.cookies
                if "__streamlit_login_signup_ui_username__" in fetched_cookies.keys():
                    if (
                        fetched_cookies["__streamlit_login_signup_ui_username__"]
                        != "1c9a923f-fb21-4a91-b3f3-5f18e3f01182"
                    ):
                        st.session_state["LOGGED_IN"] = True

        if st.session_state["LOGGED_IN"] == False:
            st.session_state["LOGOUT_BUTTON_HIT"] = False

            del_login = st.empty()
            with del_login.form("Login Form"):
                username = st.text_input("Username", placeholder="Your unique username")
                password = st.text_input(
                    "Password", placeholder="Your password", type="password"
                )
                login_submit_button = st.form_submit_button(label="Login")

                if login_submit_button == True:
                    authenticate_user_check = check_usr_pass(username, password)

                    if authenticate_user_check == False:
                        st.error("Invalid Username or Password!")


                    else:
                        st.session_state["LOGGED_IN"] = True
                        self.cookies["__streamlit_login_signup_ui_username__"] = (
                            username
                        )
                        self.cookies.save()
                        del_login.empty()

    def animation(self) -> None:
        """
        Renders the lottie animation.
        """
        # load_lottieurl(self.lottie_url)
        # st_lottie(self.lottie_json, width = self.width, height = self.height)
        pass

    def sign_up_widget(self) -> None:
        """
        Creates the sign-up widget and stores the user info in a secure way in the _secret_auth_.json file.
        """
        with st.form("Sign Up Form"):
            name_sign_up = st.text_input("Name *", placeholder="Please enter your name")
            valid_name_check = check_valid_name(name_sign_up)

            email_sign_up = st.text_input(
                "Email *", placeholder="Please enter your email"
            )
            valid_email_check = check_valid_email(email_sign_up)
            unique_email_check = check_unique_email(email_sign_up)

            username_sign_up = st.text_input(
                "Username *", placeholder="Enter a unique username"
            )
            unique_username_check = check_unique_usr(username_sign_up)

            password_sign_up = st.text_input(
                "Password *", placeholder="Create a strong password", type="password"
            )

            sign_up_submit_button = st.form_submit_button(label="Register")

            if sign_up_submit_button:
                if valid_name_check == False:
                    st.error("Please enter a valid name!")

                elif valid_email_check == False:
                    st.error("Please enter a valid Email!")

                elif unique_email_check == False:
                    st.error("Email already exists!")

                elif unique_username_check == False:
                    st.error(f"Sorry, username {username_sign_up} already exists!")

                elif unique_username_check == None:
                    st.error("Please enter a non - empty Username!")

                if valid_name_check == True:
                    if valid_email_check == True:
                        if unique_email_check == True:
                            if unique_username_check == True:
                                register_new_usr(
                                    name_sign_up,
                                    email_sign_up,
                                    username_sign_up,
                                    password_sign_up,
                                )
                                st.success("Registration Successful!")

    def forgot_password(self) -> None:
        """
        Creates the forgot password widget and after user authentication (email), triggers an email to the user
        containing a random password.
        """
        with st.form("Forgot Password Form"):
            email_forgot_passwd = st.text_input(
                "Email", placeholder="Please enter your email"
            )
            email_exists_check, username_forgot_passwd = check_email_exists(
                email_forgot_passwd
            )

            forgot_passwd_submit_button = st.form_submit_button(label="Get Password")

            if forgot_passwd_submit_button:
                if email_exists_check == False:
                    st.error("Email ID not registered with us!")

                if email_exists_check == True:
                    random_password = generate_random_passwd()
                    send_passwd_in_email(
                        self.auth_token,
                        username_forgot_passwd,
                        email_forgot_passwd,
                        random_password,
                    )
                    change_passwd(email_forgot_passwd, random_password)
                    st.success("Secure Password Sent Successfully!")

    def reset_password(self) -> None:
        """
        Creates the reset password widget and after user authentication (email and the password shared over that email),
        resets the password and updates the same in the _secret_auth_.json file.
        """
        with st.form("Reset Password Form"):
            email_reset_passwd = st.text_input(
                "Email", placeholder="Please enter your email"
            )
            email_exists_check, username_reset_passwd = check_email_exists(
                email_reset_passwd
            )

            current_passwd = st.text_input(
                "Temporary Password",
                placeholder="Please enter the password you received in the email",
            )
            current_passwd_check = check_current_passwd(
                email_reset_passwd, current_passwd
            )

            new_passwd = st.text_input(
                "New Password",
                placeholder="Please enter a new, strong password",
                type="password",
            )

            new_passwd_1 = st.text_input(
                "Re - Enter New Password",
                placeholder="Please re- enter the new password",
                type="password",
            )

            reset_passwd_submit_button = st.form_submit_button(label="Reset Password")

            if reset_passwd_submit_button:
                if email_exists_check == False:
                    st.error("Email does not exist!")

                elif current_passwd_check == False:
                    st.error("Incorrect temporary password!")

                elif new_passwd != new_passwd_1:
                    st.error("Passwords don't match!")

                if email_exists_check == True:
                    if current_passwd_check == True:
                        change_passwd(email_reset_passwd, new_passwd)
                        st.success("Password Reset Successfully!")


    def reset_password_from_settings(self) -> None:
        """
        Creates the reset password widget and after user authentication (email and the password shared over that email),
        resets the password and updates the same in the _secret_auth_.json file.
        """
        email=str()
        with open("assets/token/_secret_auth_.json", "r") as auth_json:
            authorized_users_data = json.load(auth_json)

            for user in authorized_users_data:
                if user["username"] == self.get_username():
                  email=user["email"]
                  break

        with st.form("Reset Password Form"):
            
           

            current_passwd = st.text_input(
                "Current Password",
                placeholder="Please enter the password you received in the email",
            )
            current_passwd_check = check_current_passwd(
                email, current_passwd
            )

            new_passwd = st.text_input(
                "New Password",
                placeholder="Please enter a new, strong password",
                type="password",
            )

            new_passwd_1 = st.text_input(
                "Re - Enter New Password",
                placeholder="Please re- enter the new password",
                type="password",
            )

            reset_passwd_submit_button = st.form_submit_button(label="Reset Password")

            if reset_passwd_submit_button:
               
                if current_passwd_check == False:
                    st.error("Incorrect password!")

                elif new_passwd != new_passwd_1:
                    st.error("Passwords don't match!")

                
                elif current_passwd_check == True:
                        change_passwd(email, new_passwd)
                        st.success("Password Reset Successfully!")


    def logout_widget(self) -> None:
        """
        Creates the logout widget in the sidebar only if the user is logged in.
        """

        st.session_state["LOGOUT_BUTTON_HIT"] = True
        st.session_state["LOGGED_IN"] = False
        self.cookies["__streamlit_login_signup_ui_username__"] = (
            "1c9a923f-fb21-4a91-b3f3-5f18e3f01182"
        )

    def nav_sidebar(self):
        """
        Creates the side navigaton bar
        """

        main_page_sidebar = st.sidebar.empty()

        with main_page_sidebar:
            selected_option = option_menu(
                menu_title="Navigation",
                menu_icon="list-columns-reverse",
                icons=[
                    "box-arrow-in-right",
                    "person-plus",
                    "x-circle",
                    "arrow-counterclockwise",
                ],
                options=[
                    "Login",
                    "Create Account",
                    "Forgot Password?",
                    "Reset Password",
                ],
                styles={
                    "container": {"padding": "5px"},
                    "nav-link": {
                        "font-size": "14px",
                        "text-align": "left",
                        "margin-top": "0px",
                    },
                },
            )
        return main_page_sidebar, selected_option

    def account_bar(self):
        main_page_sidebar = st.sidebar.empty()
        with main_page_sidebar:
            selected_option = option_menu(
                menu_title=f"{self.get_username().capitalize()} Account",
                menu_icon="house",
                icons=["cast", "gear", "x-circle"],
                options=["Profile", "Settings", "Log out"],
                styles={
                    "container": {"padding": "5px"},
                    "nav-link": {
                        "font-size": "14px",
                        "text-align": "left",
                        "margin-top": "0px",
                    },
                },
            )
            return main_page_sidebar, selected_option,self.get_username().capitalize()
        
    

    def build_login_ui(self):
        """
        Brings everything together, calls important functions.
        """
        if "LOGGED_IN" not in st.session_state:
            st.session_state["LOGGED_IN"] = False

        if "LOGOUT_BUTTON_HIT" not in st.session_state:
            st.session_state["LOGOUT_BUTTON_HIT"] = False

        auth_json_exists_bool = self.check_auth_json_file_exists("_secret_auth_.json")

        if auth_json_exists_bool == False:
            with open("assets/token/_secret_auth_.json", "w") as auth_json:
                json.dump([], auth_json)

        # self.hide_footer()
        main_page_sidebar, selected_option = self.nav_sidebar()

        if selected_option == "Login":

            self.login_widget()

        if selected_option == "Create Account":
            self.sign_up_widget()

        if selected_option == "Forgot Password?":
            self.forgot_password()

        if selected_option == "Reset Password":
            self.reset_password()

        if st.session_state["LOGGED_IN"] == True:
            main_page_sidebar.empty()

        return st.session_state["LOGGED_IN"]
