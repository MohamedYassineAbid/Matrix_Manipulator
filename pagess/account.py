import streamlit as st
from widgets.widgets import __login__
import pandas as pd
import json


def extract(data:str)-> list:
    temp=data[22:-4].split('*')
    res=[]
    for i in range(3):
        res.append(temp[i])
    
    test=':'.join(data[3:])
    res.append(test)
    res.append(data)
        
    
    return res

def show_account():

    __login__obj = __login__(auth_token="your api key")

    LOGGED_IN = __login__obj.build_login_ui()

    if LOGGED_IN:
        side_bar, selected_option,user = __login__obj.account_bar()
        if selected_option=="Profile":
            st.title(f"{user} History")
            header=["Type","Height","Width","Time","Link"]
            with open(f'assets/token/matrices/{user.lower()}.json','r') as file:
                data=json.load(file)
                res=[]
                for record in data:
                    res.append(record.values())
                
                res=pd.DataFrame(res,columns=header)
                st.table(res)
                
           

                
                
            

        
        elif selected_option=="Settings":
            __login__obj.reset_password_from_settings()

        elif selected_option == "Log out":
            __login__obj.logout_widget()
