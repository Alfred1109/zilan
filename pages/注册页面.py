def main(authenticator,config):    
    import bcrypt
    import streamlit as st
    import streamlit.components.v1 as components
    import streamlit_authenticator as stauth
    from  streamlit_timeline import timeline
    #在初始运行阶段增加用户登陆管理
    import yaml
    import streamlit as st
    from yaml.loader import SafeLoader
    import streamlit_authenticator as stauth
    from streamlit_authenticator.utilities import (CredentialsError,ForgotError,Hasher,LoginError,RegisterError, ResetError,UpdateError)


    #增加两列，用于存放登陆和注册页面
    col1=st.container()
    with col1:
        # Creating a new user registration widget
        if st.session_state["authentication_status"] is not True:
            try:
                (email_of_registered_user,
                username_of_registered_user,
                name_of_registered_user) = authenticator.register_user()
                if email_of_registered_user:
                    st.success('User registered successfully')
            except RegisterError as e:
                st.error(e)


    col3=st.container()
    with col3:
        # Creating a forgot password widget
        if st.session_state["authentication_status"] is not True:
            try:
                (username_of_forgotten_password,
                email_of_forgotten_password,
                new_random_password) = authenticator.forgot_password()
                if username_of_forgotten_password:
                    st.success(f"New password **'{new_random_password}'** to be sent to user securely")
                    config['credentials']['usernames'][username_of_forgotten_password]['pp'] = new_random_password
                    # Random password to be transferred to the user securely
                elif not username_of_forgotten_password:
                    st.error('Username not found')
            except ForgotError as e:
                st.error(e)
    col4=st.container()
    with col4:
        # Creating a forgot username widget
        if st.session_state["authentication_status"] is not True:
            try:
                (username_of_forgotten_username,
                email_of_forgotten_username) = authenticator.forgot_username()
                if username_of_forgotten_username:
                    st.success(f"Username **'{username_of_forgotten_username}'** to be sent to user securely")
                    # Username to be transferred to the user securely
                elif not username_of_forgotten_username:
                    st.error('Email not found')
            except ForgotError as e:
                st.error(e)

    # Creating an update user details widget
    if st.session_state["authentication_status"]:
        if st.session_state["authentication_status"] is not True:
            try:
                if authenticator.update_user_details(st.session_state["username"]):
                    st.success('Entries updated successfully')
            except UpdateError as e:
                st.error(e)

    # Saving config file
    with open('config.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False)


