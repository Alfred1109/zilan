import bcrypt
import yaml
import importlib
import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
from  streamlit_timeline import timeline
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities import (CredentialsError,ForgotError,Hasher,LoginError,RegisterError, ResetError,UpdateError)

# 加载配置文件
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader) 

# 创建 authenticator 对象
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)
login_page = st.Page(authenticator.login, title="登录")
pg=st.navigation([login_page],position="sidebar")

#增加行，用于存放登陆页面
con1=st.container()
with con1:
    # Creating a login widget
    try:
        authenticator.login()
    except LoginError as e:
        st.error(e)
    if st.session_state["authentication_status"] is True:
        with st.sidebar:
            col1,col2,col3=st.columns(3, vertical_alignment='center')
            with col1:
                st.image('logo.png')
            with col2:
                st.write(f'欢迎 *{st.session_state["name"]}* ')
            with col3:
                authenticator.logout()
       
        st.sidebar.write('___')
        # 定义子页面的名称和对应的脚本文件路径
        subpages = {
            "欢迎页面": "pages/欢迎页面.py",
            "注册页面": "pages/注册页面.py",
            "主成分分析": "pages/主成分分析Principal Component Analysis.py",
            "决策树算法": "pages/决策树算法Decision Tree Algorithm.py",
            "管理页面": "pages/管理页面.py",
            "高斯朴素贝叶斯分类器": "pages/高斯朴素贝叶分类器Highly Gaussian Naive Bayes Classifier.py",

        }

        # 在侧边栏让用户选择页面
        selected_page_name = st.sidebar.selectbox("选择子页面：", list(subpages.keys()))

        # 根据选择的页面名称动态导入对应的模块
        selected_page_path = subpages[selected_page_name]
        module_name = selected_page_path.replace("/", ".").replace(".py", "")
        module = importlib.import_module(module_name)

        # 运行子页面的 main 函数
        if selected_page_name=="注册页面":
            module.main(authenticator,config)
        else:
            module.main()
        st.sidebar.title('Some content')    
        st.sidebar.write('___')

    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')




