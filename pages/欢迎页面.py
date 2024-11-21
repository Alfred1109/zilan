def main():
    import streamlit as st
    from  streamlit_timeline import timeline
    import streamlit.components.v1 as components
    import bcrypt
    # st.set_page_config(
    #     page_title="科研算法策略平台",
    #     page_icon="🏥",
    # )

    st.write("# 👏欢迎使用科研算法策略平台! 👏")
    st.sidebar.success("👆请从上面选择想要使用的算法👆")


    st.markdown(
        """
    科研算法策略平台是基于医疗数据队列基础上，对算法选型择优的平台，用来比较各算法对于当前研究的影响，从而在科研时能够更好的选择对应的算法，得到想要的结果。
        ### 想要了解更多吗，想要更新迭代吗?
        - 请联系 [技术组](1358XXXX)
        - 查看产品手册 [documentation](https://github.com/Alfred1109/hosweb)
    """
    )
    col1, col2, col3,col4 = st.columns(4)
    col1.metric("样本数", "300 万", "1.2 万")
    col2.metric("医院数", "50 家", "3 家")
    col3.metric("论文量", "10 篇", "2 篇")
    col4.metric("患者数", "1000 个", "-50 个")

    # load data
    with open('example.json', "r") as f:
        data = f.read()
    # render timeline
    timeline(data, height=800)
    with st.sidebar:
        st.expander("领域知识",True) 
        components.iframe(src="http://localhost/chatbot/ATo4sg5g5S09v9gc",height=700)

    uploaded_files = st.file_uploader(
        "Choose a CSV file", accept_multiple_files=True
    )
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        st.write(bytes_data)