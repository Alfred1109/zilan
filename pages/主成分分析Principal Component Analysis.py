def main():
    import streamlit as st
    import time
    import numpy as np
    # 导入需要使用到的所有库
    import numpy as np
    import pandas as pd
    from fontTools.merge.util import avg_int
    from pandasrw import load ,dump
    from sklearn.decomposition import PCA  # 导入 sklearn.decomposition.PCA 类
    from sklearn.utils import Bunch
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import load_iris
    from sklearn.metrics import f1_score
    from sklearn.decomposition import FactorAnalysis as FA
    from pgmpy.models import BayesianModel
    from pgmpy.estimators import MaximumLikelihoodEstimator, AICScore
    from pgmpy.estimators import HillClimbSearch
    from pgmpy.estimators import K2Score, BicScore,BDeuScore,BDsScore,AICScore
    from pgmpy.inference import VariableElimination
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    import networkx as nx
    import japanize_matplotlib
    import matplotlib.pyplot as pyplot
    import pandas.plotting as plt


    st.markdown("# 主成分分析算法Principal Component Analysis")
    st.sidebar.header("主成分分析算法Principal Component Analysis")
    st.write(
        """因子权重是指每个因子对变量的影响程度，它表示每个因子在变量中的重要性，可以帮助我们了解因子对结果的贡献，通常因子权重越大，表示该因子对变量的影响越大，对结果的贡献也就越大。
    #因子载荷是指每个变量对因子的影响程度，它表示每个变量在因子中的重要性，可以帮助我们了解每个变量对结果的贡献，通常因子载荷越大，表示该变量对因子的影响越大，对结果的贡献也就越大。"""
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    progress_bar.progress(20)
    status_text.text("20% Complete" )
    # for i in range(1, 101):
    #     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    #     status_text.text("%i%% Complete" % i)
    #     chart.add_rows(new_rows)
    #     progress_bar.progress(i)
    #     last_rows = new_rows
    #     time.sleep(0.05)



    progress_bar.progress(40)
    status_text.text("40% Complete" )

    origin_data=pd.read_csv('./thirdteam_checkname.csv')
    origin_data=origin_data#[origin_data.二抗用药==1 ]
    origin_data.fillna(0,inplace=True)#将表中的空值全部替换为0
    origin_data.replace('NaN',0,regex=True,inplace=True)#如果存在文本NAN也将其替换为0
    df_encoded = origin_data[['缺血事件（排除死亡）','出血事件','性别(女=1)','年龄','脑梗死','糖尿病','高血压','高血脂','冠心病','动脉瘤位置','动脉瘤大小','药物依从性评分','二抗用药'
    ]]#此处选取17列因素作为样例。

    st.dataframe(df_encoded)

    progress_bar.progress(60)
    status_text.text("60% Complete" )
    st.dataframe(df_encoded.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))))#使用pandas的describle函数，描述整个数据




    #因子权重是指每个因子对变量的影响程度，它表示每个因子在变量中的重要性，可以帮助我们了解因子对结果的贡献，通常因子权重越大，表示该因子对变量的影响越大，对结果的贡献也就越大。
    #因子载荷是指每个变量对因子的影响程度，它表示每个变量在因子中的重要性，可以帮助我们了解每个变量对结果的贡献，通常因子载荷越大，表示该变量对因子的影响越大，对结果的贡献也就越大。
    # 将数据标准化
    sc = StandardScaler()
    sc.fit(df_encoded)
    z = sc.transform(df_encoded)
    st.text(z.shape)
    # 指定因子数量
    n_components=5

    # 因子分析
    fa = FA(n_components, max_iter=5000) # 
    fitted = fa.fit_transform(z) 
    st.text(fitted.shape)
    # 获得因子载荷矩阵，因子载荷矩阵主要确定每个因素在主成分中的权重
    Factor_loading_matrix = fa.components_.T

    progress_bar.progress(80)
    status_text.text("80% Complete" )

    # 因子关系
    main_fact=pd.DataFrame(Factor_loading_matrix, 
                columns=["第1因子", "第2因子", "第3因子",'第4因子','第5因子'], 
                index=[df_encoded.columns])

    st.text(main_fact)
    progress_bar.progress(100)
    status_text.text("100% Complete" )
    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")

    progress_bar.empty()