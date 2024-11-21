
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


    st.markdown("# 决策树算法Decision Tree Algorithm")
    st.sidebar.header("决策树算法Decision Tree Algorithm")
    st.write(
        """决策树是一种非参数监督学习算法，可用于分类和回归任务。它具有分层的树形结构，由根节点、分支、内部节点和叶节点组成。"""
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    progress_bar.empty()


    origin_data=pd.read_csv('./thirdteam_checkname.csv')
    origin_data=origin_data#[origin_data.二抗用药==1 ]
    origin_data.fillna(0,inplace=True)#将表中的空值全部替换为0
    origin_data.replace('NaN',0,regex=True,inplace=True)#如果存在文本NAN也将其替换为0
    df_encoded = origin_data[['缺血事件（排除死亡）','出血事件','性别(女=1)','年龄','脑梗死','糖尿病','高血压','高血脂','冠心病','动脉瘤位置','动脉瘤大小','药物依从性评分','二抗用药'
    ]]#此处选取17列因素作为样例。

    st.dataframe(df_encoded)

    st.dataframe(df_encoded.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))))#使用pandas的describle函数，描述整个数据

    #创建指定数据集格式，用于后续的训练
    matrix = df_encoded.values  #将dataframe转化成numpy格式
    matrix = np.nan_to_num(matrix)#取出numpy中nan值
    X = matrix[:, 2:]   # 取除第一列之外的所有列
    y = matrix[:, 1]    # 取第一列
    # 创建一个Bunch对象用于标准化机器学习数据
    dataset1 = Bunch(data=X, target=y, feature_names=['性别(女=1)','年龄','脑梗死','糖尿病','高血压','高血脂','冠心病','动脉瘤位置','动脉瘤大小','药物依从性评分','二抗用药'], target_names=['缺血事件（排除死亡）'])
    dataset = Bunch(data=X, target=y, feature_names=['性别(女=1)','年龄','脑梗死','糖尿病','高血压','高血脂','冠心病','动脉瘤位置','动脉瘤大小','药物依从性评分','二抗用药'], target_names=['出血事件'])
    print(type(dataset),dataset.feature_names,dataset.target_names)


    # 加载bunch数据集
    X = dataset.data
    y = dataset.target

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA降维
    pca = PCA(n_components=10)  # 保留10个主成分
    X_pca = pca.fit_transform(X_scaled)
    print(pca.n_components_)  # 返回 PCA 模型保留的主成份个数
    print(pca.explained_variance_ratio_)  # 返回 PCA 模型各主成份占比
    print(pca.singular_values_) # 返回 PCA 模型各主成份的奇异值

    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    import matplotlib.pyplot as mplt
    # 加载bunch数据集
    X = dataset.data
    y = dataset.target

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA降维
    pca = PCA(n_components=10)  # 保留10个主成分
    X_pca = pca.fit_transform(X_scaled)
    print(pca.n_components_)  # 返回 PCA 模型保留的主成份个数
    print(pca.explained_variance_ratio_)  # 返回 PCA 模型各主成份占比
    print(pca.singular_values_) # 返回 PCA 模型各主成份的奇异值


    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

    # 训练决策树，使用决策树模型
    clf=DecisionTreeClassifier (random_state=42)
    clf.fit(X_train, y_train)

    # 预测测试集
    y_pred = clf.predict(X_test)
    print(y_pred)
    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f'模型准确率: {accuracy:.2f}')

    #画出决策树
    temptree=tree.plot_tree(clf,filled=True)
    st.text(temptree)    



    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")
