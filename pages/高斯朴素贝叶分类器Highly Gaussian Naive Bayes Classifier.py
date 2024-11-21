def main():
    import streamlit as st
    import pandas as pd
    import pydeck as pdk
    from urllib.error import URLError
    st.markdown("# 高斯朴素贝叶斯分类")
    st.sidebar.header("高斯朴素贝叶斯分类")
    st.write(
        """有监督分类是量化投资中常见的情景之一。比如，我们希望根据上市公司财报中的各种指标特征，区分出优秀的和差劲的股票，这就是一个分类问题。在机器学习中，有监督分类的算法有很多，比如 SVM、ANN 以及基于决策树的 AdaBoost 和随机森林等。这其中自然也少不了今天的主角朴素贝叶斯分类器（Naïve Bayes classifiers）。它代表着一类应用贝叶斯定理的分类器的总称。朴素（naive）在这里有着特殊的含义、代表着一个非常强的假设（下文会解释）。

    朴素贝叶斯分类器虽然简单，但是用处非常广泛（尤其是在文本分类方面）。在 IEEE 协会于 2006 年列出的十大数据挖掘算法中，朴素贝叶斯分类器赫然在列（Wu et al. 2008）。捎带一提，另外九个算法是 C4.5、k-Means、SVM、Apriori、EM、PageRank、AdaBoost、kNN 和 CART（那时候深度学习还没有什么发展）."""
    )


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
    st.text(type(dataset))
    st.text(dataset.feature_names)
    st.text(dataset.target_names)

    # 加载bunch数据集
    X = dataset.data
    y = dataset.target

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA降维
    pca = PCA(n_components=10)  # 保留10个主成分
    X_pca = pca.fit_transform(X_scaled)
    st.text("PCA 模型保留的主成份个数")
    st.text(pca.n_components_)  # 返回 PCA 模型保留的主成份个数
    st.dataframe(pca.explained_variance_ratio_)  # 返回 PCA 模型各主成份占比
    st.dataframe(pca.singular_values_) # 返回 PCA 模型各主成份的奇异值


    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

    # 训练贝叶斯网络（使用高斯朴素贝叶斯分类器）
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # 预测测试集
    y_pred = gnb.predict(X_test)
    st.text(y_pred)
    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    st.text(f'模型准确率: {accuracy:.2f}')
