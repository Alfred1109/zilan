def main():    
    import streamlit as st
    import pandas as pd
    import altair as alt
    from urllib.error import URLError
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



    st.set_page_config(page_title="贝叶斯网络权重计算", page_icon="📊")

    st.markdown("# 贝叶斯网络权重计算")
    st.sidebar.header("贝叶斯网络权重计算")
    st.write(
        """This demo shows how to use `st.write` to visualize Pandas DataFrames.
    (Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)"""
    )


    origin_data=pd.read_csv('./thirdteam_checkname.csv')
    origin_data=origin_data#[origin_data.二抗用药==1 ]
    origin_data.fillna(0,inplace=True)#将表中的空值全部替换为0
    origin_data.replace('NaN',0,regex=True,inplace=True)#如果存在文本NAN也将其替换为0
    df_encoded = origin_data[['缺血事件（排除死亡）','出血事件','性别(女=1)','年龄','脑梗死','糖尿病','高血压','高血脂','冠心病','动脉瘤位置','动脉瘤大小','药物依从性评分','二抗用药'
    ]]#此处选取17列因素作为样例。

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

    st.text(X[1])
    st.text(y[1])
    # 将数据转换为pandas DataFrame
    df = pd.DataFrame(X, columns=dataset.feature_names)
    df['缺血事件（排除死亡）'] = y

    # 离散化特征
    # 这里我们使用pd.qcut来离散化特征，将每个特征分为几个区间
    df_discrete = pd.get_dummies(df[['缺血事件（排除死亡）','性别(女=1)','年龄','脑梗死','糖尿病','高血压','高血脂','冠心病','动脉瘤位置','动脉瘤大小','药物依从性评分','二抗用药'
    ]])

    #离散概率表
    def df_p(model,col):
        cpd = model.get_cpds(col)
        df = pd.DataFrame({
            'parents': cpd.variable,
            'states': cpd.state_names[col],
            'values': cpd.get_values().ravel()
        })
        return df
    #网络绘图，jupyter适用
    def showBN(model,save=False):
        '''传入BayesianModel对象，调用graphviz绘制结构图，jupyter中可直接显示'''
        from graphviz import Digraph
        node_attr = dict(
        style='filled',
        shape='box',
        align='left',
        fontsize='12',
        ranksep='0.1',
        height='0.2',
        fontname='SimHei'
        )
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        seen = set()
        edges=model.edges()
        for a,b in edges:
            dot.edge(a,b)
        if save:
            dot.view(cleanup=True)
        #输出到路径
        #dot.render('showbn.png', view=True)
        return dot

    # 加载dataframe格式数据集
    hill_data=df_discrete#pd.read_csv('./train.csv')
    from pgmpy.estimators import HillClimbSearch
    from pgmpy.estimators import  K2Score, BicScore
    print(hill_data.head(5))

    # # 创建一个空的 DAG 实例
    start_dag = BayesianNetwork()

    # 向 start_dag 中添加与数据集相同的变量
    start_dag.add_nodes_from(['缺血事件（排除死亡）','高血压','二抗用药'])
    start_dag.add_edges_from([('高血压','缺血事件（排除死亡）'), ('二抗用药','缺血事件（排除死亡）')])
    start_cpds = TabularCPD('缺血事件（排除死亡）', 2, [[1,1,1,1],[1,1,1,1],],
                            evidence=['高血压', '二抗用药'], 
                            evidence_card=[2,2])
    start_dag.add_cpds(start_cpds)
    showBN(start_dag,True)
    # 初始化爬山搜索
    hc = HillClimbSearch(hill_data,state_names=start_dag)
    # 执行爬山搜索，确定网络结构
    best_model = hc.estimate(scoring_method=BDeuScore(hill_data))
    st.text(sorted(best_model.nodes()))
    st.text(best_model.edges())
    # 创建贝叶斯网络对象
    bn = BayesianNetwork(best_model.edges())
    showBN(best_model,True)


    # 检查模型是否有效
    try:
        bn.check_model()
        print("Model is valid.")
    except ValueError as e:
        print("Model check failed:", e)


    # 估计参数
    bn.fit(hill_data)
    bn.get_cpds()
    # # 初始化估计器对象
    estimator = MaximumLikelihoodEstimator(bn,hill_data)

    # 获取每个节点的条件概率分布
    cpds = estimator.get_parameters()
    # 打印每个节点的条件概率分布
    for cpd in cpds:
        print(f"CPD for {cpd.variable}:")
        st.text(cpd)
