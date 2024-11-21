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


    st.set_page_config(page_title="DataFrame Demo", page_icon="📊")

    st.markdown("# DataFrame Demo")
    st.sidebar.header("DataFrame Demo")
    st.write(
        """贝叶斯网络（Bayesian Network），也称为信念网络（Belief Network）或有向无环图模型（Directed Acyclic Graph, DAG），是一种概率图模型，用于表示变量之间的条件依赖关系。它基于贝叶斯定理，可以用于推理和决策"""
    )



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






    #通过人工调优的方式，把学习出来的模型结构进行优化，选取最符合真实世界的网络结构，进行绘制。
    figure_dag = BayesianNetwork()
    # 向 start_dag 中添加与数据集相同的变量
    figure_dag.add_nodes_from(['缺血事件（排除死亡）','高血压','高血脂','冠心病','动脉瘤位置','动脉瘤大小','药物依从性评分','二抗用药'])
    figure_dag.add_edges_from([('高血脂','缺血事件（排除死亡）'), ('二抗用药','缺血事件（排除死亡）'),('动脉瘤位置','高血压'),('动脉瘤大小','高血压'),('高血压','二抗用药'),('高血脂','二抗用药'),('冠心病','高血脂'),('药物依从性评分','高血脂')])
    figure1_cpds = TabularCPD('缺血事件（排除死亡）', 2, [[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],],
                            evidence=['高血脂', '二抗用药'], 
                            evidence_card=[2,2])
    figure2_cpds = TabularCPD('高血脂', 2, [[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],],
                            evidence=['冠心病', '药物依从性评分'], 
                            evidence_card=[2,2])
    figure3_cpds = TabularCPD('二抗用药', 2, [[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],],
                            evidence=['高血压', '高血脂'], 
                            evidence_card=[2,2])
    figure4_cpds = TabularCPD('高血压', 2, [[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],],
                            evidence=['动脉瘤位置', '动脉瘤大小'], 
                            evidence_card=[2,2])
    figure5_cpds = TabularCPD('冠心病', 2, [[0.5],[0.5],],)
    figure6_cpds = TabularCPD('药物依从性评分', 2, [[0.5],[0.5],],)
    figure7_cpds = TabularCPD('动脉瘤位置', 2, [[0.5],[0.5],],)
    figure8_cpds = TabularCPD('动脉瘤大小', 2, [[0.5],[0.5],],)
    figure_dag.add_cpds(figure1_cpds,figure2_cpds,figure3_cpds,figure4_cpds,figure5_cpds,figure6_cpds,figure7_cpds,figure8_cpds)
    showBN(figure_dag,True)
    fbn=figure_dag
    # 检查模型是否有效
    try:
        fbn.check_model()
        print("Model is valid.")
    except ValueError as e:
        print("Model check failed:", e)


    cpds=fbn.get_cpds()
    for cpd in cpds:
        print(f"CPD for {cpd.variable}:")
        print(cpd)

    #将每个节点的概率取均值，根据人工调整后的网络图进行打印，由于nx直接用spring_layout(bn.to_direct())绘制会报错，因此通过获取网络结构和边权重，用人工拼接的方式绘制）
    pyplot.rcParams["font.sans-serif"] = ["SimHei"]
    G = nx.MultiDiGraph(fbn.to_directed())
    # 获取贝叶斯网络的条件概率作为边的权重
    edge_weights = {}
    for edge in G.edges():
        parent = edge[1]
        child = edge[0]
        cpd = fbn.get_cpds(parent)
        if parent in cpd.variable:
            # 使用 CPD 中所有概率值的平均值作为权重
            weights = cpd.values
            weight = np.mean(weights)
            edge_weights[(child, parent)] = weight
            print(child,parent,weight)
    # 获取边的权重
    edge_labels = nx.get_edge_attributes(G, 'weight')
    for (u, v) in G.edges():
        edge_labels[(u, v)] = edge_weights.get((u, v), 1)  # 使用从CPD获取的权重或默认值

    DG = nx.MultiDiGraph()
    for edge in G.edges():
        DG.add_edge(edge[0],edge[1],weight=edge_labels.get((edge[0],edge[1]),1))

    #使用 spring_layout 算法生成节点位置
    pos = nx.spring_layout(DG, weight='weight', iterations=42)

    # 绘制图形
    nx.draw(DG, pos, with_labels=True, node_color='lightblue', node_size=2000)

    # 显示边的权重
    dg_labels = nx.get_edge_attributes(DG, 'weight')
    nx.draw_networkx_edge_labels(DG, pos, edge_labels=dg_labels)
    # 根据权重调整边的宽度
    edge_width = [edge_labels[(u, v)] * 1 for (u, v) in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_width, edgelist=G.edges(), edge_color='gray')

    # 显示图形
    pyplot.show()


    from pgmpy.readwrite import XMLBIFWriter
    from pgmpy.utils import get_example_model
    writer = XMLBIFWriter(fbn)
    writer.write_xmlbif('cpds.xml')
    # 下面的代码使用了VariableElimination方法，亦可用BeliefPropagation，其有相同的接口。
    # 与fit函数类似，也提供了输入dataframe的简便推理方法predict，如下
    # 只要剔除想预测的列输入predict函数，就可以得到预测结果的dataframe


    #以下推理主要解决的是，当已知其中一个变量的情况下，另外变量的概率是多少？
    model_infer = VariableElimination(fbn)
    q = model_infer.query(variables=['缺血事件（排除死亡）'], evidence={'动脉瘤大小': 1, '动脉瘤位置': 1})
    print(q)
    #以下推理主要解决的是，当已知其中变量的情况下，另外几个变量的可能性
    q = model_infer.map_query(variables=['动脉瘤位置','动脉瘤大小'], evidence={'缺血事件（排除死亡）': 1,'高血脂':0})
    print(q)

    #增加for循环，获取不同值下的变量概率
    vari_name='二抗用药'
    fbnstats=fbn.states
    erkang_value=fbnstats[vari_name] #由于这里在推理时，只能取训练好的贝叶斯网络中的键值，所以需要从网络中获取节点所有可能值用于推理，而不是直接从原始数据中抓取全集
    result=pd.DataFrame(columns=['变量','变量值','目标','目标值','概率'])
    i=0
    for j in (0,1):
        for value in erkang_value:
            model_infer = VariableElimination(fbn)
            q = model_infer.query(variables=['缺血事件（排除死亡）'], evidence={ '动脉瘤大小': 1, '动脉瘤位置': 1,vari_name:value})
            nake=q.variables[0]
            templist=[[vari_name,value,nake,q.state_names.get(nake)[j],q.values[j]]]
            if templist is None:
                continue
            tempdf=pd.DataFrame(templist,columns=['变量','变量值','目标','目标值','概率'])
            result =pd.concat([result,tempdf], verify_integrity=False) 
            i=i+1
    st.text(result)

    #预测需要遵循测试集与训练集字段一致原则，否则会报错。
    union_data=df_discrete[['缺血事件（排除死亡）','高血压','高血脂','冠心病','动脉瘤位置','动脉瘤大小','药物依从性评分','二抗用药']]
    train_data=union_data.iloc[1:2000]
    test_data=union_data.iloc[1501:3500]
    print(test_data)
    fbn.fit(train_data)
    print(fbn.check_model())
    test_data=test_data.drop('缺血事件（排除死亡）', axis=1).reset_index(drop=True)
    print(len(test_data))
    y_pred = fbn.predict(test_data) #这里有可能报索引错误，原因是原始数据的药物依从性评分为小数，改为整数后可以正常运行。
    print((y_pred['缺血事件（排除死亡）'].values==train_data['缺血事件（排除死亡）'].values).sum()/len(test_data))
    #测试集精度0.9363550593600126



