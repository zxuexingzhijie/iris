#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os
import joblib
import seaborn as sns
import locale

# 设置系统默认编码，避免Windows中的编码问题
locale.setlocale(locale.LC_ALL, 'C')

# 配置中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['font.family'] = ['sans-serif']         # 使用上述字体族

# 确保目录存在
def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_preprocessed_data():
    """
    加载预处理后的鸢尾花数据集
    """
    # 尝试从lab2/data目录加载
    try:
        data_dir = "./lab2/data"
        df = pd.read_csv(os.path.join(data_dir, "iris_processed.csv"))
        print("从lab2/data目录成功加载数据")
        return df
    except FileNotFoundError:
        # 如果找不到，尝试从根目录的iris.data加载
        try:
            print("尝试从原始数据加载...")
            column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
            df = pd.read_csv("./iris.data", header=None, names=column_names)
            return df
        except FileNotFoundError:
            print("无法找到数据文件，请确保数据文件存在。")
            return None

def discretize_data(df):
    """
    将连续型特征离散化，便于关联规则挖掘
    """
    print("=" * 50)
    print("数据离散化处理")
    print("=" * 50)
    
    # 创建新的数据框来存储离散化后的数据
    df_discretized = df.copy()
    
    # 对每个数值特征进行离散化
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    for feature in features:
        # 使用分位数进行离散化，分为3个区间
        df_discretized[feature] = pd.qcut(df[feature], q=3, labels=[f'{feature}_低', f'{feature}_中', f'{feature}_高'])
        
    # 将种类标签保留
    df_discretized['species'] = df['species']
    
    print("离散化后的数据前5行:")
    print(df_discretized.head())
    
    # 统计离散化后的分布
    print("\n离散化后的特征分布:")
    for feature in features:
        print(f"\n{feature} 分布:")
        print(df_discretized[feature].value_counts())
    
    return df_discretized

def prepare_transactions(df_discretized):
    """
    将离散化的数据转换为交易数据格式
    """
    # 为每个样本创建一个交易列表
    transactions = []
    
    for _, row in df_discretized.iterrows():
        # 将每个样本的离散特征值和类别作为一个交易
        transaction = [
            row['sepal_length'], 
            row['sepal_width'], 
            row['petal_length'], 
            row['petal_width'],
            row['species']
        ]
        transactions.append(transaction)
    
    # 使用TransactionEncoder转换为one-hot编码
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
    
    return df_transactions

def mine_frequent_itemsets(df_transactions, min_support=0.1):
    """
    使用Apriori算法挖掘频繁项集
    """
    print("=" * 50)
    print("频繁项集挖掘")
    print("=" * 50)
    
    # 使用Apriori算法挖掘频繁项集
    frequent_itemsets = apriori(df_transactions, min_support=min_support, use_colnames=True)
    
    # 按支持度降序排序
    frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
    
    print(f"使用最小支持度 {min_support} 发现的频繁项集数量: {len(frequent_itemsets)}")
    print("\n前10个频繁项集:")
    pd.set_option('display.max_colwidth', None)
    print(frequent_itemsets.head(10))
    
    return frequent_itemsets

def extract_association_rules(frequent_itemsets, min_confidence=0.7):
    """
    从频繁项集中提取关联规则
    """
    print("=" * 50)
    print("关联规则提取")
    print("=" * 50)
    
    # 从频繁项集中提取关联规则
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # 按置信度降序排序
    rules = rules.sort_values('confidence', ascending=False)
    
    print(f"使用最小置信度 {min_confidence} 发现的关联规则数量: {len(rules)}")
    print("\n前10个关联规则:")
    pd.set_option('display.max_colwidth', None)
    print(rules.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    
    return rules

def analyze_rules_by_class(rules):
    """
    分析与不同鸢尾花品种相关的规则
    """
    print("=" * 50)
    print("按品种分析关联规则")
    print("=" * 50)
    
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    
    for cls in classes:
        # 找出后件包含当前类别的规则
        class_rules = rules[rules['consequents'].apply(lambda x: cls in x)]
        
        print(f"\n预测 {cls} 的规则数量: {len(class_rules)}")
        if not class_rules.empty:
            print("前3个规则:")
            for idx, row in class_rules.head(3).iterrows():
                antecedents = list(row['antecedents'])
                print(f"  如果 {antecedents} 那么 {cls} (置信度: {row['confidence']:.3f}, 支持度: {row['support']:.3f}, 提升度: {row['lift']:.3f})")
    
    return

def visualize_results(frequent_itemsets, rules):
    """
    可视化频繁项集和关联规则
    """
    print("=" * 50)
    print("结果可视化")
    print("=" * 50)
    
    # 确保目录存在
    ensure_dir("./lab1")
    
    # 1. 频繁项集大小分布
    plt.figure(figsize=(10, 6))
    size_counts = frequent_itemsets['itemsets'].apply(lambda x: len(x)).value_counts().sort_index()
    ax = size_counts.plot(kind='bar')
    plt.title('频繁项集大小分布')
    plt.xlabel('项集大小')
    plt.ylabel('数量')
    for i, v in enumerate(size_counts):
        ax.text(i, v + 0.5, str(v), ha='center')
    plt.tight_layout()
    plt.savefig('./lab1/frequent_itemset_size_distribution.png')
    plt.close()
    
    # 2. 关联规则评估指标分布
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # 支持度分布
    sns.histplot(rules['support'], bins=10, ax=axes[0])
    axes[0].set_title('关联规则支持度分布')
    axes[0].set_xlabel('支持度')
    axes[0].set_ylabel('数量')
    
    # 置信度分布
    sns.histplot(rules['confidence'], bins=10, ax=axes[1])
    axes[1].set_title('关联规则置信度分布')
    axes[1].set_xlabel('置信度')
    axes[1].set_ylabel('数量')
    
    # 提升度分布
    sns.histplot(rules['lift'], bins=10, ax=axes[2])
    axes[2].set_title('关联规则提升度分布')
    axes[2].set_xlabel('提升度')
    axes[2].set_ylabel('数量')
    
    plt.tight_layout()
    plt.savefig('./lab1/association_rules_metrics.png')
    plt.close()
    
    # 3. 气泡图：支持度、置信度和提升度
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(rules['support'], rules['confidence'], 
               alpha=0.5, s=rules['lift']*30,
               c=rules['lift'], cmap='viridis')
    plt.colorbar(scatter, label='提升度')
    plt.title('关联规则指标气泡图')
    plt.xlabel('支持度')
    plt.ylabel('置信度')
    plt.tight_layout()
    plt.savefig('./lab1/association_rules_bubble_chart.png')
    plt.close()

def save_results(frequent_itemsets, rules):
    """
    保存频繁项集和关联规则结果
    """
    # 确保目录存在
    ensure_dir("./lab2/data")
    
    # 保存频繁项集
    frequent_itemsets.to_csv("./lab2/data/frequent_itemsets.csv", index=False)
    
    # 保存关联规则
    rules.to_csv("./lab2/data/association_rules.csv", index=False)
    
    print("\n结果已保存到 './lab2/data/' 目录")

def main():
    """
    主函数
    """
    print("=" * 50)
    print("鸢尾花数据集关联规则挖掘")
    print("=" * 50)
    
    # 1. 加载预处理数据
    df = load_preprocessed_data()
    if df is None:
        return
    
    # 2. 数据离散化
    df_discretized = discretize_data(df)
    
    # 3. 准备交易数据
    df_transactions = prepare_transactions(df_discretized)
    
    # 4. 挖掘频繁项集
    frequent_itemsets = mine_frequent_itemsets(df_transactions, min_support=0.1)
    
    # 5. 提取关联规则
    rules = extract_association_rules(frequent_itemsets, min_confidence=0.7)
    
    # 6. 按品种分析关联规则
    analyze_rules_by_class(rules)
    
    # 7. 结果可视化
    visualize_results(frequent_itemsets, rules)
    
    # 8. 保存结果
    save_results(frequent_itemsets, rules)
    
    print("\n鸢尾花数据集关联规则挖掘完成!")

if __name__ == "__main__":
    main() 