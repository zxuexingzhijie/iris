#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import seaborn as sns
import os
import joblib
import locale

# 设置系统默认编码，避免Windows中的编码问题
locale.setlocale(locale.LC_ALL, 'C')

# 配置中文字体支持
import matplotlib
# 设置使用符号体作为默认字体，这个字体支持大多数中文字符
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['font.family'] = ['sans-serif']  # 使用上述字体族

# 确保目录存在
def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# 数据加载与查看
def load_and_examine_data(file_path):
    """
    加载数据集并进行初步检查
    """
    # 读取数据集，由于数据集没有列名，我们需要添加列名
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv(file_path, header=None, names=column_names)
    
    print("=" * 50)
    print("数据集基本信息:")
    print("=" * 50)
    print(f"数据集形状: {df.shape}")
    print("\n数据集前5行:")
    print(df.head())
    
    print("\n数据集基本统计信息:")
    print(df.describe())
    
    print("\n数据集的类别分布:")
    print(df['species'].value_counts())
    
    print("\n检查是否有缺失值:")
    print(df.isnull().sum())
    
    return df

# 数据可视化
def visualize_data(df):
    """
    对数据进行可视化分析
    """
    print("=" * 50)
    print("数据可视化:")
    print("=" * 50)
    
    # 确保lab1图片目录存在
    ensure_dir("./lab1")
    
    # 1. 散点图矩阵(单独生成一张图)
    plt.figure(figsize=(10, 8))
    pair_plot = sns.pairplot(df, hue='species')
    pair_plot.fig.suptitle('鸢尾花数据集的散点图矩阵', y=1.02)
    plt.savefig('./lab1/iris_pairplot.png')
    plt.close()
    
    # 2. 创建一个新图形，包含箱线图和热图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 箱线图
    df_melt = pd.melt(df, id_vars=['species'], 
                     value_vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    sns.boxplot(x='variable', y='value', hue='species', data=df_melt, ax=axes[0])
    axes[0].set_title('不同特征的箱线图')
    axes[0].set_xlabel('特征')
    axes[0].set_ylabel('数值')
    
    # 相关性热图
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1])
    axes[1].set_title('特征之间的相关性热图')
    
    plt.tight_layout()
    plt.savefig('./lab1/iris_visualization.png')
    plt.close()
    
    # 3. 创建一个独立的PCA可视化图(用于初始数据集的可视化)
    # 对原始数据进行PCA降维以可视化类别分布
    X = df.iloc[:, :4].values  # 取出特征
    y = df['species']  # 取出标签
    
    # 标准化
    X_std = StandardScaler().fit_transform(X)
    
    # PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    # 创建DataFrame便于绘图
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['species'] = df['species'].values
    
    # 绘制PCA散点图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df, palette='viridis')
    plt.title('鸢尾花数据的PCA降维可视化')
    plt.xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]:.2f})')
    plt.savefig('./lab1/iris_pca_visualization.png')
    plt.close()

# 数据预处理
def preprocess_data(df):
    """
    对数据进行预处理
    """
    print("=" * 50)
    print("数据预处理:")
    print("=" * 50)
    
    # 1. 处理缺失值 (虽然这个数据集中没有缺失值，但我们仍然展示这个步骤)
    print("\n处理缺失值:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # 2. 异常值检测与处理
    print("\n异常值检测:")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            print(f"列 {col} 中发现 {len(outliers)} 个异常值")
            
            # 用上下限替换异常值
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    # 3. 特征编码 - 将类别特征转换为数值
    print("\n特征编码:")
    label_encoder = LabelEncoder()
    df['species_encoded'] = label_encoder.fit_transform(df['species'])
    print("编码后的类别映射:")
    for i, category in enumerate(label_encoder.classes_):
        print(f"{category} -> {i}")
    
    # 4. 特征分割
    print("\n特征分割:")
    X = df[numeric_cols]
    y = df['species_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, label_encoder

# 数据变换
def transform_data(X_train, X_test, y_train):
    """
    对数据进行各种变换
    """
    print("=" * 50)
    print("数据变换:")
    print("=" * 50)
    
    # 1. 标准化
    print("\n标准化:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("标准化前的特征统计 (训练集):")
    print(f"均值: {X_train.mean().values}")
    print(f"标准差: {X_train.std().values}")
    
    print("\n标准化后的特征统计 (训练集):")
    print(f"均值: {X_train_scaled.mean(axis=0)}")
    print(f"标准差: {X_train_scaled.std(axis=0)}")
    
    # 2. 归一化
    print("\n归一化:")
    min_max_scaler = MinMaxScaler()
    X_train_normalized = min_max_scaler.fit_transform(X_train)
    X_test_normalized = min_max_scaler.transform(X_test)
    
    print("归一化后的特征范围 (训练集):")
    print(f"最小值: {X_train_normalized.min(axis=0)}")
    print(f"最大值: {X_train_normalized.max(axis=0)}")
    
    # 3. 主成分分析 (PCA)
    print("\n主成分分析 (PCA):")
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"解释方差比例: {pca.explained_variance_ratio_}")
    print(f"累积解释方差: {sum(pca.explained_variance_ratio_):.4f}")
    print(f"主成分的形状: {X_train_pca.shape}")
    
    # PCA 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k')
    plt.colorbar(label='Species')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: 将4维特征降为2维')
    plt.savefig('./lab1/iris_pca.png')
    plt.close()
    
    return X_train_scaled, X_test_scaled, X_train_normalized, X_test_normalized, X_train_pca, X_test_pca, scaler, min_max_scaler, pca

# 特征选择与生成
def feature_engineering(df):
    """
    特征工程：特征选择与新特征生成
    """
    print("=" * 50)
    print("特征工程:")
    print("=" * 50)
    
    # 创建一个新的数据框来存储原始特征和新特征
    df_features = df.copy()
    
    # 1. 新特征生成 - 比率特征
    print("\n创建比率特征:")
    df_features['sepal_ratio'] = df['sepal_length'] / df['sepal_width']
    df_features['petal_ratio'] = df['petal_length'] / df['petal_width']
    
    # 2. 新特征生成 - 面积特征
    print("创建面积特征:")
    df_features['sepal_area'] = df['sepal_length'] * df['sepal_width']
    df_features['petal_area'] = df['petal_length'] * df['petal_width']
    
    print("\n新增特征统计:")
    print(df_features[['sepal_ratio', 'petal_ratio', 'sepal_area', 'petal_area']].describe())
    
    # 3. 特征重要性分析
    print("\n特征相关性分析:")
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    # 移除目标变量编码
    if 'species_encoded' in numeric_cols:
        numeric_cols = numeric_cols.drop('species_encoded')
    
    corr = df_features[numeric_cols].corr()
    
    # 与目标变量的相关性（如果存在species_encoded）
    if 'species_encoded' in df_features.columns:
        target_corr = df_features[numeric_cols].corrwith(df_features['species_encoded'])
        print("各特征与目标变量的相关性:")
        print(target_corr.sort_values(ascending=False))
    
    # 特征相关性可视化
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('特征相关性热图')
    plt.savefig('./lab1/iris_feature_correlation.png')
    plt.close()
    
    return df_features

# 保存预处理后的数据到lab2文件夹
def save_processed_data(df, X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler):
    """
    保存处理后的数据到第二个实验文件夹
    """
    # 确保目录存在
    ensure_dir("./lab2/data")
    
    # 原始数据框
    df.to_csv("./lab2/data/iris_processed.csv", index=False)
    
    # 划分的数据集
    np.save("./lab2/data/X_train_scaled.npy", X_train_scaled)
    np.save("./lab2/data/X_test_scaled.npy", X_test_scaled)
    np.save("./lab2/data/y_train.npy", y_train)
    np.save("./lab2/data/y_test.npy", y_test)
    
    # 保存模型
    joblib.dump(scaler, "./lab2/data/scaler.joblib")
    joblib.dump(label_encoder, "./lab2/data/label_encoder.joblib")
    
    print("\n预处理后的数据已保存到'./lab2/data/'文件夹")

# 主函数
def main():
    # 使用绝对路径
    file_path = 'F:/opensource/iris/iris.data'
    
    # 1. 加载并检查数据
    df = load_and_examine_data(file_path)
    
    # 2. 数据可视化
    visualize_data(df)
    
    # 3. 特征工程
    df_features = feature_engineering(df)
    
    # 4. 数据预处理
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)
    
    # 5. 数据变换
    X_train_scaled, X_test_scaled, X_train_normalized, X_test_normalized, X_train_pca, X_test_pca, scaler, min_max_scaler, pca = transform_data(X_train, X_test, y_train)
    
    # 6. 保存预处理后的数据到第二个实验文件夹
    save_processed_data(df, X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler)
    
    print("\n数据预处理和变换完成。预处理后的数据已保存到lab2文件夹，可视化结果已保存到lab1文件夹。")

if __name__ == "__main__":
    main() 