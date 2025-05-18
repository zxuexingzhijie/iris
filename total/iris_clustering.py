#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import seaborn as sns
import os
import joblib
import locale
import warnings
warnings.filterwarnings('ignore')

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

def preprocess_data(df):
    """
    数据预处理：特征缩放
    """
    # 提取特征和标签
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y = df['species_encoded'].values  # 用于评估聚类结果
    species = df['species'].values    # 原始类别名称
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, species

def apply_kmeans(X, n_clusters=3):
    """
    应用K-means聚类算法
    """
    print("=" * 50)
    print("K-means聚类分析")
    print("=" * 50)
    
    # 创建并训练K-means模型
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    
    # 获取聚类中心
    centers = kmeans.cluster_centers_
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(X, y_pred)
    print(f"K-means聚类的轮廓系数: {silhouette_avg:.4f}")
    
    # 记录每个簇的样本量
    unique_labels, counts = np.unique(y_pred, return_counts=True)
    print("\n聚类结果分布:")
    for label, count in zip(unique_labels, counts):
        print(f"簇 {label}: {count} 个样本")
    
    return y_pred, centers, silhouette_avg

def apply_hierarchical(X, n_clusters=3):
    """
    应用层次聚类算法
    """
    print("\n" + "=" * 50)
    print("层次聚类分析")
    print("=" * 50)
    
    # 创建并训练层次聚类模型
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    y_pred = hierarchical.fit_predict(X)
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(X, y_pred)
    print(f"层次聚类的轮廓系数: {silhouette_avg:.4f}")
    
    # 记录每个簇的样本量
    unique_labels, counts = np.unique(y_pred, return_counts=True)
    print("\n聚类结果分布:")
    for label, count in zip(unique_labels, counts):
        print(f"簇 {label}: {count} 个样本")
    
    return y_pred, silhouette_avg

def apply_dbscan(X, eps=0.5, min_samples=5):
    """
    应用DBSCAN聚类算法
    """
    print("\n" + "=" * 50)
    print("DBSCAN聚类分析")
    print("=" * 50)
    
    # 创建并训练DBSCAN模型
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = dbscan.fit_predict(X)
    
    # 计算轮廓系数 (如果有多于一个簇)
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    print(f"DBSCAN检测到 {n_clusters} 个簇")
    
    if n_clusters > 1:
        # 排除噪声点进行轮廓系数计算
        if -1 in y_pred:
            silhouette_avg = silhouette_score(X[y_pred != -1], y_pred[y_pred != -1])
        else:
            silhouette_avg = silhouette_score(X, y_pred)
        print(f"DBSCAN聚类的轮廓系数: {silhouette_avg:.4f}")
    else:
        silhouette_avg = 0
        print("DBSCAN只检测到一个簇，无法计算轮廓系数")
    
    # 记录每个簇的样本量
    unique_labels, counts = np.unique(y_pred, return_counts=True)
    print("\n聚类结果分布:")
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"噪声点: {count} 个样本")
        else:
            print(f"簇 {label}: {count} 个样本")
    
    return y_pred, silhouette_avg

def evaluate_clustering(y_true, y_pred_kmeans, y_pred_hierarchical, y_pred_dbscan, silhouette_kmeans, silhouette_hierarchical, silhouette_dbscan):
    """
    评估聚类结果与真实标签的一致性（使用调整兰德指数）
    """
    print("\n" + "=" * 50)
    print("聚类结果评估")
    print("=" * 50)
    
    # 计算调整兰德指数
    ari_kmeans = adjusted_rand_score(y_true, y_pred_kmeans)
    ari_hierarchical = adjusted_rand_score(y_true, y_pred_hierarchical)
    ari_dbscan = adjusted_rand_score(y_true, y_pred_dbscan)
    
    print(f"K-means的调整兰德指数: {ari_kmeans:.4f}")
    print(f"层次聚类的调整兰德指数: {ari_hierarchical:.4f}")
    print(f"DBSCAN的调整兰德指数: {ari_dbscan:.4f}")
    
    # 创建总结表格
    results = pd.DataFrame({
        '算法': ['K-means', '层次聚类', 'DBSCAN'],
        '轮廓系数': [silhouette_kmeans, silhouette_hierarchical, silhouette_dbscan],
        '调整兰德指数': [ari_kmeans, ari_hierarchical, ari_dbscan]
    })
    
    print("\n聚类算法性能比较:")
    print(results)
    
    return results

def visualize_clusters(X, y_true, y_pred_kmeans, y_pred_hierarchical, y_pred_dbscan, species, kmeans_centers=None):
    """
    可视化聚类结果
    """
    print("\n" + "=" * 50)
    print("聚类结果可视化")
    print("=" * 50)
    
    # 使用PCA将数据降至二维以便可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 真实标签
    scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', alpha=0.8)
    axes[0, 0].set_title('真实类别')
    legend1 = axes[0, 0].legend(*scatter.legend_elements(), title="类别")
    axes[0, 0].add_artist(legend1)
    axes[0, 0].set_xlabel('主成分1')
    axes[0, 0].set_ylabel('主成分2')
    
    # 为每个类别添加标签
    for i, label in enumerate(np.unique(y_true)):
        idx = np.where(y_true == label)[0][0]
        axes[0, 0].annotate(species[idx], 
                           (X_pca[idx, 0], X_pca[idx, 1]),
                           xytext=(5, 5),
                           textcoords='offset points')
    
    # K-means聚类结果
    scatter = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_kmeans, cmap='viridis', alpha=0.8)
    axes[0, 1].set_title('K-means聚类结果')
    legend2 = axes[0, 1].legend(*scatter.legend_elements(), title="簇")
    axes[0, 1].add_artist(legend2)
    axes[0, 1].set_xlabel('主成分1')
    axes[0, 1].set_ylabel('主成分2')
    
    # 如果有聚类中心，也进行可视化
    if kmeans_centers is not None:
        centers_pca = pca.transform(kmeans_centers)
        axes[0, 1].scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=100, label='聚类中心')
        axes[0, 1].legend()
    
    # 层次聚类结果
    scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_hierarchical, cmap='viridis', alpha=0.8)
    axes[1, 0].set_title('层次聚类结果')
    legend3 = axes[1, 0].legend(*scatter.legend_elements(), title="簇")
    axes[1, 0].add_artist(legend3)
    axes[1, 0].set_xlabel('主成分1')
    axes[1, 0].set_ylabel('主成分2')
    
    # DBSCAN聚类结果
    scatter = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_dbscan, cmap='viridis', alpha=0.8)
    axes[1, 1].set_title('DBSCAN聚类结果')
    legend4 = axes[1, 1].legend(*scatter.legend_elements(), title="簇")
    axes[1, 1].add_artist(legend4)
    axes[1, 1].set_xlabel('主成分1')
    axes[1, 1].set_ylabel('主成分2')
    
    plt.tight_layout()
    
    # 确保目录存在
    ensure_dir("./lab1")
    
    # 保存图表
    plt.savefig('./lab1/clustering_comparison.png', dpi=300)
    plt.close()
    print("聚类结果可视化已保存至 './lab1/clustering_comparison.png'")
    
    # 创建混淆矩阵可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # K-means混淆矩阵
    confusion_matrix_kmeans = pd.crosstab(y_true, y_pred_kmeans, rownames=['真实标签'], colnames=['预测簇'])
    sns.heatmap(confusion_matrix_kmeans, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('K-means聚类混淆矩阵')
    
    # 层次聚类混淆矩阵
    confusion_matrix_hierarchical = pd.crosstab(y_true, y_pred_hierarchical, rownames=['真实标签'], colnames=['预测簇'])
    sns.heatmap(confusion_matrix_hierarchical, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('层次聚类混淆矩阵')
    
    # DBSCAN混淆矩阵
    confusion_matrix_dbscan = pd.crosstab(y_true, y_pred_dbscan, rownames=['真实标签'], colnames=['预测簇'])
    sns.heatmap(confusion_matrix_dbscan, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title('DBSCAN聚类混淆矩阵')
    
    plt.tight_layout()
    plt.savefig('./lab1/clustering_confusion_matrices.png', dpi=300)
    plt.close()
    print("聚类混淆矩阵已保存至 './lab1/clustering_confusion_matrices.png'")

def main():
    """
    主函数
    """
    print("=" * 50)
    print("鸢尾花数据集聚类分析")
    print("=" * 50)
    
    # 1. 加载预处理数据
    df = load_preprocessed_data()
    if df is None:
        return
    
    # 2. 数据预处理
    X_scaled, y_true, species = preprocess_data(df)
    
    # 3. 应用聚类算法
    # K-means
    y_pred_kmeans, kmeans_centers, silhouette_kmeans = apply_kmeans(X_scaled, n_clusters=3)
    
    # 层次聚类
    y_pred_hierarchical, silhouette_hierarchical = apply_hierarchical(X_scaled, n_clusters=3)
    
    # DBSCAN (优化参数)
    y_pred_dbscan, silhouette_dbscan = apply_dbscan(X_scaled, eps=0.8, min_samples=5)
    
    # 4. 评估聚类结果
    results = evaluate_clustering(y_true, y_pred_kmeans, y_pred_hierarchical, y_pred_dbscan, silhouette_kmeans, silhouette_hierarchical, silhouette_dbscan)
    
    # 5. 可视化聚类结果
    visualize_clusters(X_scaled, y_true, y_pred_kmeans, y_pred_hierarchical, y_pred_dbscan, species, kmeans_centers)
    
    # 6. 保存聚类结果
    results_dir = "./lab2/data"
    ensure_dir(results_dir)
    results.to_csv(os.path.join(results_dir, "clustering_results.csv"), index=False)
    
    print("\n鸢尾花数据集聚类分析完成!")

if __name__ == "__main__":
    main() 