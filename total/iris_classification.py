#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import time
import os
import joblib
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

def check_and_load_processed_data():
    """
    检查并加载预处理后的数据
    """
    data_dir = "./data"  # 相对于当前脚本的路径
    
    # 检查是否存在预处理后的数据
    if os.path.exists(data_dir) and all(os.path.exists(os.path.join(data_dir, f)) 
                                      for f in ["X_train_scaled.npy", "X_test_scaled.npy", 
                                                "y_train.npy", "y_test.npy",
                                                "label_encoder.joblib"]):
        print("加载预处理后的数据...")
        X_train = np.load(os.path.join(data_dir, "X_train_scaled.npy"))
        X_test = np.load(os.path.join(data_dir, "X_test_scaled.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))
        y_test = np.load(os.path.join(data_dir, "y_test.npy"))
        label_encoder = joblib.load(os.path.join(data_dir, "label_encoder.joblib"))
        
        # 合并训练集和测试集用于交叉验证
        X = np.vstack((X_train, X_test))
        y = np.concatenate((y_train, y_test))
        
        return X, y, label_encoder.classes_
    else:
        print("未找到预处理后的数据，从原始数据加载...")
        return load_data('F:/opensource/iris/iris.data')  # 使用绝对路径

def load_data(file_path):
    """
    加载数据集
    """
    # 读取数据集
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv(file_path, header=None, names=column_names)
    
    # 特征和标签
    X = df.iloc[:, :4].values
    y = df['species'].values
    
    # 将标签编码为数值
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, label_encoder.classes_

def train_classifiers(X, y, cv=5):
    """
    训练多种分类器并使用交叉验证评估性能
    """
    # 定义要评估的分类器
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }
    
    # 定义评估指标
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr']
    
    # 存储结果
    results = {}
    cv_results = {}
    training_time = {}
    
    # 设定交叉验证
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 对每个分类器进行评估
    for name, classifier in classifiers.items():
        print(f"评估分类器: {name}")
        
        # 记录训练时间
        start_time = time.time()
        
        # 执行交叉验证
        cv_result = cross_validate(
            classifier, X, y, 
            cv=kfold, 
            scoring=scoring, 
            return_train_score=True,
            n_jobs=1  # 设置为1避免并行处理导致的编码问题
        )
        
        # 记录训练时间
        end_time = time.time()
        training_time[name] = end_time - start_time
        
        # 存储交叉验证的详细结果
        cv_results[name] = cv_result
        
        # 计算平均性能指标
        results[name] = {
            'accuracy': cv_result['test_accuracy'].mean(),
            'precision': cv_result['test_precision_macro'].mean(),
            'recall': cv_result['test_recall_macro'].mean(),
            'f1': cv_result['test_f1_macro'].mean(),
            'roc_auc': cv_result['test_roc_auc_ovr'].mean(),
            'std_accuracy': cv_result['test_accuracy'].std(),
            'training_time': training_time[name]
        }
        
    return results, cv_results

def visualize_results(results):
    """
    可视化不同分类器的性能比较
    """
    # 确保目录存在
    ensure_dir("F:/opensource/iris/total/lab1")
    
    # 将结果转换为DataFrame便于可视化
    df_results = pd.DataFrame(results).T
    
    # 准备图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 准确率比较
    df_results[['accuracy']].plot(kind='bar', yerr=df_results['std_accuracy'], 
                                  ax=axes[0, 0], color='skyblue', edgecolor='black')
    axes[0, 0].set_title('分类器准确率比较')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].set_ylim([0.8, 1.0])  # 调整y轴范围以便更好地显示差异
    
    # 2. 精确率、召回率和F1分数比较
    df_results[['precision', 'recall', 'f1']].plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('精确率、召回率和F1分数')
    axes[0, 1].set_ylabel('得分')
    
    # 3. ROC AUC比较
    df_results[['roc_auc']].plot(kind='bar', ax=axes[1, 0], color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('ROC AUC比较')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_ylim([0.8, 1.0])  # 调整y轴范围以便更好地显示差异
    
    # 4. 训练时间比较
    df_results[['training_time']].plot(kind='bar', ax=axes[1, 1], color='salmon', edgecolor='black')
    axes[1, 1].set_title('训练时间比较')
    axes[1, 1].set_ylabel('时间（秒）')
    
    plt.tight_layout()
    plt.savefig('F:/opensource/iris/total/lab1/classification_results.png')
    plt.close()
    
    # 创建综合性能表格可视化
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_results[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']], 
                annot=True, cmap='YlGnBu', fmt='.4f', linewidths=0.5)
    plt.title('分类器性能比较热图')
    plt.savefig('F:/opensource/iris/total/lab1/classification_heatmap.png')
    plt.close()

def analyze_best_classifier(X, y, best_classifier_name):
    """
    对表现最好的分类器进行更详细的分析
    """
    # 选择最佳分类器
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }
    best_clf = classifiers[best_classifier_name]
    
    # 使用训练测试集分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练最佳分类器
    best_clf.fit(X_train, y_train)
    
    # 预测
    y_pred = best_clf.predict(X_test)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 创建分类报告
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{best_classifier_name} 混淆矩阵')
    plt.ylabel('实际类别')
    plt.xlabel('预测类别')
    plt.savefig('F:/opensource/iris/total/lab1/best_classifier_confusion_matrix.png')
    plt.close()
    
    return report

def main():
    """
    主函数
    """
    print("=" * 50)
    print("分类算法性能比较")
    print("=" * 50)
    
    # 1. 加载数据
    X, y, class_names = check_and_load_processed_data()
    print(f"数据集大小: {X.shape}")
    print(f"类别: {class_names}")
    
    # 2. 训练并评估分类器
    results, cv_results = train_classifiers(X, y, cv=5)
    
    # 打印结果
    print("\n分类器性能比较:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  准确率: {metrics['accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  训练时间: {metrics['training_time']:.4f} 秒")
    
    # 3. 可视化结果比较
    visualize_results(results)
    
    # 4. 寻找表现最好的分类器
    best_classifier = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\n最佳分类器: {best_classifier}")
    
    # 5. 对最佳分类器进行详细分析
    best_report = analyze_best_classifier(X, y, best_classifier)
    print(f"\n{best_classifier} 详细性能:")
    for cls, metrics in best_report.items():
        if cls in ['0', '1', '2']:  # 只打印各个类别的指标
            print(f"  类别 {cls} ({class_names[int(cls)]}):")
            print(f"    精确率: {metrics['precision']:.4f}")
            print(f"    召回率: {metrics['recall']:.4f}")
            print(f"    F1分数: {metrics['f1-score']:.4f}")  # 修正键名为'f1-score'
    
    print("\n分析完成。结果已保存为图像。")

if __name__ == "__main__":
    main() 