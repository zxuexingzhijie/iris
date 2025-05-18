# 鸢尾花数据集处理与分析项目

## 项目简介
本项目基于经典的鸢尾花(Iris)数据集，实现了完整的数据科学流程，包括数据预处理、特征工程、数据变换、可视化分析以及多种机器学习算法的实现与比较。项目全面展示了包括分类、聚类和关联规则挖掘等多种数据分析方法，为处理实际数据科学任务提供了系统性参考。

## 项目结构
```
iris/
│
├── iris.data                # 原始鸢尾花数据集
├── iris.names              # 数据集说明文件
├── bezdekIris.data         # 另一版本的鸢尾花数据集
│
├── lab1/                   # 存放数据可视化结果
│   ├── iris_pairplot.png   # 散点图矩阵
│   ├── iris_visualization.png  # 箱线图和相关性热图
│   ├── iris_pca.png        # PCA降维可视化
│   ├── classification_results.png  # 分类结果比较
│   ├── clustering_comparison.png  # 聚类结果比较
│   └── association_rules_metrics.png  # 关联规则评估指标
│
├── lab2/                   # 存放处理后的数据
│   └── data/               # 预处理数据存储目录
│       ├── iris_processed.csv  # 处理后的数据
│       ├── X_train_scaled.npy  # 标准化后的训练集
│       ├── X_test_scaled.npy   # 标准化后的测试集
│       ├── association_rules.csv  # 挖掘的关联规则
│       └── frequent_itemsets.csv  # 挖掘的频繁项集
│
├── total/                  # 存放主要算法实现
│   ├── iris_preprocessing.py  # 数据预处理与特征工程
│   ├── iris_classification.py # 分类算法实现
│   ├── iris_clustering.py     # 聚类算法实现
│   └── iris_association_rules.py  # 关联规则挖掘
│
└── README.md               # 本文件
```

## 功能特性
本项目主要实现以下功能：

### 1. 数据预处理与特征工程
- 数据加载与探索性分析
- 异常值检测与处理（基于箱线图和Z-score方法）
- 特征创建（花瓣面积、花萼面积、比率特征等）
- 特征编码（类别标签编码）
- 数据分割（训练集/测试集）

### 2. 数据变换
- 标准化变换（Z-score标准化）
- 归一化变换（Min-Max归一化）
- PCA降维（主成分分析）

### 3. 数据可视化
- 特征分布及关系可视化
- 异常值检测可视化
- 降维后数据可视化
- 聚类结果可视化
- 关联规则可视化

### 4. 分类算法实现与比较
- 逻辑回归
- 支持向量机(SVM)
- 决策树
- 随机森林
- K近邻(KNN)

### 5. 聚类算法实现与比较
- K-means聚类
- 层次聚类
- DBSCAN密度聚类
- 聚类有效性评估（轮廓系数、调整兰德指数）

### 6. 关联规则挖掘
- 数据离散化处理
- 频繁项集挖掘（Apriori算法）
- 关联规则提取与评估
- 按花卉品种分析关联规则

## 主要发现
- PCA降维分析显示，前两个主成分可以保留约95.18%的信息
- 在sepal_width特征中发现了4个异常值
- 特征工程创建的复合特征（面积和比率）提高了分类模型性能
- K-means聚类在鸢尾花数据集上表现良好，轮廓系数达到0.55以上
- 分类模型中随机森林和SVM算法表现最佳，准确率超过95%
- 关联规则分析显示花瓣大小特征与花卉种类有很强的关联性

## 使用说明

### 环境要求
```
pandas
numpy
matplotlib
seaborn
scikit-learn
mlxtend (用于关联规则挖掘)
```

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行流程
1. 数据预处理与变换：
```bash
python total/iris_preprocessing.py
```

2. 分类算法实现与比较：
```bash
python total/iris_classification.py
```

3. 聚类分析：
```bash
python total/iris_clustering.py
```

4. 关联规则挖掘：
```bash
python total/iris_association_rules.py
```

## 注意事项
- 在Windows系统上运行时，已解决UnicodeEncodeError编码问题
- 数据可视化结果将保存在lab1目录下
- 预处理后的数据将保存在lab2/data目录下
- 每个算法模块可以独立运行，并会自动检查和加载需要的预处理数据 