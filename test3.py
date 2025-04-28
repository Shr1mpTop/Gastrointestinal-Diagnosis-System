import numpy as np
import pandas as pd
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 生成一些示例数据
np.random.seed(42)
n_samples = 100
data = {
    'color': np.random.choice(['red', 'blue', 'green'], n_samples),
    'texture': np.random.choice(['smooth', 'rough'], n_samples),
   'shape': np.random.choice(['circle','square', 'triangle'], n_samples),
    'target': np.random.randint(0, 2, n_samples)
}
df = pd.DataFrame(data)

# 将分类特征编码为数值
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
df[['color', 'texture','shape']] = encoder.fit_transform(df[['color', 'texture','shape']])

# 划分训练集和测试集
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练一个决策树模型
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 计算 SHAP 值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 绘制 SHAP 摘要图
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)