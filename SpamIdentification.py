
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression #逻辑回归

# 读取文件路径
file_path = r"d:\Test\spam.csv"
data = pd.read_csv(file_path, encoding='latin1')

#检查能否读取数据
print("数据预览：")

#将读取到的数据colume改名
data = data[['v1', 'v2']]
data = data.rename(columns={'v1': 'label', 'v2': 'text'})  # 重命名列名为更直观
# 将标签映射为数字：`ham` -> 0, `spam` -> 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 检查数据预览
print(data.head())


# 将文本向量化 提取特征和标签
X = data['text']  # 短信文本
y = data['label']  # 标签

# 使用 CountVectorizer 将文本转换为数值表示
vectorizer = CountVectorizer()

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对训练数据进行向量化
X_train_vectorized = vectorizer.fit_transform(X_train)

# 对测试数据进行向量化
X_test_vectorized = vectorizer.transform(X_test)

# 检查向量化结果的形状
print(f"X_train_vectorized shape: {X_train_vectorized.shape}")
print(f"X_test_vectorized shape: {X_test_vectorized.shape}")

#训练模型：
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
print("Model training completed!")

#预测数据结果：
y_pred = model.predict(X_test_vectorized)
#使用accuracy_score和classification_report评估预测
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))

#使用新的电子邮件示例测试模型：
test_emails = ["Win a free iPhone! Click here!", "Meeting at 3 PM tomorrow."]
test_emails_vectorized = vectorizer.transform(test_emails)
predictions = model.predict(test_emails_vectorized)
print(predictions)

#创建条形图
plt.bar(['Spam', 'Not Spam'], [sum(y_pred == 1), sum(y_pred == 0)])
plt.title('Model Predictions')
plt.show()