import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_curve, auc, recall_score, classification_report,
                             precision_score, roc_auc_score, cohen_kappa_score)
from keras.callbacks import EarlyStopping
from keras import models
import pickle  # 新增：pkl加载依赖
from load_common_pkl_to_tf import load_common_pkl_to_tf  # 导入pkl转换函数
from TEANet_model import TEANet_model

# ===================== 1. 初始化评估指标 =====================
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
auc_scores = []
kappa_scores = []
conf_matrices = []
histories = []
all_y_true = []
all_y_pred_prob = []

# ===================== 2. 加载并转换PKL数据 =====================
# 配置参数
PKL_FILE_PATH = "C:/Users/12992/Desktop/实验室/stress_pretrain/sota/TEANet-ruet-spml-main/TEANet-ruet-spml-main/finetune_dataset_common.pkl"
BATCH_SIZE = 64
SHUFFLE_BUFFER = 1000

# 加载pkl并转换为TF Dataset
tf_datasets, data_info = load_common_pkl_to_tf(
    common_pkl_path=PKL_FILE_PATH,
    batch_size=BATCH_SIZE,
    shuffle_buffer_size=SHUFFLE_BUFFER
)

# 选择目标数据集（根据实际pkl中的数据集名称调整，示例取第一个数据集）
TARGET_DATASET = list(tf_datasets.keys())[1]
train_ds = tf_datasets[TARGET_DATASET]["train"]
val_ds = tf_datasets[TARGET_DATASET]["val"]
test_ds = tf_datasets[TARGET_DATASET]["test"]

# 获取特征维度（用于模型输入）
feature_shape = next(item for item in data_info if item["dataset_name"] == TARGET_DATASET)["feature_shape"]
input_shape = (feature_shape[-1], 1)  # 适配原模型输入格式

# 编译模型
num_classes = 2  # WESAD数据集为二分类（normal/stressed）
model = TEANet_model(input_shape, num_classes)
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# 早停策略
es = EarlyStopping(
    monitor='val_accuracy',
    patience=70,
    verbose=1,
    mode='max',
    restore_best_weights=True
)

# ===================== 4. 模型训练 =====================
print(f"===== 训练数据集：{TARGET_DATASET} =====")
# 计算训练步数（适配repeat()的无限迭代）
train_sample_num = next(item for item in data_info if item["dataset_name"] == TARGET_DATASET)["train_samples"]
steps_per_epoch = train_sample_num // BATCH_SIZE

# 训练模型
history = model.fit(
    train_ds,
    epochs=1000,
    steps_per_epoch=steps_per_epoch,  # 限制每个epoch的步数
    validation_data=val_ds,
    callbacks=[es],
    verbose=1
)
histories.append(history)

# ===================== 5. 模型评估 =====================
# 提取测试集标签和预测结果
test_labels = []
test_pred_prob = []

# 遍历测试集获取所有样本
for batch_data, batch_labels in test_ds:
    # 预测当前批次
    batch_pred = model.predict(batch_data, verbose=0)
    # 收集标签和预测概率
    test_labels.extend(batch_labels.numpy())
    test_pred_prob.extend(batch_pred)

# 转换为numpy数组
test_labels = np.array(test_labels)
test_pred_prob = np.array(test_pred_prob)
test_pred = np.argmax(test_pred_prob, axis=1)

# 计算评估指标
accuracy = accuracy_score(test_labels, test_pred)
f1 = f1_score(test_labels, test_pred)
recall_in = recall_score(test_labels, test_pred)
precision_score_in = precision_score(test_labels, test_pred)
kappa_scores_in = cohen_kappa_score(test_labels, test_pred)
auc_scores_in = roc_auc_score(test_labels, test_pred_prob[:, 1])

# 保存指标到列表
accuracy_scores.append(accuracy)
f1_scores.append(f1)
conf_matrices.append(confusion_matrix(test_labels, test_pred))
precision_scores.append(precision_score_in)
recall_scores.append(recall_in)
kappa_scores.append(kappa_scores_in)
auc_scores.append(auc_scores_in)

# 汇总所有标签和预测概率
all_y_true.extend(test_labels)
all_y_pred_prob.extend(test_pred_prob[:, 1])

# ===================== 6. 打印评估结果 =====================
print(f"\n===== {TARGET_DATASET} 评估结果 =====")
print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision_score_in:.4f}")
print(f"召回率: {recall_in:.4f}")
print(f"F1分数: {f1:.4f}")
print(f"Kappa系数: {kappa_scores_in:.4f}")
print(f"AUC值: {auc_scores_in:.4f}")
print("\n混淆矩阵:")
print(confusion_matrix(test_labels, test_pred))
print("\n分类报告:")
print(classification_report(test_labels, test_pred, target_names=['normal', 'stressed']))