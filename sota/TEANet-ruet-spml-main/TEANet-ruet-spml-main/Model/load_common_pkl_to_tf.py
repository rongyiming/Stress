"""
功能：读取通用NumPy格式的pkl文件，转换为TensorFlow Dataset对象
纯TensorFlow环境运行，无PyTorch依赖
"""
import pickle
import tensorflow as tf
import numpy as np


def load_common_pkl_to_tf(
        common_pkl_path="C:/Users/12992/Desktop/实验室/stress_pretrain/sota/TEANet-ruet-spml-main/TEANet-ruet-spml-main/finetune_dataset_common.pkl",
        batch_size=64,
        shuffle_buffer_size=1000
):
    """
    加载通用格式pkl，转换为TensorFlow Dataset

    Args:
        common_pkl_path: 通用格式pkl文件路径
        batch_size: TF Dataset的批次大小
        shuffle_buffer_size: 洗牌缓冲区大小

    Returns:
        tf_datasets: 字典，包含每个数据集的train/val/test TF Dataset
        data_info: 数据基本信息汇总
    """
    # 1. 加载通用格式pkl
    print(f"正在加载通用格式pkl文件：{common_pkl_path}")
    try:
        with open(common_pkl_path, 'rb') as f:
            common_data = pickle.load(f)
        print(f"成功加载，包含{len(common_data)}个数据集")
    except Exception as e:
        raise RuntimeError(f"加载通用pkl失败：{e}")

    # 2. 转换为TensorFlow Dataset
    tf_datasets = {}
    data_info = []

    for dataset_item in common_data:
        dataset_name = dataset_item["dataset_name"]
        print(f"\n处理数据集：{dataset_name}")

        # 定义转换函数
        def convert_to_tf_dataset(numpy_data, numpy_labels, is_train=False):
            """将NumPy数据转为TF Dataset"""
            # 转换为TF张量（自动适配类型）
            tf_data = tf.convert_to_tensor(numpy_data, dtype=tf.float32)
            if numpy_labels is not None:
                tf_labels = tf.convert_to_tensor(numpy_labels, dtype=tf.int64)
                dataset = tf.data.Dataset.from_tensor_slices((tf_data, tf_labels))
            else:
                dataset = tf.data.Dataset.from_tensor_slices(tf_data)

            # 训练集增加洗牌和预取（提升性能）
            if is_train:
                dataset = dataset.shuffle(shuffle_buffer_size)
                dataset = dataset.repeat()  # 可选：训练时重复迭代

            # 设置批次大小和预取
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset

        # 转换train/val/test
        train_ds = convert_to_tf_dataset(
            dataset_item["train"]["data"],
            dataset_item["train"]["labels"],
            is_train=True
        )
        val_ds = convert_to_tf_dataset(
            dataset_item["val"]["data"],
            dataset_item["val"]["labels"]
        )
        test_ds = convert_to_tf_dataset(
            dataset_item["test"]["data"],
            dataset_item["test"]["labels"]
        )

        # 保存结果
        tf_datasets[dataset_name] = {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds
        }

        # 记录数据信息
        data_info.append({
            "dataset_name": dataset_name,
            "train_samples": dataset_item["train"]["sample_num"],
            "val_samples": dataset_item["val"]["sample_num"],
            "test_samples": dataset_item["test"]["sample_num"],
            "feature_shape": dataset_item["train"]["feature_shape"]
        })

    # 3. 打印数据汇总信息
    print("\n=== 数据转换完成 ===")
    for info in data_info:
        print(f"\n数据集：{info['dataset_name']}")
        print(f"  训练集样本数：{info['train_samples']}")
        print(f"  验证集样本数：{info['val_samples']}")
        print(f"  测试集样本数：{info['test_samples']}")
        print(f"  特征维度：{info['feature_shape']}")

    return tf_datasets, data_info


# ------------------- 测试使用示例 -------------------
def test_tf_dataset(tf_datasets):
    """测试TF Dataset是否可用"""
    # 取第一个数据集的训练集测试
    first_dataset_name = list(tf_datasets.keys())[0]
    train_ds = tf_datasets[first_dataset_name]["train"]

    print(f"\n=== 测试{first_dataset_name}训练集 ===")
    # 取前2个批次验证
    for batch_idx, (batch_data, batch_labels) in enumerate(train_ds.take(2)):
        print(f"\n批次{batch_idx + 1}：")
        print(f"  数据形状：{batch_data.shape}")
        print(f"  标签形状：{batch_labels.shape}")
        print(f"  第一个样本标签：{batch_labels[0].numpy()}")


if __name__ == "__main__":

    # 加载并转换为TF Dataset
    tf_ds, info = load_common_pkl_to_tf()

    # 测试数据集
    test_tf_dataset(tf_ds)

    # 示例：使用某个数据集的训练集/验证集
    # dataset_name = "your_dataset_name"
    # train_ds = tf_ds[dataset_name]["train"]
    # val_ds = tf_ds[dataset_name]["val"]
    # 可直接传入model.fit(train_ds, validation_data=val_ds, epochs=10)