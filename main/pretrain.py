import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataset.dataset import pretrain_create_dataloader
from main.model import pretrainModel
import random
import numpy as np
import os
from tqdm import tqdm  # 导入tqdm库用于进度条
import time  # 用于计算训练时间
from datetime import datetime


local_now = datetime.now()

def set_seed(seed=42):
    """固定所有随机种子（支持CPU/GPU）"""
    # 基础随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch设置
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时使用
    
    # 深度确定性设置（可能降低性能但提高复现性）
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 固定CUDA卷积算法

ALL_DATASETS = ['CLAS', 'WESAD', 'MTSPD']
DATASETS = ['CLAS', 'WESAD']

chosenlabels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def parse_args():
    parser = argparse.ArgumentParser(description="模型进行时间序列预训练")
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='计算设备')
    parser.add_argument('--label', type=int, default=-1, help='预训练标签类型')
    parser.add_argument('--dataset', type=str, nargs='+', choices=ALL_DATASETS, default=DATASETS, help='选择数据集: CLAS, WESAD, MTSPD')
    parser.add_argument('--datalength', type=int, default=10)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--resnet_depth', type=int, default=18, help='ResNet深度选择: 18, 34, 50')
    parser.add_argument('--frequency', type=int, default=32, help='数据采样频率')
    return parser.parse_args()

def analyze_feature_variability(features):
    """分析特征的变异性"""
    features = features.detach().cpu()  # 转CPU，脱离计算图
    batch_size = features.shape[0]  # 32
    feature_dim = features.shape[1]  # 256

    print("\n=== Batch内样本差异性分析 ===")

    # 1. 样本间余弦相似度（衡量向量方向差异）
    # 原理：余弦相似度∈[0,1]，值越接近0表示样本特征方向差异越大，越接近1表示方向越相似
    from torch.nn.functional import cosine_similarity
    # 计算所有样本两两之间的余弦相似度（得到32×32的相似度矩阵）
    sim_matrix = torch.zeros((batch_size, batch_size))
    for i in range(batch_size):
        for j in range(batch_size):
            sim_matrix[i, j] = cosine_similarity(features[i].unsqueeze(0), 
                                                features[j].unsqueeze(0), 
                                                dim=1)
    # 统计相似度分布（排除对角线“自身与自身的相似度=1”）
    off_diag_sim = sim_matrix[~torch.eye(batch_size, dtype=bool)]  # 取非对角线元素
    print(f"1. 样本间余弦相似度统计：")
    print(f"   - 均值：{off_diag_sim.mean():.4f}")  # 均值越小，整体差异越大
    print(f"   - 标准差：{off_diag_sim.std():.4f}")  # 标准差越大，样本间差异越不均衡
    print(f"   - 最小值：{off_diag_sim.min():.4f}")  # 最小相似度（差异最大的样本对）
    print(f"   - 最大值：{off_diag_sim.max():.4f}")  # 最大相似度（差异最小的样本对）

    # 2. 样本间欧氏距离（衡量向量空间距离差异）
    # 原理：欧氏距离越大，样本特征在空间中的位置越远，差异越大
    euclid_dist = torch.cdist(features, features, p=2)  # 32×32距离矩阵
    off_diag_dist = euclid_dist[~torch.eye(batch_size, dtype=bool)]
    print(f"\n2. 样本间欧氏距离统计：")
    print(f"   - 均值：{off_diag_dist.mean():.4f}")  # 均值越大，整体差异越大
    print(f"   - 标准差：{off_diag_dist.std():.4f}")
    print(f"   - 最小值：{off_diag_dist.min():.4f}")
    print(f"   - 最大值：{off_diag_dist.max():.4f}")

    # 3. 特征维度的方差（衡量单特征在batch内的离散度）
    # 原理：每个特征维度的方差越大，说明该特征对不同样本的区分能力越强
    feature_var = torch.var(features, dim=0)  # 对每个特征维度算方差（256个值）
    print(f"\n3. 特征维度方差统计：")
    print(f"   - 所有特征方差均值：{feature_var.mean():.6f}")  # 整体特征离散度
    print(f"   - 方差大于0.1的特征数：{torch.sum(feature_var > 0.1).item()}")  # 高区分度特征数量
    print(f"   - 方差最小的10个特征均值：{torch.topk(feature_var, 10, largest=False).values.mean():.6f}")  # 低区分度特征情况

    # 4. 样本特征的L2范数（衡量单样本特征的整体强度差异）
    # 原理：范数差异大，说明样本特征的“强度”有区分度（非必要，但可辅助判断）
    sample_norm = torch.norm(features, dim=1)  # 每个样本的L2范数（32个值）
    print(f"\n4. 样本特征L2范数统计：")
    print(f"   - 均值：{sample_norm.mean():.4f}")
    print(f"   - 标准差：{sample_norm.std():.4f}")
    print(f"   - 最大值/最小值：{sample_norm.max():.4f}/{sample_norm.min():.4f}")

    return

def loss_function(outputs, labels, std, criterion):
    total_loss = 0
    for i in range(len(chosenlabels)):
        # total_loss += criterion(outputs[:, i], labels[:, i])/std[chosenlabels[i]]
        total_loss += criterion(outputs[:, i], labels[:, i])
    return total_loss

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    if not args.label == -1:
        chosenlabels = [args.label]
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 创建数据加载器
    train_loader, val_loader, test_loader, std_per_type = pretrain_create_dataloader(batch_size=args.batch_size, T=args.datalength, frequency=args.frequency, overlap=args.overlap, datasets=args.dataset)

    # 初始化模型、损失函数和优化器
    model = pretrainModel(input_channels=1, resnet_depth=args.resnet_depth, hidden_size=64, output_size=len(chosenlabels))
    # model = model.double()

    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    short_time = local_now.strftime("%Y%m%d_%H%M%S")
    dataname = [x[0] for x in args.dataset]
    dataname = "".join(dataname)
    model_path = f'./save/models/pretrainmodel_{args.resnet_depth}_{dataname}_{args.datalength}X{args.frequency}_label{args.label}_pretrained.pth'
    best_val_loss = float('inf')
    losscnt = 0

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        first = True
        for inputs, labels, dlabels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            inputs, labels = inputs.to(device), labels[:, chosenlabels].to(device)

            optimizer.zero_grad()
            if first and epoch % 5 == 0:  # 每5个epoch打印一次，避免输出过多
                first = False
                print(f"输入形状: {inputs.shape}")  # (batch, channels, seq_len)
                outputs, tmp_out = model(inputs, return_intermediates=True)
                # 打印tmp输出特征统计
                print("\n=== tmp输出特征统计 ===")
                print(f"输出形状: {tmp_out.shape}")  # (batch, channels, seq_len_reduced)
                tmp_out = tmp_out.reshape(inputs.size(0), -1) # 展平为(batch, feature_dim)
                analyze_feature_variability(tmp_out)
                
            else:
                outputs, tmp_out = model(inputs, return_intermediates=True)

            if outputs.size(0) != labels.size(0):
                quit("输出和标签的批次大小不匹配！")
            
            total_loss = loss_function(outputs, labels, std_per_type, criterion)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.6f}, Time: {elapsed_time:.6f}s")
        # 评估模型在验证集上的表现
        model.eval()
        val_losses = 0.0
        with torch.no_grad():
            for val_inputs, val_labels, val_dlabels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels[:, chosenlabels].to(device)
                val_outputs = model(val_inputs)
                val_loss = loss_function(val_outputs, val_labels, std_per_type, criterion)
                val_losses += val_loss.item() * val_inputs.size(0)

        val_loss_epoch = val_losses / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{args.epochs}, Val Loss: {val_loss_epoch:.6f}")
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(model.state_dict(), model_path)
            losscnt = 0
        elif losscnt > 10 and running_loss < best_val_loss:
            print("验证损失未降低，提前停止训练。")
            break
        losscnt += 1

    print("训练完成！")
    print(f"模型已保存到 {model_path}")

    # 在测试集上评估模型
    model.eval()
    test_losses = 0.0
    mse_loss = nn.MSELoss()
    mse_total = 0.0
    total_samples = 0
    labels = []
    with torch.no_grad():
        first = True
        for test_inputs, test_labels, test_dlabels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels[:, chosenlabels].to(device)
            test_outputs = model(test_inputs)
            test_loss = loss_function(test_outputs, test_labels, std_per_type, criterion)
            test_losses += test_loss.item() * test_inputs.size(0)
            labels.extend(test_labels.cpu().tolist())
            if first:
                first = False
                print("测试标签：", test_labels.cpu().tolist())
                print("测试预测：", test_outputs.cpu().tolist())
            mse_total += mse_loss(test_outputs, test_labels)
            total_samples += test_inputs.size(0)

    arr = np.array(labels)  # 转换为numpy数组

    # 总体标准差（默认ddof=0）
    pop_std = np.std(arr)
    print("总体标准差：", pop_std)

    test_loss_epoch = test_losses / len(test_loader.dataset)
    print(f"测试集损失: {test_loss_epoch:.4f}")
    overall_mse = mse_total / total_samples  # 整体MSE
    overall_rmse = torch.sqrt(overall_mse)
    print(f"整体RMSE: {overall_rmse:.4f}")
