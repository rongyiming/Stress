import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset.dataset import finetune_pd_create_dataloader
from main.model import ResLSTMModel, finetuneModel, pretrainModel
import random
import numpy as np
import os
from tqdm import tqdm  # 导入tqdm库用于进度条
import time  # 用于计算训练时间
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import torch.nn.init as init

local_now = datetime.now()


# 初始化输出层的权重和偏置
def init_output_layer(layer):
    # 权重初始化：若前一层用ReLU，推荐Kaiming正态分布
    init.xavier_uniform_(layer.weight, gain=1.0)  # gain是缩放因子，默认1.0
    # 偏置初始化：通常设为0
    init.zeros_(layer.bias)


def set_seed(seed=1024):
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

chosenlabels = [0, 1, 5]

def parse_args():
    parser = argparse.ArgumentParser(description="模型进行时间序列预训练")
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='计算设备')
    parser.add_argument('--label', type=int, default=-1, help='预训练标签类型')
    parser.add_argument('--model_name', type=str, default='ResLSTM', help='模型名称')
    parser.add_argument('--pretrain_name', type=str, default='pretrainmodel', help='预训练模型名称')
    parser.add_argument('--dataset', type=str, nargs='+', choices=ALL_DATASETS, default=DATASETS, help='选择数据集: CLAS, WESAD, MTSPD')
    parser.add_argument('--datalength', type=int, default=10)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--resnet_depth', type=int, default=18, help='ResNet深度选择: 18, 34, 50')
    parser.add_argument('--frequency', type=int, default=32, help='数据采样频率')
    parser.add_argument('--lambda1', type=float, default=0.01, help='MoE负载均衡Loss权重')
    parser.add_argument('--lambda2', type=float, default=0.5, help='特征-参数一致性Loss权重')
    return parser.parse_args()

def freeze_resnet_except_layer2(model):
    """
    冻结ResLSTMModel中ResNet部分除layer2之外的所有参数
    :param model: ResLSTMModel实例
    """
    # 获取ResNet部分
    resnet = model.resnet
    
    # 冻结初始卷积层和批归一化层
    for param in resnet.conv1.parameters():
        param.requires_grad = False
    for param in resnet.bn1.parameters():
        param.requires_grad = False
    
    # 冻结最大池化层（如果有可训练参数，通常MaxPool1d没有）
    for param in resnet.maxpool.parameters():
        param.requires_grad = False
    
    # 冻结layer1
    for param in resnet.layer1.parameters():
        param.requires_grad = False
    
    # 解冻layer2（确保其参数可训练）
    for param in resnet.layer2.parameters():
        param.requires_grad = True

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    with open(f'./main/resualt.log', 'a', encoding='utf-8') as f:
        f.write(f"lambda1:{args.lambda1};lambda2:{args.lambda2}\n")

    if not args.label == -1:
        chosenlabels = [args.label]
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 创建数据加载器
    datasetlist = finetune_pd_create_dataloader(batch_size=args.batch_size, T=args.datalength, frequency=args.frequency, overlap=args.overlap, datasets=args.dataset)

    alpha = 0.6  # 融合权重，控制个体特征和输入特征的融合比例，范围[0,1]，可调节以优化性能
    with open(f'./main/resualt.log', 'a', encoding='utf-8') as f:
        f.write(f"alpha:{alpha}\n")
    # 初始化模型、损失函数和优化器
    model = finetuneModel(input_channels=1, resnet_depth=args.resnet_depth, lstm_input_size=256, lstm_hidden_size=64, lstm_num_layers=1, output_size=2, expert_num=10, alpha=alpha)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),  weight_decay=1e-3)

    pretrain_model = pretrainModel(input_channels=1, resnet_depth=args.resnet_depth, hidden_size=64, output_size=len(chosenlabels))

    load_path = f'./save/models/{args.pretrain_name}_{args.resnet_depth}_CW_{args.datalength}X{args.frequency}_label{args.label}_pretrained.pth'
    pretrained_weights = torch.load(load_path)

    pretrain_model.load_state_dict(pretrained_weights, strict=False)  # strict=False忽略无关参数

    # ===================== 3. 提取ResNet部分的参数 =====================
    # 过滤出key以"resnet."开头的参数（即ResNet模块的参数）
    resnet_pretrain_params = {
        k: v for k, v in pretrain_model.state_dict().items()
        if k.startswith("resnet.")
    }
    

    # 训练循环
    for dataset_name, train_loader, val_loader, _ in datasetlist:
        best_val_loss = float('inf')
        losscnt = 0
        model_path = f'./save/finetune/{args.model_name}_{args.resnet_depth}_{dataset_name}_{args.datalength}_label{args.label}_finetune.pth'
        model.load_state_dict(resnet_pretrain_params, strict=False)
        model.finetune()

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            c_loss = 0.0
            moe_loss = 0.0
            consist_loss = 0.0
            start_time = time.time()

            for inputs, labels, _, individual in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
                inputs, labels, individual = inputs.to(device), labels.to(device), individual.to(device)
                optimizer.zero_grad()
                outputs, (loss1, loss2) = model(inputs, individual=individual)
                if outputs.size(0) != labels.size(0):
                    quit("输出和标签的批次大小不匹配！")
                crossEloss = criterion(outputs, labels)
                loss = crossEloss + args.lambda1 * loss1 + args.lambda2 * loss2
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                c_loss += crossEloss.item() * inputs.size(0)
                moe_loss += loss1.item() * inputs.size(0)
                consist_loss += loss2.item() * inputs.size(0)

            length = len(train_loader.dataset)
            epoch_loss = running_loss / length
            ec_loss = c_loss / length
            em_loss = moe_loss / length
            econsist_loss = consist_loss / length
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{args.epochs}, Cross entropy Loss: {ec_loss:.4f}, Moe Loss: {em_loss:.4f}, Consist Loss: {econsist_loss:.4f}, Time: {elapsed_time:.2f}s")
            # 评估模型在验证集上的表现
            model.eval()
            val_losses = 0.0
            with torch.no_grad():
                for val_inputs, val_labels, _, val_individual in val_loader:
                    val_inputs, val_labels, val_individual = val_inputs.to(device), val_labels.to(device), val_individual.to(device)
                    val_outputs, _ = model(val_inputs, individual=val_individual)
                    val_loss = criterion(val_outputs, val_labels)
                    val_losses += val_loss.item() * val_inputs.size(0)

            val_loss_epoch = val_losses / len(val_loader.dataset)
            print(f"Epoch {epoch+1}/{args.epochs}, Val Loss: {val_loss_epoch:.4f}")
            losscnt += 1
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                torch.save(model.state_dict(), model_path)
                losscnt = 0
            elif losscnt > 50:
                print("验证损失未降低，提前停止训练。")
                break

        print("训练完成！")
        print(f"模型已保存到 {model_path}")
        for dataset_name2, train_loader, val_loader, test_loader in datasetlist:
            # 在测试集上评估模型
            model.eval()
            test_losses = 0.0

            total_samples = 0
            labels = []
            preds = []
            with torch.no_grad():
                for test_inputs, test_labels, _, test_individual in test_loader:

                    test_inputs, test_labels, test_individual = test_inputs.to(device), test_labels.to(device), test_individual.to(device)
                    test_outputs, _ = model(test_inputs, individual=test_individual)
                    test_loss = criterion(test_outputs, test_labels)
                    test_losses += test_loss.item() * test_inputs.size(0)

                    probabilities = F.softmax(test_outputs, dim=1)  # 形状: (batch_size, num_classes)
                    pred_labels = torch.argmax(probabilities, dim=1)  # 或直接用logits: torch.argmax(outputs, dim=1)
                    labels.extend(test_labels.cpu().tolist())
                    preds.extend(pred_labels.cpu().tolist())
                    total_samples += test_inputs.size(0)

            accuracy = accuracy_score(labels, preds)
            precision = precision_score(labels, preds, pos_label=1)
            recall = recall_score(labels, preds, pos_label=1)
            f1 = f1_score(labels, preds, pos_label=1)
            cm = confusion_matrix(labels, preds)

            print(f"train:{dataset_name};test:{dataset_name2}")
            print("混淆矩阵：")
            print(cm)
            print(f"Accuracy:{accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1: {f1:.4f}")

            with open(f'./main/resualt.log', 'a', encoding='utf-8') as f:
                f.write(f"train:{dataset_name};test:{dataset_name2}\n")
                f.write("混淆矩阵：\n")
                f.write(str(cm) + "\n")
                f.write(f"Accuracy:{accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1: {f1:.4f}\n\n")
