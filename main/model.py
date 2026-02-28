import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from main.models.resnet1d import resnet18_1d, resnet34_1d, resnet50_1d

class ResLSTMModel(nn.Module):
    def __init__(self, input_channels, resnet_depth, lstm_input_size, lstm_hidden_size, lstm_num_layers, output_size):
        super(ResLSTMModel, self).__init__()
        if resnet_depth == 18:
            self.resnet = resnet18_1d(input_channels=input_channels)
        elif resnet_depth == 34:
            self.resnet = resnet34_1d(input_channels=input_channels)
        elif resnet_depth == 50:
            self.resnet = resnet50_1d(input_channels=input_channels)
        else:
            raise ValueError("Unsupported ResNet depth. Choose from 18, 34, or 50.")
        self.resnet_outpusize = 512
        self.linear = nn.Linear(self.resnet_outpusize, lstm_input_size)
        self.bn0 = nn.BatchNorm1d(lstm_input_size)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False
        )
        self._init_lstm_weights()
        self.bn1 = nn.BatchNorm1d(lstm_hidden_size)
        self.fc1 = nn.Linear(lstm_hidden_size, 128)  # 中间层增加维度
        self.relu = nn.ReLU()  # 引入非线性激活
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_size)  # 最终输出
        self.dropout = nn.Dropout(0.5)

    def _init_lstm_weights(self):
        """自定义LSTM权重初始化，避免偏向正向激活"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # 输入到隐藏层的权重
                nn.init.xavier_uniform_(param.data)  # Xavier初始化，适合激活函数为tanh/sigmoid
            elif 'weight_hh' in name:  # 隐藏层到隐藏层的权重
                nn.init.orthogonal_(param.data)  # 正交初始化，减少特征冗余
            elif 'bias' in name:  # 偏置项（控制门控的阈值）
                # 偏置初始化为较小值，避免过度抑制某一方向（正/负）
                param.data.fill_(0.01)  # 替代默认的0初始化，轻微正向但幅度小

    def finetune(self):
        """冻结LSTM之前的所有参数（ResNet、线性层和BatchNorm层）"""
        # 冻结ResNet部分参数
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # 冻结LSTM前的线性层参数
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # 冻结LSTM前的BatchNorm层参数
        for param in self.bn0.parameters():
            param.requires_grad = False

    def forward(self, x, return_intermediates=False):
        x = x.view(x.size(0), x.size(1), -1)  # (batch, seq_len, channels)
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = self.resnet(x)  # (batch, output_channels, seq_len)
        tmp_out = x
        x = x.permute(0, 2, 1)  # (batch, seq_len, output_channels)
        x = self.linear(x)  # (batch, seq_len, lstm_input_size)
        x = self.relu(x)
        batch_size, seq_len, feat_size = x.size()
        x = self.bn0(x.contiguous().view(-1, feat_size)).view(batch_size, seq_len, feat_size)
        x, (lstm_hidden, lstm_cell) = self.lstm(x)  # (batch, seq_len, output_channels * 2)
        x = x[:, -1, :]
        x = self.bn1(x)
        x = self.fc1(x)  # (batch, output_size)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn2(x)
        x = self.fc2(x)  # (batch, output_size)
        if not return_intermediates:
            return x
        return x, tmp_out, (lstm_hidden, lstm_cell)
    

class pretrainModel(nn.Module):
    def __init__(self, input_channels, resnet_depth, hidden_size, output_size):
        super(pretrainModel, self).__init__()
        if resnet_depth == 18:
            self.resnet = resnet18_1d(input_channels=input_channels)
        elif resnet_depth == 34:
            self.resnet = resnet34_1d(input_channels=input_channels)
        elif resnet_depth == 50:
            self.resnet = resnet50_1d(input_channels=input_channels)
        else:
            raise ValueError("Unsupported ResNet depth. Choose from 18, 34, or 50.")
        self.resnet_outpusize = 512
        self.linear = nn.Linear(self.resnet_outpusize, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 128)  # 中间层增加维度
        self.relu = nn.ReLU()  # 引入非线性激活
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_size)  # 最终输出
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, return_intermediates=False):
        x = x.float()
        x = x.view(x.size(0), x.size(1), -1)  # (batch, seq_len, channels)
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = self.resnet(x)  # (batch, output_channels, seq_len)
        tmp_out = x
        x = x.permute(0, 2, 1)  # (batch, seq_len, output_channels)
        x = self.linear(x)  # (batch, seq_len, lstm_input_size)
        x = self.relu(x)
        x = torch.mean(x, dim=1)
        x = self.bn1(x)
        x = self.fc1(x)  # (batch, output_size)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn2(x)
        x = self.fc2(x)  # (batch, output_size)
        if not return_intermediates:
            return x
        return x, tmp_out
    

class finetuneModel(nn.Module):
    def __init__(self, input_channels, resnet_depth, expert_num, lstm_input_size, lstm_hidden_size, lstm_num_layers, output_size, alpha = 0.2):
        super(finetuneModel, self).__init__()
        # 1. 特征提取层（保持不变）
        if resnet_depth == 18:
            self.resnet = resnet18_1d(input_channels=input_channels)
        elif resnet_depth == 34:
            self.resnet = resnet34_1d(input_channels=input_channels)
        elif resnet_depth == 50:
            self.resnet = resnet50_1d(input_channels=input_channels)
        else:
            raise ValueError("Unsupported ResNet depth. Choose from 18, 34, or 50.")
        self.alpha = alpha
        self.individual_size = 12
        self.resnet_outpusize = 512
        self.linear = nn.Linear(self.resnet_outpusize, lstm_input_size)
        self.bn0 = nn.BatchNorm1d(lstm_input_size)
        
        # 2. 核心改动：创建多个LSTM专家（替换原单个LSTM）
        self.lstm_experts = nn.ModuleList([  # 用ModuleList管理多个LSTM，确保参数被正确注册
            nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                batch_first=True,
                bidirectional=False
            ) for _ in range(expert_num)
        ])
        self._init_lstm_experts_weights()  # 初始化所有LSTM专家的权重

        self.fusion_weights_individual = nn.Linear(self.individual_size, expert_num)
        self.fusion_weights_feature = nn.Linear(lstm_input_size, expert_num)

        self.lstm_hidden_size = lstm_hidden_size
        # 3. 后续全连接层（保持不变，适配融合后的维度）
        self.bn1 = nn.BatchNorm1d(lstm_hidden_size)
        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_size)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout5 = nn.Dropout(0.5)

    def _init_lstm_experts_weights(self):
        """初始化所有LSTM专家的权重（适配多专家场景）"""
        for lstm in self.lstm_experts:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.0)
                    # LSTM偏置的forget gate设为1（经典初始化）
                    n = param.size(0)
                    param.data[n//4 : n//2].fill_(1.0)

    def finetune(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def moe_balance_loss(self,expert_weights):
        """
        计算MOE专家负载均衡Loss（方差形式，高效）
        :param expert_weights: 形状 [B, N]，B=批次大小，N=专家数量
        :return: 标量，均衡Loss
        """
        # 步骤1：计算每个专家在批次内的平均激活值 [N]
        expert_avg = torch.mean(expert_weights, dim=0)
        # 步骤2：计算方差（方差越小，负载越均衡）
        expert_var = torch.var(expert_avg, unbiased=False)
        return expert_var

    def feature_expert_consist_loss(self, special_features, expert_params, sample_pairs_num=None):
        """
       特征-参数一致性Loss（随机采样样本对，控制计算量）
        :param special_features: 形状 [B, D]，B=批次大小，D=特殊特征的维度
        :param expert_params: 形状 [B, N]，B=批次大小，N=专家参数/激活权重的维度
        :param sample_pairs_num: 采样的样本对数量，默认为批次大小的2倍（平衡效果与计算量）
        :return: 标量，一致性Loss
        """
        B, _ = special_features.shape
        sample_pairs_num = sample_pairs_num or min(2 * B, 1024)  # 限制最大采样数，避免批次过大溢出
        
        # 步骤1：随机生成采样的样本对索引（b1, b2）
        b1 = torch.randint(0, B, (sample_pairs_num,), device=special_features.device)
        b2 = torch.randint(0, B, (sample_pairs_num,), device=special_features.device)
        
        # 步骤2：提取对应样本对的特征和专家参数
        f1, f2 = special_features[b1], special_features[b2]
        p1, p2 = expert_params[b1], expert_params[b2]
        
        # 步骤3：计算余弦相似度（归一化后计算，避免数值规模影响）
        # 特征余弦相似度 [sample_pairs_num]
        sim_f = F.cosine_similarity(F.normalize(f1, dim=1), F.normalize(f2, dim=1), dim=1)
        # 专家参数余弦相似度 [sample_pairs_num]
        sim_p = F.cosine_similarity(F.normalize(p1, dim=1), F.normalize(p2, dim=1), dim=1)
        
        # 步骤4：计算MSE损失（让两种相似度尽可能接近）
        consist_loss = F.mse_loss(sim_f, sim_p)
        
        return consist_loss

    def forward(self, x, individual=None, return_intermediates=False):
        x = x.float()
        individual = individual.float()
        # 特征提取阶段（保持不变）
        x = x.view(x.size(0), x.size(1), -1)  # (batch, seq_len, channels)
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = self.resnet(x)  # (batch, output_channels, seq_len)
        tmp_out = x
        x = x.permute(0, 2, 1)  # (batch, seq_len, output_channels)
        x = self.linear(x)  # (batch, seq_len, lstm_input_size)
        x = self.relu(x)
        batch_size, seq_len, feat_size = x.size()
        x = self.bn0(x.contiguous().view(-1, feat_size)).view(batch_size, seq_len, feat_size)
        x = self.dropout2(x)

        # 4. 核心改动：多LSTM专家前向传播 + 输出融合
        expert_outputs = []
        expert_hiddens = []
        expert_cells = []
        for lstm in self.lstm_experts:
            lstm_out, (lstm_hidden, lstm_cell) = lstm(x)  # 每个专家处理相同输入
            # 取最后一个时间步输出（和原逻辑一致）
            expert_out_last = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)
            expert_outputs.append(expert_out_last)
            expert_hiddens.append(lstm_hidden)
            expert_cells.append(lstm_cell)
        
        # 融合策略1：平均融合（简单且稳定，推荐默认使用）
        # x = torch.stack(expert_outputs, dim=1)  # (batch, expert_num, lstm_hidden_size)
        # x = torch.mean(x, dim=1)  # (batch, lstm_hidden_size)
        
        # 【可选】融合策略2：门控加权融合（更灵活，需新增融合层）
        # 若需要加权融合，替换上面2行代码为以下内容：
        individual_weights = self.fusion_weights_individual(individual)  # (batch, expert_num)
        feature_weights = self.fusion_weights_feature(torch.mean(x, dim=1))  # (batch, expert_num)
        fusion_weights = (1-self.alpha) * individual_weights + self.alpha * feature_weights
        weights = torch.softmax(fusion_weights, dim=1)  # (batch, expert_num)
        x = torch.bmm(weights.unsqueeze(1), torch.stack(expert_outputs, dim=1)).squeeze(1)  # (batch, lstm_hidden_size)
        moe_load_balance_loss = self.moe_balance_loss(weights)
        feature_expert_consist_loss = self.feature_expert_consist_loss(
            special_features=individual,
            expert_params=weights
        )

        # 后续全连接层（保持不变）
        x = self.dropout2(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout5(x)
        x = self.bn2(x)
        x = self.fc2(x)

        if not return_intermediates:
            return x, (moe_load_balance_loss, feature_expert_consist_loss)
        # 返回中间结果时，补充所有专家的隐状态/细胞状态（可选）
        return x, tmp_out, (expert_hiddens, expert_cells, moe_load_balance_loss, feature_expert_consist_loss)