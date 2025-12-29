import torch
from torch import nn
import torch.nn.functional as F
import pywt
import numpy as np
import math
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
        self.linear = nn.Linear(self.resnet_outpusize, lstm_hidden_size)
        # self.bn0 = nn.BatchNorm1d(lstm_input_size)
        # self.lstm = nn.LSTM(
        #     input_size=lstm_input_size,
        #     hidden_size=lstm_hidden_size,
        #     num_layers=lstm_num_layers,
        #     batch_first=True,
        #     bidirectional=False
        # )
        # self._init_lstm_weights()
        self.bn1 = nn.BatchNorm1d(lstm_hidden_size)
        self.fc1 = nn.Linear(lstm_hidden_size, 128)  # 中间层增加维度
        self.relu = nn.ReLU()  # 引入非线性激活
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_size)  # 最终输出
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, return_intermediates=False):
        x = x.view(x.size(0), x.size(1), -1)  # (batch, seq_len, channels)
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = self.resnet(x)  # (batch, output_channels, seq_len)
        tmp_out = x
        x = x.permute(0, 2, 1)  # (batch, seq_len, output_channels)
        x = self.linear(x)  # (batch, seq_len, lstm_input_size)
        x = self.relu(x)
        batch_size, seq_len, feat_size = x.size()
        x = self.bn1(x.contiguous().view(-1, feat_size)).view(batch_size, seq_len, feat_size)
        x = self.fc1(x)  # (batch, output_size)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn2(x)
        x = self.fc2(x)  # (batch, output_size)
        if not return_intermediates:
            return x
        return x, tmp_out