import torch
from torch import nn
from torch.nn import functional as F
from board import Board


class ConvBlock(nn.Module):
    """
    卷积块
    """
    def __init__(self, in_channels: int, out_channel: int, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channel,
                              kernel_size=kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        return F.relu(self.batch_norm(self.conv(x)))


class ResidueBlock(nn.Module):
    """
    残差块
    """
    def __init__(self, in_channels=128, out_channels=128):
        """
        初始化。
        :param in_channels: 输入通道数。
        :param out_channels: 输出通道数。
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        out = F.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))
        return F.relu(out) + x


class PolicyHead(nn.Module):
    """
    策略头
    """
    def __init__(self, in_channels=128, board_len=11):
        """
        初始化。
        :param in_channels: 输入通道数。
        :param board_len: 棋盘长度。
        """
        super().__init__()
        self.board_len = board_len
        self.in_channels = in_channels
        self.conv = ConvBlock(in_channels=in_channels, out_channel=1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return F.log_softmax(x, dim=1)


class ValueHead(nn.Module):
    """
    策略头。
    """

    def __init__(self, in_channels=128, board_len=11):
        """
        初始化。
        :param in_channels: 输入通道数。
        :param board_len: 棋盘大小。
        """
        super().__init__()
        self.in_channels = in_channels
        self.board_len = board_len
        self.conv = ConvBlock(in_channels=in_channels, out_channel=1, kernel_size=1, padding=0)
        self.fc = nn.Sequential(
            nn.Linear(board_len**2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        return x


class PolicyValueNet(nn.Module):
    """
    策略价值网络。
    """

    def __init__(self, board_len=11, n_feature_planes=2, num_block=20, num_channel=128, is_use_gpu=True):
        """
        初始化。
        :param board_len: 棋盘大小。
        :param n_feature_planes: 棋盘特征数。
        :param is_use_gpu: 是否使用GPU计算
        """
        super().__init__()
        self.board_len = board_len
        self.is_use_gpu = is_use_gpu
        self.n_feature_planes = n_feature_planes
        self.device = torch.device('cuda:0' if is_use_gpu else 'cpu')
        self.conv = ConvBlock(in_channels=n_feature_planes, out_channel=num_channel, kernel_size=3, padding=1)
        # 4个残差块
        self.residues = nn.Sequential(
            *[ResidueBlock(num_channel, num_channel) for i in range(num_block)])
        self.policy_head = PolicyHead(in_channels=num_channel, board_len=board_len)
        self.value_head = ValueHead(in_channels=num_channel, board_len=board_len)

    def forward(self, x):
        """
        前馈，输出p_hat和value
        :param x: Tensor -> shape:(N, C, H, W),棋局的状态特征平面张量
        :return:
                *p_hat: Tensor -> shape:(N, board_len^2),对数先验概率向量
                *value: Tensor -> shape:(N, 1),当前局面的估值
        """
        x = self.conv(x)
        x = self.residues(x)
        p_hat = self.policy_head(x)
        value = self.value_head(x)
        return p_hat, value

    def predict(self, board: Board):
        """
        获取当前局面上所有可用action和他对应的先验概率P(s, a),以及局面的value
        :param board: Board 棋盘。
        :return:
                *probs: np.ndarray -> shape: (len(chess_board.available_actions), )
                        当前局面上所有可用action对应的先验概率P(s, a)
                *value: float 当前局面的估值。
        """
        feature_planes = board.get_feature_planes().to(self.device)
        feature_planes.unsqueeze_(0)
        p_hat, value = self(feature_planes)
        if board.current_player == board.WHITE:
            p_hat = p_hat.reshape((1, 11, 11))
            p_hat = p_hat.transpose(1, 2)
        # 将对数概率转换为概率
        p = torch.exp(p_hat).flatten()
        # 只取可行的落点
        if self.is_use_gpu:
            p = p[board.available_action].cpu().detach().numpy()
        else:
            p = p[board.available_action].detach().numpy()
        return p, value[0].item()

    def set_device(self, is_use_gpu: bool):
        """
        设置神经网络运行设备
        :param is_use_gpu: 是否使用GPU。
        :return:
        """
        self.is_use_gpu = is_use_gpu
        self.device = torch.device('cuda:0' if is_use_gpu else 'cpu')
