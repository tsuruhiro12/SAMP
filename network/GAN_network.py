import torch
import torch.nn as nn
# Generatorクラスの定義
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 出力は出力次元数に合わせて調整
        return self.sigmoid(x)
# Discriminatorクラスの定義
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 出力は1ノード（本物 or 偽）
        return self.sigmoid(x)
# GANクラスの定義
class GAN(nn.Module):
    def __init__(self, features, lstm_hidden_size):
        super(GAN, self).__init__()
        # Generatorの出力をDiscriminatorの入力に合わせるためにlstm_hidden_sizeを指定
        self.generator = Generator(input_dim=features, output_dim=lstm_hidden_size)
        self.discriminator = Discriminator(input_dim=lstm_hidden_size)
    def forward(self, emb_mat):
        generated_data = self.generator(emb_mat)  # [バッチサイズ, lstm_hidden_size] の形状に生成
        discriminator_output = self.discriminator(generated_data)  # [バッチサイズ, 1] の出力
        return  discriminator_output[:,-1]