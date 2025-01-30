import torch
import math
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('encoding', self._get_timing_signal(max_len, d_model))

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

    def _get_timing_signal(self, length, channels):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2) * -(math.log(10000.0) / channels))
        pe = torch.zeros(length, channels)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_len=5000):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True  # ここを追加
        )

    def forward(self, src, tgt):
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt)
        return output

class AIMP(nn.Module):
    def __init__(self, pre_feas_dim, hidden, n_transformer, dropout):
        super(AIMP, self).__init__()

        # バッチ正規化層（pre_feasのみ）
        self.bn = nn.BatchNorm1d(pre_feas_dim)

        # 前処理用の埋め込み層
        self.pre_embedding = nn.Sequential(
            nn.Conv1d(pre_feas_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        # 埋め込み層（pre_feasのみ）
        self.embedding = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        # Transformer層
        self.n_transformer = n_transformer
        self.transformer = TransformerModel(
            d_model=hidden,
            nhead=4,
            num_layers=self.n_transformer,
            dim_feedforward=2048
        )

        # Transformer後の活性化層
        self.transformer_act = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        # Transformerの残差接続層
        self.transformer_res = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1),  # concat後のチャネル数をhidden * 2に変更
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        # プーリング層
        self.transformer_pool = nn.AdaptiveAvgPool2d((1, None))

        # 分類層
        self.clf = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),       # 出力ユニットを1に変更
            nn.Sigmoid(),               # シグモイド関数を追加
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.pre_embedding, self.embedding, self.clf]:
            for sublayer in layer:
                if isinstance(sublayer, (nn.Conv1d, nn.Linear)):
                    nn.init.xavier_uniform_(sublayer.weight)
        for layer in [self.transformer_act, self.transformer_res]:
            for sublayer in layer:
                if isinstance(sublayer, (nn.Conv1d, nn.Linear)):
                    nn.init.xavier_uniform_(sublayer.weight)

    def forward(self, pre_feas):
        # バッチ正規化
        pre_feas = self.bn(pre_feas.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, seq_length, pre_feas_dim]

        # 前処理用埋め込み
        pre_feas = self.pre_embedding(pre_feas.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, seq_length, hidden]

        # 埋め込み層
        feas_em = self.embedding(pre_feas.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, seq_length, hidden]

        # Transformerの処理（自己注意）
        transformer_out = self.transformer(feas_em, feas_em)  # [batch_size, seq_length, hidden]

        # Transformer後の活性化
        transformer_out = self.transformer_act(transformer_out.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, seq_length, hidden]

        # 残差接続
        transformer_out = self.transformer_res(
            torch.cat([transformer_out, feas_em], dim=-1).permute(0, 2, 1)
        ).permute(0, 2, 1)  # [batch_size, seq_length, hidden]

        # プーリング
        transformer_out = self.transformer_pool(transformer_out).squeeze(1)  # [batch_size, hidden]

        # 分類
        out = self.clf(transformer_out)  # [batch_size, 1]
        # out = torch.nn.functional.softmax(out, dim=-1)  # ソフトマックスは不要

        return out  # [batch_size, 1]
