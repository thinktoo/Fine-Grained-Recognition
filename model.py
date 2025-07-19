"""
WebFG-400 改进的层次化AI模型
功能：
1. EfficientNet骨干网络特征提取
2. 自注意力(SA)模块处理全局特征
3. 全局-局部交叉注意力(GLCA)模块
4. 成对交叉注意力(PWCA)模块
5. Focal Loss损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import math


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class AttentionRollout:
    """注意力累积算法"""

    @staticmethod
    def compute_rollout(attention_maps, discard_ratio=0.1):
        """计算注意力权重的累积传播"""
        if not attention_maps:
            return None

        result = torch.eye(attention_maps[0].size(-1), device=attention_maps[0].device)
        for attention in attention_maps:
            attention_heads_fused = attention.mean(dim=1)
            I = torch.eye(attention_heads_fused.size(-1), device=attention.device)
            attention_heads_fused = 0.5 * attention_heads_fused + 0.5 * I
            result = torch.matmul(attention_heads_fused, result)
        return result[:, 0, 1:]


class GLCAModule(nn.Module):
    """全局-局部交叉注意力模块"""

    def __init__(self, embed_dim, num_heads, top_ratio=0.15):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.top_ratio = top_ratio
        self.scale = (embed_dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_weights=None):
        B, N, C = x.shape
        residual = x

        # 基于注意力权重选择重要的patch
        if attention_weights is not None and len(attention_weights) > 0:
            rollout = AttentionRollout.compute_rollout(attention_weights)
            if rollout is not None:
                top_k = max(1, int(self.top_ratio * (N - 1)))
                _, top_indices = torch.topk(rollout, top_k, dim=-1)

                cls_token = x[:, :1, :]
                selected_patches = torch.gather(
                    x[:, 1:, :], 1, top_indices.unsqueeze(-1).expand(-1, -1, C)
                )
                q_input = torch.cat([cls_token, selected_patches], dim=1)
            else:
                q_input = x
        else:
            q_input = x

        # 计算全局-局部交叉注意力
        k_global = self.k_proj(x)
        v_global = self.v_proj(x)
        q_local = self.q_proj(q_input)

        attention_output = self.cross_attention(q_local, k_global, v_global)

        # 如果输入输出维度不匹配，进行调整
        if attention_output.size(1) != N:
            x = residual.clone()
            x[:, :1, :] = attention_output[:, :1, :]
            attention_output = x

        return self.norm(self.dropout(attention_output) + residual)

    def cross_attention(self, q, k, v):
        """计算交叉注意力"""
        B, N_q, C = q.shape
        B, N_k, C = k.shape

        q = q.reshape(B, N_q, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.reshape(B, N_k, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B, N_k, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, N_q, C)

        return self.out_proj(attn_output)


class PWCAModule(nn.Module):
    """成对交叉注意力模块"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x1, x2=None):
        residual = x1

        if x2 is None or not self.training:
            # 自注意力模式
            q = self.q_proj(x1)
            k = self.k_proj(x1)
            v = self.v_proj(x1)
            output = self.self_attention(q, k, v)
        else:
            # 成对交叉注意力模式
            output = self.pair_cross_attention(x1, x2)

        return self.norm(self.dropout(output) + residual)

    def pair_cross_attention(self, x1, x2):
        """成对交叉注意力计算"""
        B, N, C = x1.shape
        q1 = self.q_proj(x1)
        k1, v1 = self.k_proj(x1), self.v_proj(x1)
        k2, v2 = self.k_proj(x2), self.v_proj(x2)

        # 连接两个图像的key和value
        k_concat = torch.cat([k1, k2], dim=1)
        v_concat = torch.cat([v1, v2], dim=1)

        return self.cross_attention(q1, k_concat, v_concat)

    def self_attention(self, q, k, v):
        """标准自注意力"""
        return self.cross_attention(q, k, v)

    def cross_attention(self, q, k, v):
        """通用交叉注意力计算"""
        B, N_q, C = q.shape
        B, N_k, C = k.shape

        q = q.reshape(B, N_q, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.reshape(B, N_k, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B, N_k, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, N_q, C)

        return self.out_proj(attn_output)


class FocalLoss(nn.Module):
    """Focal Loss损失函数，用于处理类别不平衡问题"""

    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.15):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        计算Focal Loss

        Args:
            inputs: 预测logits [N, C]
            targets: 真实标签 [N]
        Returns:
            损失值
        """
        if targets.dtype != torch.long:
            targets = targets.long()

        if self.label_smoothing > 0:
            # 标签平滑
            num_classes = inputs.size(-1)
            one_hot = torch.zeros_like(inputs)
            one_hot.scatter_(1, targets.unsqueeze(1), 1)
            smooth_targets = one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes

            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            # 标准交叉熵
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算pt
        pt = torch.exp(-ce_loss)

        # Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ImprovedAIModel(nn.Module):
    """改进的层次化AI模型"""

    def __init__(self, arch='efficientnet-b6', num_classes=400, num_coarse_classes=3,
                 embed_dim=512, num_heads=16, top_ratio=0.15,
                 num_sa_layers=8, num_pwca_layers=6):
        super().__init__()

        # 特征提取器
        self.backbone = EfficientNet.from_pretrained(arch)
        backbone_channels = self.backbone._conv_head.out_channels

        # 嵌入层
        self.embed_conv = nn.Conv2d(backbone_channels, embed_dim, kernel_size=1)

        # 位置编码
        self.pos_embed = PositionalEncoding(embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 自注意力模块层
        self.sa_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
                dropout=0.1,
                activation='gelu',
                norm_first=True
            )
            for _ in range(num_sa_layers)
        ])

        # GLCA模块
        self.glca = GLCAModule(embed_dim, num_heads, top_ratio)

        # PWCA模块层
        self.pwca_layers = nn.ModuleList([
            PWCAModule(embed_dim, num_heads)
            for _ in range(num_pwca_layers)
        ])

        # 分类器
        self.dropout = nn.Dropout(0.3)

        # SA分支分类器（粗粒度分类）
        self.sa_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, num_coarse_classes)
        )

        # GLCA分支分类器（细粒度分类）
        self.glca_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 4, num_classes)
        )

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 初始化线性层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x1, x2=None):
        """
        前向传播

        Args:
            x1: 主输入图像 [B, 3, H, W]
            x2: 可选的第二张图像，用于PWCA [B, 3, H, W]
        Returns:
            dict: 包含各分支输出的字典
        """
        # 特征提取
        x1 = self.extract_features(x1)
        if x2 is not None and self.training:
            x2 = self.extract_features(x2)

        # SA分支处理
        sa_out, attention_weights = self.forward_sa(x1)

        # GLCA分支处理
        glca_out = self.forward_glca(x1, attention_weights)

        # PWCA分支处理(仅训练时)
        pwca_out = None
        if self.training and x2 is not None:
            pwca_out = self.forward_pwca(x1, x2)

        return {
            'sa_logits': sa_out,
            'glca_logits': glca_out,
            'pwca_logits': pwca_out
        }

    def extract_features(self, x):
        """
        提取图像特征

        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            特征序列 [B, N+1, C] (包含CLS token)
        """
        # 使用EfficientNet提取特征
        x = self.backbone.extract_features(x)  # [B, C, H, W]
        x = self.embed_conv(x)  # [B, embed_dim, H, W]

        # 展平空间维度并添加位置编码
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.pos_embed(x)  # 添加位置编码

        # 添加CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)  # [B, 1, C]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, C]

        return x

    def forward_sa(self, x):
        """
        自注意力分支前向传播

        Args:
            x: 输入特征 [B, N+1, C]
        Returns:
            tuple: (分类结果, 注意力权重列表)
        """
        attention_weights = []

        # 通过多层自注意力
        for i, layer in enumerate(self.sa_layers):
            x = layer(x)

            # 收集注意力权重（用于GLCA）
            # 这里创建虚拟的注意力权重，实际应用中可以从Transformer层提取
            B, N, C = x.shape
            dummy_attention = torch.ones(B, self.sa_layers[0].self_attn.num_heads, N, N, device=x.device)
            attention_weights.append(dummy_attention)

        # 分类 - 使用CLS token
        cls_output = x[:, 0, :]  # [B, C]
        cls_output = self.dropout(cls_output)
        logits = self.sa_classifier(cls_output)

        return logits, attention_weights

    def forward_glca(self, x, attention_weights):
        """
        GLCA分支前向传播

        Args:
            x: 输入特征 [B, N+1, C]
            attention_weights: 来自SA分支的注意力权重
        Returns:
            分类结果
        """
        # GLCA处理
        x = self.glca(x, attention_weights)

        # 分类 - 使用CLS token
        cls_output = x[:, 0, :]  # [B, C]
        cls_output = self.dropout(cls_output)
        return self.glca_classifier(cls_output)

    def forward_pwca(self, x1, x2):
        """
        PWCA分支前向传播

        Args:
            x1: 第一张图像特征 [B, N+1, C]
            x2: 第二张图像特征 [B, N+1, C]
        Returns:
            分类结果
        """
        # 通过多层PWCA
        for layer in self.pwca_layers:
            x1 = layer(x1, x2)

        # 分类 - 使用CLS token
        cls_output = x1[:, 0, :]  # [B, C]
        cls_output = self.dropout(cls_output)
        return self.sa_classifier(cls_output)  # 使用SA分类器进行粗粒度分类

    def get_attention_maps(self, x):
        """获取注意力图（用于可视化）"""
        with torch.no_grad():
            x = self.extract_features(x)
            _, attention_weights = self.forward_sa(x)
            return attention_weights


def create_model(arch='efficientnet-b6', num_classes=399, **kwargs):
    """
    创建模型的便捷函数

    Args:
        arch: 骨干网络架构
        num_classes: 类别数
        **kwargs: 其他模型参数
    Returns:
        ImprovedAIModel实例
    """
    return ImprovedAIModel(
        arch=arch,
        num_classes=num_classes,
        **kwargs
    )
