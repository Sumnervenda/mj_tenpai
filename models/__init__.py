"""日麻 AI 模型：ResNet 基线 + Transformer + MTL 多任务网络。"""

from .feature_encoder import StateFeatureEncoder
from .resnet1d import ResidualBlock1D, ResNet1DBackbone
from .policy_value_net import MahjongPolicyValueNet
from .model_io import (
    save_checkpoint, load_checkpoint, load_checkpoint_metadata,
    save_resume_checkpoint, load_resume_checkpoint,
)

# Transformer 模块
from .tokenizer import MahjongTokenizer, TokenVocab, TokenType, Token, TokenSequence
from .transformer_backbone import TransformerBlock, TransformerBackbone
from .multi_task_heads import (
    MultiTaskHeads, ShantenHead, EfficiencyHead, DangerHead,
    UkeireHead, ScoreHead, PolicyHead, ValueHead, OracleValueHead,
)
from .transformer_policy_value import TransformerPolicyValueNet
# 中文注释：模型包公共导出入口，集中暴露网络结构和 checkpoint 工具。
