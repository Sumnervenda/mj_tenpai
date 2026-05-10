"""日麻 AI 训练模块：SL 预训练 + PPO 自对弈 + 奖励塑形。"""

from .agents import (
    Agent, ResNetAgent, TransformerAgent, OracleTeacherAgent, build_agent,
)
# 中文注释：训练包公共导出入口。
