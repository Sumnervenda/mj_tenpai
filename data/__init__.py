"""数据流水线：对局记录解析 → 训练样本提取 → PyTorch Dataset。"""

from .data_generator import HeuristicAgent
from .record_parser import JSONLRecordParser, TrainingSample
from .mjson_parser import MJSONRecordParser, mjson_str_to_type, type_to_mjson_str
from .dataset import (
    MahjongStateActionDataset, MJSONIterableDataset, MJSONTokenIterableDataset,
    MJSONPublicPrivateTokenIterableDataset,
    TensorShardBatchDataset, TokenShardBatchDataset, TokenMmapShardBatchDataset,
    TokenDataset, collate_transformer_batch,
)
# 中文注释：数据包公共导出入口，集中暴露解析器和数据集类型。
