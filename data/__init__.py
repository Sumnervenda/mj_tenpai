"""数据流水线：对局记录解析 → 训练样本提取 → PyTorch Dataset。"""

from .data_generator import HeuristicAgent
from .record_parser import JSONLRecordParser, TrainingSample
from .mjson_parser import MJSONRecordParser, mjson_str_to_type, type_to_mjson_str
from .dataset import MahjongStateActionDataset, MJSONIterableDataset
