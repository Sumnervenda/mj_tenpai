# 服务器训练 Runbook

这份文档只关心一件事：把服务器上的环境配置、数据准备、benchmark、训练、续训流程固定下来，减少临时排错。

## 推荐流程

```bash
git clone git@github.com:Sumnervenda/mj_tenpai.git
cd mj_tenpai
chmod +x setup_server.sh run_benchmark.sh run_train.sh run_resume.sh
./setup_server.sh
```

上传数据集时，优先打包成一个大文件再传，避免几十万小文件让 `scp` 或网盘拖慢：

```bash
# local
tar -cf dataset_2021_2026.tar dataset/datasets_years/202[1-6]*
scp dataset_2021_2026.tar user@server:~/mj_tenpai/

# server
tar -xf dataset_2021_2026.tar
```

构建 Transformer token cache。这个步骤只需要做一次；训练阶段会直接读 token shard，不再解析 mjson。

```bash
python -m training.mjson_token_cache build \
  --source dataset/datasets_years \
  --cache dataset/token_cache_2021-2026 \
  --years 2021,2022,2023,2024,2025,2026 \
  --num_workers 16 \
  --shard_size 10000

python -m training.mjson_token_cache info --cache dataset/token_cache_2021-2026
```

如果上一次 cache 构建中断，直接覆盖重建：

```bash
python -m training.mjson_token_cache build \
  --source dataset/datasets_years \
  --cache dataset/token_cache_2021-2026 \
  --years 2021,2022,2023,2024,2025,2026 \
  --num_workers 16 \
  --shard_size 10000 \
  --overwrite
```

跑 benchmark 后再开始长训：

```bash
./run_benchmark.sh
./run_train.sh
```

断点续训：

```bash
./run_resume.sh
# or
./run_resume.sh checkpoints/transformer_server/sl_resume.pt
```

## 常用环境变量

这些变量可以直接写在命令前，不需要改脚本：

```bash
BATCH_SIZE=512 EPOCHS=20 ./run_train.sh
USE_WANDB=0 ./run_train.sh
DATA_MODE=stream_mjson ./run_train.sh
AUTO_BUILD_TOKEN_CACHE=1 ./run_train.sh
AUTO_BUILD_TOKEN_CACHE=1 CACHE_OVERWRITE=1 ./run_train.sh
BENCHMARK_SIZES=128,256 BENCHMARK_BATCHES=200 ./run_benchmark.sh
RUN_TESTS=0 ./setup_server.sh
TORCH_CUDA_TAG=cu124 ./setup_server.sh
ALLOW_CPU=1 ./setup_server.sh
```

关键变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DATA_DIR` | `dataset/datasets_years` | mjson 年份目录 |
| `TOKEN_CACHE_DIR` | `dataset/token_cache_2021-2026` | token cache 输出/读取目录 |
| `DATA_MODE` | `token_cache` | `token_cache` 或 `stream_mjson` |
| `AUTO_BUILD_TOKEN_CACHE` | `0` | cache 不存在时是否自动构建 |
| `CACHE_OVERWRITE` | `0` | 自动构建 cache 时是否覆盖旧 cache |
| `YEARS` | `2021,2022,2023,2024,2025,2026` | 使用的年份 |
| `BATCH_SIZE` | `256` | 训练 batch size |
| `SAVE_INTERVAL_MIN` | `30` | step-level checkpoint 间隔 |
| `USE_WANDB` | `1` | 是否启用 W&B |
| `BENCHMARK_OUTPUT` | 空 | benchmark JSON 输出路径 |

## 常见故障

`torch.cuda.is_available() 为 False`：优先检查 `nvidia-smi` 是否正常。如果 `nvidia-smi` 可见但 PyTorch 不认 CUDA，通常是 CUDA wheel 和 driver 不匹配。可以用 `TORCH_CUDA_TAG=cu124 ./setup_server.sh` 明确指定。

`token cache manifest is incomplete`：cache 构建中断过，manifest 只写了一部分。用 `--overwrite` 或 `CACHE_OVERWRITE=1` 重建。

`run_benchmark.sh checkpoint not found`：脚本现在会直接失败，不会静默改用随机模型。确认路径，或者不传参数来测随机初始化模型。

训练开始前就退出并提示 `token cache 未完成`：这是预期行为。默认训练走 `token_cache`，目的是避免长训时反复解析 mjson。临时测试可用 `DATA_MODE=stream_mjson ./run_train.sh`。

W&B 登录失败：不想联网就用 `USE_WANDB=0 ./run_train.sh`；想离线记录就设置 `WANDB_MODE=offline`，训练后再 `wandb sync wandb/offline-run-*`。

## 快速体检命令

```bash
bash -n setup_server.sh run_train.sh run_resume.sh run_benchmark.sh
python -m training.benchmark --device cpu --batches 1 --sizes 1 --no_compile --output none
python -m pytest tests -q -p no:cacheprovider
```
