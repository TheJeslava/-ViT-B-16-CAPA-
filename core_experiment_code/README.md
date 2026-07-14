# 核心实验代码说明

本目录保存 CAPA / Para-CVPT 相关实验的核心可复用代码。当前包只包含训练、数据、模型和顺序运行管线所需文件，不包含临时脚本、报告生成脚本、画图脚本、实验输出、缓存文件和原始数据。

## 目录结构

```text
core_experiment_code/
├── README.md
├── pipeline_config_template.yaml
├── run_all_experiments.py
└── src/
    ├── __init__.py
    ├── data.py
    ├── modeling.py
    ├── cvpt_expert_adapter.py
    └── train.py
```

## 文件说明

- `pipeline_config_template.yaml`：完整实验配置模板，包含数据集、训练协议、方法列表、路由设置、输出规则和消融配置。
- `run_all_experiments.py`：顺序运行多个实验的入口脚本。每个实验结束后会读取/写入 `summary.json`，并持续更新累计表 `outputs/tables/full_results.csv` 和运行清单 `outputs/tables/full_run_manifest.jsonl`。
- `src/train.py`：单个实验的训练、验证、测试入口，负责日志、指标、checkpoint、路由统计和可视化预览的保存。
- `src/data.py`：数据集读取和 dataloader 构建，包括 CIFAR-100、VTAB-1K CIFAR-100、CUB-200-2011、Oxford Flowers-102、VTAB-1K Flowers-102、sNORB-Azim、sNORB-Elev 和调试用 FakeData。
- `src/modeling.py`：ViT-B/16 主模型和常规 PEFT 方法实现，包括 Linear Probe、Full Fine-tuning、VPT、CVPT、Para-CVPT、AdaptFormer、SSF 和 LoRA。
- `src/cvpt_expert_adapter.py`：Expert Adapter Pool 与 CAPA 实现。CAPA 在代码中对应方法名 `cvpt_expert_adapter`。

## 方法名称

配置文件中的 `methods.main_table` 默认包含 10 个主实验方法：

1. `linear_probe`
2. `full_finetune`
3. `adaptformer`
4. `ssf`
5. `lora`
6. `vpt`
7. `cvpt`
8. `para_cvpt`
9. `cvpt_expert_adapter`
10. `expert_adapter_pool`

其中 `cvpt_expert_adapter` 即本文新方案 CAPA，结构为 CVPT cross-attention prompt 分支加 top-k 专家 adapter 池；`expert_adapter_pool` 是只保留专家 adapter 池的结构对照。

## 使用的数据集

本实验使用的数据集是：
- `vtab1k_cifar100`
- `cub_200_2011`
- `oxford_flowers_102`

## 数据目录约定

默认数据根目录为 `data/`，也可以通过命令行参数 `--data-root` 覆盖。常用目录如下：

- CIFAR-100：`data/cifar-100-python`
- CUB-200-2011：`data/CUB_200_2011` 或 `data/CUB/CUB_200_2011`
- Oxford Flowers-102：torchvision `Flowers102` 所需目录，默认位于 `data/flowers-102`
- smallNORB：整理后的 smallNORB 原始 gzip 文件目录，用于 `snorb_azim` 和 `snorb_elev`

CIFAR-100 和 Flowers102 可使用 `--download` 允许 torchvision 下载。CUB 和 smallNORB 需要提前手动放好原始数据。

## 环境依赖

代码运行依赖 Python、PyTorch 和 torchvision。最小依赖可按以下方式准备：

```bash
pip install torch torchvision numpy matplotlib pyyaml pillow scipy
```

建议使用带 CUDA 的 PyTorch 版本。配置模板中默认启用 AMP、TF32、cudnn benchmark 和较高 matmul precision，以便在 GPU 上加速训练。

## 单个实验运行

当前包内没有单独的 `configs/` 目录，直接使用根目录下的配置模板即可：

```bash
python src/train.py \
  --config pipeline_config_template.yaml \
  --dataset vtab1k_cifar100 \
  --method cvpt_expert_adapter \
  --data-root data \
  --output-root outputs
```

如需快速检查管线是否可运行，可使用调试数据集：

```bash
python src/train.py \
  --config pipeline_config_template.yaml \
  --dataset debug_fake \
  --method linear_probe \
  --output-root outputs_debug \
  --epochs 1 \
  --batch-size 4
```

## 顺序运行多组实验

运行最终报告三数据集上的 10 个主方法：

```bash
python run_all_experiments.py \
  --config pipeline_config_template.yaml \
  --datasets vtab1k_cifar100 cub_200_2011 oxford_flowers_102 \
  --methods linear_probe full_finetune adaptformer ssf lora vpt cvpt para_cvpt cvpt_expert_adapter expert_adapter_pool \
  --data-root data \
  --output-root outputs \
  --keep-going
```

只查看将要运行的顺序，不实际训练：

```bash
python run_all_experiments.py \
  --config pipeline_config_template.yaml \
  --datasets vtab1k_cifar100 cub_200_2011 oxford_flowers_102 \
  --data-root data \
  --output-root outputs \
  --dry-run
```

脚本会按数据集和方法顺序逐个运行。若某个 run 的 `summary.json` 已存在，默认跳过；使用 `--force` 可强制重跑。使用 `--keep-going` 可在某组失败后继续后续实验。

## 输出内容

每个实验默认写入：

- `config.yaml`：当次实验实际使用配置。
- `train.log`：训练日志。
- `metrics.csv`：逐 epoch 训练、验证指标。
- `summary.json`：最佳验证 epoch 对应的测试结果和资源统计。
- `router_stats.npz`：路由方法的专家权重和负载统计。
- `best_checkpoint.pth`：按配置保存的最佳 checkpoint。
- `visualizations/`：收敛曲线和路由预览图。

顺序运行脚本还会维护：

- `outputs/tables/full_results.csv`
- `outputs/tables/full_run_manifest.jsonl`
- `outputs/tables/full_run_failures.jsonl`

这些文件用于确认每一组训练-测试结束后是否已完成当次结果归档。

## 实验协议

默认协议来自 `pipeline_config_template.yaml`：

- backbone：`vit_base_patch16_224`
- 预训练：ImageNet-1K
- 输入尺寸：224
- seed：0
- epoch：100
- batch size：128
- optimizer：AdamW
- scheduler：cosine + warmup
- AMP：启用，默认 `float16`
- 验证集选择：按 `val_top1` 选择最佳 epoch
- 测试规则：只报告最佳验证 epoch 对应的测试集 top-1
- early stopping：默认关闭

不同方法的学习率在 `training.method_overrides` 中单独配置。运行时可用 `--epochs`、`--batch-size`、`--seed`、`--device` 等命令行参数临时覆盖。

## 消融实验

配置模板保留以下消融组：

- `core_structure`
- `collapse_guard`
- `capacity`
- `scope`

运行方式：

```bash
python run_all_experiments.py \
  --config pipeline_config_template.yaml \
  --include-ablations \
  --ablation-groups core_structure collapse_guard \
  --ablation-dataset vtab1k_cifar100 \
  --data-root data \
  --output-root outputs \
  --keep-going
```

消融 run 会以 `ablation_<group>_<variant>` 命名，并同样写入 `summary.json`、累计结果表和 manifest。

## 复现注意事项

- 本目录只是核心代码包，不包含数据和已训练结果。
- 若直接从本目录运行，请确保当前工作目录是 `core_experiment_code/`，或者命令中的路径改成绝对路径。
- `--no-pretrained` 会禁用 ImageNet 预训练权重，仅用于调试，不应用于正式报告复现。
- 路由统计只对 `para_cvpt`、`cvpt_expert_adapter` 和 `expert_adapter_pool` 等路由方法有意义，非路由方法对应字段为空。
- 若需要与最终报告完全一致，应固定 seed、数据划分、训练轮数、batch size 和配置模板中的方法超参数。
