# ViT-B/16 高效微调与 CAPA 实验仓库

本仓库整理了基于 ViT-B/16 的参数高效微调实验代码与三数据集有效实验输出。项目比较 Linear Probe、Full Fine-tuning、VPT、CVPT、AdaptFormer、SSF、LoRA、Para-CVPT、Expert Adapter Pool 和 CAPA 等方法，并重点研究 CAPA（Cross-Attentive Prompt-Adapter Expert Pooling）：一种结合 CVPT prompt 交互与动态专家 Adapter 池的结构。

当前仓库已经去掉外层打包目录，代码、配置、结果表、图像和输出归档直接暴露在仓库根目录下。

## 目录结构

```text
.
├── README.md
├── pipeline_config_template.yaml
├── run_all_experiments.py
├── src/
│   ├── data.py
│   ├── modeling.py
│   ├── cvpt_expert_adapter.py
│   └── train.py
├── outputs/
│   ├── vtab1k_cifar100/
│   ├── cub_200_2011/
│   └── oxford_flowers_102/
├── figures/
│   ├── report/
│   └── routing_heatmaps/
├── tables/
│   ├── full_results.csv
│   └── main_results.csv
└── manifest_3datasets.json
```

## 主要内容

- `src/`：核心实验代码，包括数据集读取、ViT/PEFT 模型实现、CAPA/专家池模块和单次训练入口。
- `pipeline_config_template.yaml`：完整实验配置模板，包含数据集、训练协议、方法列表、路由设置、输出规则和消融配置。
- `run_all_experiments.py`：顺序运行多组实验的入口脚本，会在每组训练-测试结束后归档 `summary.json` 并更新累计结果表。
- `outputs/`：三个报告数据集上的有效实验输出，包括日志、逐 epoch 指标、配置、结果摘要、路由统计和可视化预览。
- `figures/report/`：报告正文使用的汇总图。
- `figures/routing_heatmaps/`：路由热力图及其统计记录。
- `tables/`：整理后的实验结果表。
- `manifest_3datasets.json`：三数据集有效输出的运行清单。

## 数据集与方法

实验覆盖三个数据集：

- `vtab1k_cifar100`
- `cub_200_2011`
- `oxford_flowers_102`

主实验方法包括：

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

其中 `cvpt_expert_adapter` 对应本文提出的 CAPA 方法；`expert_adapter_pool` 是只保留专家 Adapter 池的结构对照。

## 环境依赖

建议使用带 CUDA 的 PyTorch 环境：

```bash
pip install torch torchvision numpy matplotlib pyyaml pillow scipy
```

配置模板默认启用 AMP、TF32、cuDNN benchmark 和较高 matmul precision，以提高 GPU 训练效率。

## 单个实验运行

```bash
python src/train.py \
  --config pipeline_config_template.yaml \
  --dataset vtab1k_cifar100 \
  --method cvpt_expert_adapter \
  --data-root data \
  --output-root outputs
```

快速检查管线可使用调试数据：

```bash
python src/train.py \
  --config pipeline_config_template.yaml \
  --dataset debug_fake \
  --method linear_probe \
  --output-root outputs_debug \
  --epochs 1 \
  --batch-size 4
```

## 顺序运行主实验

```bash
python run_all_experiments.py \
  --config pipeline_config_template.yaml \
  --datasets vtab1k_cifar100 cub_200_2011 oxford_flowers_102 \
  --methods linear_probe full_finetune adaptformer ssf lora vpt cvpt para_cvpt cvpt_expert_adapter expert_adapter_pool \
  --data-root data \
  --output-root outputs \
  --keep-going
```

只查看运行顺序、不实际训练：

```bash
python run_all_experiments.py \
  --config pipeline_config_template.yaml \
  --datasets vtab1k_cifar100 cub_200_2011 oxford_flowers_102 \
  --data-root data \
  --output-root outputs \
  --dry-run
```

## 输出说明

每个实验目录通常包含：

- `summary.json`：最佳验证轮次及其对应测试结果。
- `metrics.csv`：逐 epoch 训练、验证指标。
- `train.log`：训练日志。
- `config.yaml`：当次实验实际配置。
- `router_stats.npz`：路由方法的专家权重和负载统计。
- `visualizations/`：收敛曲线和路由预览图。

顺序运行脚本会维护累计结果文件，例如 `tables/full_results.csv`。查看最终分数时，优先读取 `tables/` 和各 run 的 `summary.json`。

## Checkpoint 说明

本仓库的 GitHub 发布版不包含 `best_checkpoint.pth`。这些权重文件单个约数百 MB，超过普通 GitHub Git 的 100MB 单文件限制。如需发布权重，应使用 Git LFS、GitHub Release 附件或外部存储。

## 复现注意事项

- 默认 backbone 为 `vit_base_patch16_224`，使用 ImageNet-1K 预训练权重。
- 默认训练协议为 seed 0、100 epoch、batch size 128、AdamW、cosine 调度和 AMP。
- 验证集选择按 `val_top1` 选取最佳 epoch，并报告该 checkpoint 对应的测试集 top-1。
- `--no-pretrained` 只应用于调试，不应用于正式复现。
- 路由统计只对 `para_cvpt`、`cvpt_expert_adapter` 和 `expert_adapter_pool` 等路由方法有意义。
