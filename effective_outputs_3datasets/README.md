# 三个报告数据集的有效实验输出

本目录保存最终报告实际使用的有效实验输出，仅包含以下三个数据集：

- `vtab1k_cifar100`
- `cub_200_2011`
- `oxford_flowers_102`

目录中不包含探索性数据集、冒烟测试输出、临时进程日志和无关实验结果。

## 目录结构

- `outputs/<dataset>/<method>/seed_0/`：10 个报告方法在各数据集上的单次实验归档。
- `figures/report/`：报告正文使用的汇总图。
- `figures/routing_heatmaps/`：三个路由方法在三个报告数据集上的路由热力图。
- `tables/`：整理后的累计结果表。
- `manifest_3datasets.json`：本目录的文件清单和来源说明。

## 单次实验文件

典型单次实验目录包含：

- `summary.json`：最终结果摘要，包括最佳验证轮次和该 checkpoint 对应的测试结果。
- `metrics.csv`：逐 epoch 的训练、验证指标。
- `train.log`：原始训练日志。
- `config.yaml`：该次实验实际使用的配置。
- `visualizations/`：收敛曲线和路由预览图。
- `router_stats.npz`：路由方法在测试集上的专家权重和负载统计。
- `best_checkpoint.pth`：本地完整输出包中，启用 checkpoint 保存的方法会包含对应的最佳权重文件。

注意：`best_checkpoint.pth` 单文件较大，普通 GitHub Git 推送有 100MB 单文件限制。因此 GitHub 发布版默认不包含这些 checkpoint 文件；如需上传权重，应使用 Git LFS、Release 附件或外部网盘。

## 包含的方法

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

其中 `cvpt_expert_adapter` 对应本文提出的 CAPA 方法；`expert_adapter_pool` 是只保留专家 adapter 池的结构对照。

## 使用建议

- 查看最终分数时，优先读取 `tables/main_results.csv` 和各 run 的 `summary.json`。
- 复核训练过程时，查看对应 run 的 `metrics.csv` 与 `train.log`。
- 分析路由机制时，使用 `figures/routing_heatmaps/` 下的热力图和 `router_stats.npz`。
- 复现报告图表时，优先使用 `figures/report/` 和 `tables/` 中的整理结果。
