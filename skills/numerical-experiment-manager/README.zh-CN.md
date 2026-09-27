<div align="center">

# 数值实验管理器

以指导为核心的 skill：为 Python 数值/机器学习实验项目维护结构版本记录，并在用户
要求时进行测试——torch.fx 数据流、固定基线的等墙钟实验、早停与微调。

[English](README.md) · [SKILL](SKILL.md) · [仓库](https://github.com/zanew314/numerical-experiment-manager) · [参考](references/) · [脚本](scripts/)

![version](https://img.shields.io/badge/version-0.1.0-blue)
![scripts](https://img.shields.io/badge/scripts-4-2ea44f)
![license](https://img.shields.io/badge/license-MIT-green)

</div>

## 这个 skill 是什么

本 package 教会编码 agent 把数值/ML 实验项目当作可复现的战役来管理。**指导主体在
[`SKILL.md`](SKILL.md)**；四个轻量、零依赖的脚本只负责确定性记账。它不是独立流水线——
每一步交互都由 agent 在对话中执行。

| 工作流 | 触发 | 行为 |
| --- | --- | --- |
| A. 结构版本化 | 首次看到项目，或模型文件被修改 | 用 AST 生成版本化快照（`v0001`、`v0002`…）；作者的每次结构改动成为一个版本；出结果后把结构差异与指标交给 LLM 解释性能变化。 |
| B. 自动化测试 | 用户说“帮我测试”/“run this” | 运行程序，可选用 `torch.fx` 追踪数据流，产出精确结构图与实验报告；当相对误差长期高于 0.10 时早停；可微调少量小参数。 |

证据纪律：保留原始 loss 序列，在等墙钟下比较 **best-so-far**，绝不用看似合理的摘要替换
真实测量。

## 脚本

| 脚本 | 用途 |
| --- | --- |
| [`scripts/scan_structure.py`](scripts/scan_structure.py) | AST 扫描 → 版本化的 `.nems/structure/` JSON + Markdown。 |
| [`scripts/diff_structure.py`](scripts/diff_structure.py) | 比较两个版本的结构差异 → 供 LLM 使用的 JSON。 |
| [`scripts/fx_dataflow.py`](scripts/fx_dataflow.py) | `torch.fx` 追踪 → Mermaid/DOT 数据流图 + 形状。 |
| [`scripts/watch_experiment.py`](scripts/watch_experiment.py) | 运行程序、记录墙钟 loss、早停、重训一次并写运行产物。 |

## 快速开始

```bash
# 1. 版本化项目结构（首次扫描生成 v0001）。
python scripts/scan_structure.py /path/to/project

# 2. 作者修改模型后再次扫描（生成 v0002）并比较。
python scripts/scan_structure.py /path/to/project
python scripts/diff_structure.py /path/to/project --from v0001 --to v0002

# 3. 追踪数据流（可选，需要 PyTorch）。
python scripts/fx_dataflow.py /path/to/project/model.py \
    --entry MLP --input-shape 1,2 --save .nems/figures

# 4. 先记录一次基线，再运行并监控实验。
python scripts/watch_experiment.py /path/to/project --run-id baseline \
    --cmd "python main.py" --loss-source jsonl
python scripts/watch_experiment.py /path/to/project --run-id run_001 \
    --cmd "python main.py" --baseline .nems/baseline/loss_time.jsonl \
    --tolerance 0.10 --sustain 3
```

随后填写 [`prompts/structure_diff.md`](prompts/structure_diff.md) 与
[`prompts/experiment_report.md`](prompts/experiment_report.md)，由 agent 写出变更报告与
`REPORT.md`。

## 产物

所有持久状态都写在被管理项目根目录的 `.nems/` 下：

```text
.nems/
├── structure/   # index.json + vNNNN.json/md（版本化 AST 快照）
├── baseline/    # 固定的墙钟基线曲线
├── runs/<id>/   # loss_time.jsonl、metrics.json、early_stop.json、REPORT.md
├── reports/     # vNNNN_summary.md、vNNNN_to_vNNNN_change.md
└── figures/     # dataflow_*.md/dot
```

`.nems/` 是证据而非源码，应加入 `.gitignore`。完整契约见
[`references/artifacts.md`](references/artifacts.md)。

## 许可证与边界

MIT（见仓库 [`LICENSE`](../../LICENSE)）。不要提交生成的运行产物、私有数据集或密钥。
本 skill 面向 Python 项目；Julia 与 MATLAB 不在范围内。
