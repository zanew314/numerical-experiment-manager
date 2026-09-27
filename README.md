<div align="center">

# Numerical Experiment Manager

A guidance-first skill package for reproducible Python numerical/ML experiments:
versioned structure tracking, torch.fx data flow, wall-clock baseline comparison,
early stopping, and small-parameter tuning.

[中文说明](README.zh-CN.md) · [Contributors](CONTRIBUTORS.md) · [Skill packages](#skill-packages) · [Installation](#installation) · [Quick start](#quick-start) · [Security and scope](#security-and-scope)

![version](https://img.shields.io/badge/version-0.1.0-blue)
![skills](https://img.shields.io/badge/skills-1-2ea44f)
![license](https://img.shields.io/badge/license-MIT-green)

</div>

<p align="center">
  If this project helps your work, please consider giving the repository a Star ⭐
  <a href="https://github.com/zanew314/numerical-experiment-manager"><img alt="GitHub stars" src="https://img.shields.io/github/stars/zanew314/numerical-experiment-manager?style=social"></a>
</p>

## What This Repository Is

This repository is the home of the Numerical Experiment Manager. It keeps a
versioned structure record of a Python numerical/ML experiment project and, on
request, tests it: it runs the program against a fixed wall-clock baseline,
traces the model's data flow with `torch.fx`, produces an experiment report,
stops a run that looks trapped in a local optimum, and micro-tunes small
parameters.

The design is **guidance-first**: the workflow lives in the package's `SKILL.md`
and `prompts/`; a few small, dependency-free scripts do the deterministic
bookkeeping. The coding agent drives every interactive step. Use the root page as
the public map, then open the skill package for concrete use.

## Skill Packages

| Package | Use it for | Start here |
| --- | --- | --- |
| [`numerical-experiment-manager`](skills/numerical-experiment-manager/) | Versioning a project's structure, diffing versions, tracing data flow with torch.fx, running wall-clock experiments against a baseline, early stopping on a local optimum, and small-parameter tuning. | [`README`](skills/numerical-experiment-manager/README.md) · [`SKILL`](skills/numerical-experiment-manager/SKILL.md) |

## Installation

The recommended path is AI-assisted installation: ask your coding agent to clone
or update this repository, read the Skill instructions, link the package that
contains `SKILL.md`, and verify discovery.

```text
Please install the Numerical Experiment Manager Skill for me.

Repository: https://github.com/zanew314/numerical-experiment-manager.git
Branch: main
Skill path:
- skills/numerical-experiment-manager

Steps:
1. Clone or update the repository locally.
2. Read README.md, skills/numerical-experiment-manager/SKILL.md, and AGENTS.md if present.
3. If this environment supports local Skill discovery, link the directory that contains SKILL.md into the local skills directory.
4. Verify that the installed Skill is discoverable.
5. Tell me the installed path, whether a restart is needed, and give me one test prompt.
```

Manual fallback for opencode and Codex-style local discovery:

```bash
git clone https://github.com/zanew314/numerical-experiment-manager.git
cd numerical-experiment-manager

# opencode skill directory
mkdir -p ~/.config/opencode/skill
ln -s "$PWD/skills/numerical-experiment-manager" \
      ~/.config/opencode/skill/numerical-experiment-manager

# Codex-style local discovery
mkdir -p ~/.codex/skills
ln -s "$PWD/skills/numerical-experiment-manager" \
      ~/.codex/skills/numerical-experiment-manager
```

On Windows, create a junction instead of a symlink:

```powershell
New-Item -ItemType Junction `
  -Path "$env:USERPROFILE\.config\opencode\skill\numerical-experiment-manager" `
  -Target "E:\path\to\numerical-experiment-manager\skills\numerical-experiment-manager"
```

If your agent uses a different local Skill directory, replace the path above with
that configured path.

## Quick Start

```bash
git clone https://github.com/zanew314/numerical-experiment-manager.git
cd numerical-experiment-manager/skills/numerical-experiment-manager

python scripts/scan_structure.py /path/to/project
python scripts/diff_structure.py /path/to/project --from v0001 --to v0002
python scripts/fx_dataflow.py /path/to/project/model.py --entry MLP --input-shape 1,2
python scripts/watch_experiment.py /path/to/project --run-id baseline --cmd "python main.py"
```

See the package [README](skills/numerical-experiment-manager/README.md) and
[`SKILL.md`](skills/numerical-experiment-manager/SKILL.md) for the full workflow.

## Repository Layout

```text
numerical-experiment-manager/
├── README.md
├── README.zh-CN.md
├── SKILL.md
├── CONTRIBUTORS.md
├── LICENSE
└── skills/
    └── numerical-experiment-manager/
        ├── README.md
        ├── README.zh-CN.md
        ├── SKILL.md
        ├── agents/
        ├── prompts/
        ├── references/
        ├── scripts/
        └── requirements-test.txt
```

Keep each package's prompts, references, scripts, and generated evidence inside
the package that owns them.

## Validation

There is no root build step. The four package scripts use only the Python
standard library; `fx_dataflow.py` additionally needs PyTorch. Validate a change
by running the scripts against a small project (see the package README) and by
checking the package's `SKILL.md` and README links.

## Security and Scope

Do not commit generated run outputs, private datasets, `.env` files, or secrets.
`.nems/` is generated evidence and should be gitignored in managed projects.
Public examples should include source notes for benchmark data. This skill targets
Python projects; Julia and MATLAB are out of scope.
