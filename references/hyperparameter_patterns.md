# Hyperparameter Extraction Patterns

This reference lists the source-code patterns the manager recognizes when extracting
hyperparameters from a Python numerical/ML experiment project. Extraction is text/regex based
(no `ast`, no dynamic probing); the location is confirmed by grep plus the LLM hint.

## 1. Class-encapsulated config (highest confidence)

```python
class Config:
    LR = 0.001          # -> name=LR,       value=0.001, type=float, confidence=high
    NUM_LAYERS = 4      # -> name=NUM_LAYERS, value=4,    type=int,  confidence=high
    HIDDEN_DIM = 64
    EPOCHS = 1000
```

Detection: an indented `NAME = <literal>` where `NAME` is `UPPER_CASE` inside a class whose name
contains `config`, `param`, `setting`, or `hparam`. Location form: `main.py:12 (Config.LR)`.

## 2. Module-level constants (high confidence)

```python
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
```

Detection: a top-level `NAME = <literal>` with `UPPER_CASE` name. Confidence `high` when the name
contains a known keyword (`lr`, `learning_rate`, `epoch`, `batch`, `hidden`, `layer`, `dim`,
`weight`, `tol`, `threshold`), otherwise `medium`.

## 3. Instance attributes (medium confidence)

```python
class Trainer:
    def __init__(self):
        self.lr = 1e-3
        self.epochs = 500
```

Detection: `self.NAME = <literal>`. Location form: `train.py:20 (Trainer.__init__.lr)`.

## 4. Function defaults (medium confidence)

```python
def build_model(num_layers=4, hidden_dim=64):
    ...
```

Detection: `<literal>` defaults in a `def` signature whose name is a known keyword.

## 5. argparse arguments (medium confidence)

```python
parser.add_argument("--lr", type=float, default=1e-3)
```

Detection: `--name` plus `default=<literal>`. Location form: `main.py:30 (argparse --lr)`.

## Value typing

| Literal form | type |
| --- | --- |
| `0.001`, `1e-3`, `1.0` | float |
| `64`, `1000` | int |
| `True` / `False` | bool |
| `"adam"` / `'sin'` | str |
| `None` | null |
| anything else | expr |

## Injection rules

- Only the parameters the user selected are replaced, with
  `NAME = get_param("NAME", default=<original literal>)`.
- Unselected parameters keep their original literal.
- The adapter module is placed at the project root and resolves a non-conflicting name:
  `nems_config.py` -> `nems_param_loader.py` -> `_nems_config.py`.
- Original sources are always snapshotted under `.nems/code_snapshots/<timestamp>/` before edits.
