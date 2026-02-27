# Contributing

Development setup, testing, CI pipeline, and code style guide for SimScaleAI.

---

## Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/SimScaleAI.git
cd SimScaleAI

python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[all]"
```

---

## Running Tests

### Full Test Suite

```bash
pytest tests/ -v
```

### By Module

```bash
# Model tests (BC, VLA, registry)
pytest tests/test_models.py -v

# RL tests (PPO, rewards)
pytest tests/test_rl.py -v

# Simulation tests (environments, factory)
pytest tests/test_sim.py -v

# Training tests (trainer, datasets)
pytest tests/test_training.py -v
```

### Markers

```bash
# Skip slow tests
pytest tests/ -v -m "not slow"

# Skip GPU tests
pytest tests/ -v -m "not gpu"
```

### Coverage

```bash
pytest tests/ --cov=simscaleai --cov-report=term-missing
```

### Test Structure

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_models.py` | 9 | BC forward/predict, VLA forward/predict/chunking, registry CRUD |
| `test_rl.py` | 5 | ActorCritic forward/action, RolloutBuffer, DistanceReward, CompositeReward |
| `test_sim.py` | 6 | ReachEnv creation/reset/step/multi-step, factory list/invalid |
| `test_training.py` | 4 | Trainer smoke test, DummyDataset state/image, DataLoader batching |

All tests run on CPU with small model configs for fast execution (~2 seconds total).

---

## Code Style

### Linting

SimScaleAI uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Ruff Configuration

From `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM"]
```

| Rule Set | Description |
|----------|-------------|
| `E` | pycodestyle errors |
| `F` | pyflakes |
| `I` | isort (import sorting) |
| `N` | pep8-naming |
| `W` | pycodestyle warnings |
| `UP` | pyupgrade (modern Python syntax) |
| `B` | flake8-bugbear |
| `SIM` | flake8-simplify |

### Type Checking

```bash
# Run mypy (non-blocking in CI)
mypy simscaleai/
```

---

## CI Pipeline

The GitHub Actions CI runs on every push and pull request:

### Job 1: `lint-and-test`

Matrix: Python 3.10, 3.11, 3.12

1. Install package with `[dev]` extras
2. Run `ruff check .` — lint
3. Run `mypy simscaleai/` — type check (continue-on-error)
4. Run `pytest tests/ -v -m "not slow and not gpu"` — unit tests

### Job 2: `integration-test`

Python 3.11 only, depends on `lint-and-test` passing

1. Install package with `[all]` extras
2. **BC smoke test**: Train BC model for 5 steps on CPU with dummy data
3. **VLA smoke test**: Train VLA model for 3 steps on CPU with image data

### CI Configuration

Located in `.github/workflows/ci.yml`.

---

## Commit Conventions

Follow conventional commits for clear history:

```
feat: add push environment
fix: correct MuJoCo renderer API for depth rendering
docs: add simulation module documentation
test: add pick-place environment tests
refactor: extract Jacobian IK to utility function
chore: update dependencies in pyproject.toml
```

---

## Adding Tests

### Model Test Pattern

```python
class TestMyModel:
    def test_forward(self):
        """Model forward pass produces correct shapes and loss."""
        model = ModelRegistry.create("my_model", state_dim=10, action_dim=3)
        batch = {
            "observations": {"state": torch.randn(8, 10)},
            "actions": torch.randn(8, 3),
        }
        output = model(batch)

        assert "loss" in output
        assert output["loss"].ndim == 0  # scalar
        assert output["predicted_actions"].shape == (8, 3)

    def test_predict(self):
        """Inference produces correct action shape."""
        model = ModelRegistry.create("my_model", state_dim=10, action_dim=3)
        obs = {"state": torch.randn(1, 10)}

        with torch.no_grad():
            actions = model.predict(obs)

        assert actions.shape == (1, 3)
```

### Environment Test Pattern

```python
class TestMyEnv:
    def test_env_creation(self):
        env = make_env("my_task")
        assert env.action_space.shape == (4,)
        env.close()

    def test_reset(self):
        env = make_env("my_task")
        obs, info = env.reset(seed=42)
        # Assert observation shapes
        env.close()

    def test_step(self):
        env = make_env("my_task")
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        env.close()
```

---

## Project Conventions

### File Organization

- One class per file for major components (environments, models, agents)
- `__init__.py` files re-export key public APIs
- Tests mirror source structure: `simscaleai/models/bc.py` → `tests/test_models.py`

### Naming

- Classes: `PascalCase` — `BehaviorCloning`, `PPOAgent`, `ReachEnv`
- Functions/methods: `snake_case` — `make_env`, `create_model`, `compute_gae`
- Private methods: `_prefix` — `_apply_action`, `_get_obs`, `_train_step`
- Constants: `UPPER_SNAKE` — `_ASSET_DIR`, `_ENV_REGISTRY`
- Config classes: `*Config` suffix — `SimConfig`, `TrainConfig`, `PPOConfig`

### Docstrings

- Module-level docstrings on all files
- Class-level docstrings describing purpose and key attributes
- Method docstrings for public API methods
- Type hints on all function signatures

### Dependencies

- Core dependencies in `[project.dependencies]`
- Optional extras in `[project.optional-dependencies]`
- Pin minimum versions, not exact versions: `torch>=2.2` not `torch==2.2.0`
