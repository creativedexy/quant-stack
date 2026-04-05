# CI/CD Workflows

GitHub Actions runs automated checks on every push and pull request so that
broken code never reaches the main branch unnoticed. Both workflows defined
here execute in parallel — a commit is only green when **both** pass.

---

## ci.yml — Test Suite

**What it checks:** Installs the full project, generates synthetic test data
offline, and runs the entire pytest suite (excluding PyCaret/AutoML tests
which have a known dependency conflict).

**Typical duration:**
- First run on a fresh cache: ~5-6 minutes (pip install dominates).
- Subsequent runs with a warm cache: ~2-3 minutes. The pip cache is keyed on
  the hash of `pyproject.toml`, so it invalidates automatically when
  dependencies change. This reduces the install step from ~3 minutes to ~45
  seconds on repeat runs.

**Steps in order:**
1. Checkout the repository.
2. Set up Python 3.11.
3. Restore pip cache (keyed on `pyproject.toml` hash).
4. `pip install -e ".[all]"` — installs every optional dependency group.
5. `py -3 -m scripts.fetch_data --source synthetic` — generates Parquet files
   that the test suite depends on. This **must** run before pytest.
6. `py -3 -m pytest tests/ -v --ignore=tests/test_models/test_automl.py --tb=short -q`

**Why caching matters:** Without caching, pip downloads and installs ~200 MB
of packages on every run. The `actions/cache` step stores the pip download
cache between runs, keyed on the OS and the hash of `pyproject.toml`. When
dependencies haven't changed, pip skips the download entirely and installs
from the local cache. This typically reduces the install step from ~3 minutes
to ~45 seconds, saving both time and GitHub Actions minutes.

---

## lint.yml — Code Linting

**What it checks:** Runs [ruff](https://docs.astral.sh/ruff/) against `src/`
and `tests/` to catch syntax errors, undefined names, unused imports, and
common Python mistakes.

**Why ruff:** Ruff is a single-binary Python linter written in Rust. It
replaces flake8, isort, and pyflakes with a single tool that installs in
under 2 seconds and lints the entire codebase in under 1 second. No pip
cache is needed because the install is so fast.

**Rules enforced:**
- `E` — pycodestyle errors (formatting issues)
- `F` — pyflakes (undefined names, unused imports, syntax errors)
- `W` — pycodestyle warnings

**Rules deliberately ignored:**
- `E501` — line length. This project does not enforce a maximum line length.

**Note:** The `--exit-zero` flag is intentionally **not** used. If ruff finds
errors, the workflow fails. This is by design — lint failures must block the
build.

---

## Reading a Workflow Failure

When a workflow fails, GitHub shows a red X next to the commit or pull request.

1. Go to the **Actions** tab in the repository.
2. Click the failed workflow run (it will have a red icon).
3. Click the failed job name (e.g., "Test Suite" or "Ruff Linter").
4. Expand the failed step to see the full log output.
5. For pytest failures, look for the `FAILED` lines and the short tracebacks.
6. For ruff failures, each line shows `file:line:col: CODE message`.

---

## Adding a New Test

1. Create a test file in the appropriate `tests/` subdirectory (e.g.,
   `tests/test_features/test_my_new_feature.py`).
2. Follow existing naming conventions: `test_*.py` files, `test_*` functions.
3. If the test needs synthetic data, it will be available automatically — the
   CI workflow generates it before running pytest.
4. Push the commit. The Actions tab will show the new test running.
5. If the test requires network access, mark it with
   `@pytest.mark.integration` — these are skipped in CI by default.

---

## Adding Secrets

Some future features (e.g., alert email delivery, API keys for live data)
will require secrets. GitHub Actions Secrets keep sensitive values out of
YAML files and source code.

**To add a secret:**

1. Go to the repository on GitHub.
2. Navigate to **Settings > Secrets and variables > Actions**.
3. Click **New repository secret**.
4. Enter the name (e.g., `SMTP_PASSWORD`) and value.
5. Click **Add secret**.

**To use a secret in a workflow:**

```yaml
env:
  SMTP_PASSWORD: ${{ secrets.SMTP_PASSWORD }}
```

Secrets are masked in logs — GitHub replaces their values with `***` in all
output. They are not available to pull requests from forks (for security).

**Planned secrets (not yet needed):**
- `SMTP_PASSWORD` — for alert email delivery
- `ANTHROPIC_API_KEY` — for intelligent notebook features
- `IB_ACCOUNT_ID` — for Interactive Brokers paper trading

Never put API keys, passwords, or tokens directly in workflow YAML files or
in `config/settings.yaml`. Always use GitHub Actions Secrets.

---

## Manual Verification After First Push

After pushing these workflows to GitHub for the first time:

1. Go to the **Actions** tab in the repository.
2. Confirm both workflows ("CI" and "Lint") appear in the left sidebar.
3. Click into each running workflow and verify:
   - **CI**: All 6 steps complete successfully, pytest reports pass counts.
   - **Lint**: ruff exits with code 0 (no errors found).
4. Check the commit or PR — it should show two green checkmarks.
5. Verify the badge URLs in README.md resolve (they will show "no status"
   until the first workflow run completes).
