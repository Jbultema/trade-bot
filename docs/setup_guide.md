# Trade Bot Setup Guide

Status: canonical setup guide. Last reviewed: 2026-07-05.

This guide is for users who do not regularly use GitHub, VS Code, Poetry, pyenv,
or local Python dashboards. It walks through the full setup process and explains
what each step does.

## What You Are Setting Up

Trade Bot is a local Python project. It runs on your machine and stores data
locally. It uses:

- Git for version control,
- Python 3.12 for execution,
- pyenv for Python version management,
- Poetry for dependencies and virtual environments,
- DuckDB and SQLite for local storage,
- Streamlit for the dashboard,
- VS Code or another editor for file edits.

The app does not need cloud hosting for normal use.

## Skill Level Assumption

You should be comfortable copying commands into Terminal, but you do not need to
be a software engineer. Follow the commands from the repo root unless the guide
says otherwise.

## Required Tools

### macOS Command Line Tools

Install Apple's command line tools:

```bash
xcode-select --install
```

If already installed, macOS will tell you.

### Homebrew

Install Homebrew if it is not installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then confirm:

```bash
brew --version
```

### Git

Install Git:

```bash
brew install git
git --version
```

Configure your identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

Use the email associated with the GitHub account you want commits attributed to.

### pyenv

Install pyenv:

```bash
brew install pyenv
```

Add pyenv to your shell startup file. For zsh on macOS, edit `~/.zshrc` and add:

```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Restart Terminal, then check:

```bash
pyenv --version
```

### Python 3.12.6

Install the project Python:

```bash
pyenv install -s 3.12.6
```

### Poetry

Install Poetry:

```bash
brew install poetry
poetry --version
```

Tell Poetry to create virtual environments inside the repo:

```bash
poetry config virtualenvs.in-project true
```

### VS Code

Install VS Code:

```bash
brew install --cask visual-studio-code
```

Recommended extensions:

- Python,
- Ruff,
- YAML,
- Markdown All in One,
- GitLens if you like Git history views.

## Getting The Repo

### Option A: Clone From GitHub

If the project is on GitHub and you have access:

```bash
mkdir -p ~/repos
cd ~/repos
git clone git@github.com:YOUR_USER_OR_ORG/trade-bot.git
cd trade-bot
```

If SSH is not set up, use the HTTPS URL from GitHub instead.

### Option B: Use A Local Folder

If someone gives you a zipped folder:

1. Unzip it.
2. Move it to a stable location such as `~/repos/trade-bot`.
3. Open Terminal.
4. Run:

```bash
cd ~/repos/trade-bot
git status --short
```

If Git says it is not a repository, initialize it only if you are expected to
track local changes:

```bash
git init
git add README.md docs src tests configs pyproject.toml poetry.lock poetry.toml
git commit -m "Initial local trade-bot import"
```

Do not add `data/`, `reports/`, `.env`, or `.venv/`.

## Initial Project Setup

From the repo root:

```bash
pyenv local 3.12.6
poetry env use "$(pyenv which python)"
poetry install
```

Confirm versions:

```bash
python --version
poetry run python --version
poetry env info
```

Both Python versions should be 3.12.x, preferably 3.12.6.

## Environment Variables

Create `.env` only if optional data sources require keys.

```bash
touch .env
```

Never commit `.env`.

If a data vendor key is used, store it as:

```text
SOME_VENDOR_API_KEY=your_key_here
```

Keep secrets out of screenshots, commits, and shared docs.

## First Smoke Test

Run the test suite:

```bash
poetry run pytest -q
```

If that is too slow for first setup, run a smaller smoke test:

```bash
poetry run pytest tests/test_config.py tests/test_dashboard_app.py -q
```

Run lint:

```bash
poetry run ruff check src tests
```

## First Data Refresh

Build the first daily snapshot:

```bash
poetry run trade-bot run-daily-update
```

This may take time because it fills local caches.

If you only want to test cached behavior after a successful first run:

```bash
poetry run trade-bot run-daily-update --cached-data --cached-macro --cached-news
```

## Open The Dashboard

```bash
poetry run trade-bot run-dashboard
```

Open:

```text
http://localhost:8501
```

If the port is busy:

```bash
poetry run trade-bot run-dashboard --port 8502 --pid-path reports/streamlit-8502.pid --log-path reports/streamlit-8502.log
```

When you are done, stop the managed dashboard without relying on Ctrl-C:

```bash
poetry run trade-bot stop-dashboard
```

Use the dashboard sidebar default:

```text
Latest snapshot (fast)
```

Use `Live pipeline` only for intentional recomputation.

## Confirm The App Is Working

In the dashboard:

1. Check the latest update strip below the header.
2. Confirm the market date is recent.
3. Read the Action Headline.
4. Confirm Book Alignment appears near the top.
5. Open Research Lab and verify the aggregate section renders.
6. Open Monitoring and verify warehouse health.

## Seed Paper Monitoring

After the first daily update:

```bash
poetry run trade-bot migrate-warehouse
poetry run trade-bot seed-monitoring-windows --start-date YYYY-MM-DD --top-n 5 --capital-base 10000
poetry run trade-bot run-paper-valuation
```

Use the latest snapshot market date for `YYYY-MM-DD`:

```bash
poetry run trade-bot list-snapshots --limit 10
```

## Recommended Daily Commands

Most users need only:

```bash
poetry run trade-bot run-daily-update
poetry run trade-bot run-dashboard
```

Optional:

```bash
poetry run trade-bot list-champion-challenger
```

## Recommended Weekly Commands

```bash
poetry run trade-bot run-daily-update
poetry run trade-bot run-signal-evidence --experiment-dir data/experiments_reset_v2
poetry run trade-bot list-champion-challenger
poetry run pytest -q
```

## Recommended Monthly Commands

```bash
poetry run trade-bot run-ml-diagnostics --config configs/baseline.yaml --profile standard
poetry run trade-bot run-entry-date-analysis
poetry run ruff check src tests
poetry run pytest -q
```

## Using VS Code

Open the repo:

```bash
code .
```

In VS Code:

1. Open the Command Palette.
2. Choose `Python: Select Interpreter`.
3. Select `.venv/bin/python` inside the repo.
4. Open the integrated terminal.
5. Confirm:

```bash
poetry run python --version
```

Use VS Code for editing. Use Terminal for running commands.

## Using Git Safely

Check what changed:

```bash
git status --short
```

See file differences:

```bash
git diff
```

Stage source/docs/config changes:

```bash
git add README.md docs src tests configs pyproject.toml poetry.lock poetry.toml
```

Commit:

```bash
git commit -m "Describe the change"
```

Do not stage:

- `.env`,
- `.venv/`,
- `data/`,
- `reports/`,
- `*.duckdb`,
- `*.sqlite`,
- large CSV/parquet/cache files.

If you accidentally stage a file:

```bash
git restore --staged PATH_TO_FILE
```

## GitHub Basics

To push to GitHub, you need:

- a GitHub account,
- access to the repo,
- SSH or HTTPS authentication configured.

Check remotes:

```bash
git remote -v
```

Push:

```bash
git push
```

If authentication is confusing, do not keep retrying random credential changes.
Ask for help or use a local-only Git workflow until GitHub access is configured.

## Working With Codex

Start Codex from the repo root. The working directory matters.

Good prompts:

- "Run the dashboard tests and fix failures."
- "Add a new strategy family and update docs."
- "Explain why Monitoring is not updating."
- "Refactor this dashboard section without changing behavior."

Good safety habits:

- Ask Codex to inspect before editing.
- Ask Codex to run focused tests.
- Ask Codex to avoid committing secrets or data.
- Ask Codex to summarize changed files.

If Codex keeps asking for approvals unexpectedly:

1. Confirm the current working directory:

```bash
pwd
```

2. Confirm it is inside the repo.
3. Restart Codex from the repo root if needed.

## Local Data And Backups

Important local state lives under:

- `data/run_store/trade_bot.duckdb`,
- `data/run_store/snapshots/`,
- `data/trading_journal.sqlite`,
- `reports/`.

These are not normally committed. If you need to move machines, copy the repo
and relevant local data folders deliberately. Do not publish them accidentally.

## Troubleshooting

### Poetry cannot find Python 3.12

Run:

```bash
pyenv install -s 3.12.6
pyenv local 3.12.6
poetry env use "$(pyenv which python)"
```

### Dependencies seem broken

Run:

```bash
poetry install --sync
```

If needed:

```bash
poetry env remove .venv
poetry install
```

### Streamlit port is busy

Use another port:

```bash
poetry run trade-bot run-dashboard --port 8502 --pid-path reports/streamlit-8502.pid --log-path reports/streamlit-8502.log
```

The archived V1 dashboard is available only when you need comparison/debugging:

```bash
poetry run trade-bot run-dashboard-v1
```

### Dashboard is stale

Run:

```bash
poetry run trade-bot run-daily-update
```

Refresh the browser.

### Monitoring is blank

Run:

```bash
poetry run trade-bot migrate-warehouse
poetry run trade-bot seed-monitoring-windows --start-date YYYY-MM-DD
poetry run trade-bot run-paper-valuation
```

### A command fails with missing data

First refresh the daily stack:

```bash
poetry run trade-bot run-daily-update
```

Then rerun the command.

### Tests fail after local edits

Run the focused failing test first. Then run:

```bash
poetry run ruff check src tests
poetry run pytest -q
```

### You want to reset generated local data

Do not delete data casually if you have paper/live journal records. If you need a
clean research rerun, prefer writing to a new output directory rather than
removing old data.

## Handoff Checklist For A New User

Before a new user operates the app:

- They can open Terminal.
- They can run `poetry run trade-bot run-daily-update`.
- They can open the dashboard.
- They understand it does not place trades.
- They understand paper monitoring versus Forward Test.
- They know where not to commit secrets/data.
- They know how to read the Action Headline and Book Alignment.
- They know how to use the right-side Term Lookup.
- They know who to ask before using live money.

## Minimal First-Day Path

For a new user who just needs the shortest path:

```bash
cd /path/to/trade-bot
pyenv local 3.12.6
poetry install
poetry run pytest tests/test_config.py -q
poetry run trade-bot run-daily-update
poetry run trade-bot run-dashboard
```

Then read `docs/user_guide.md`.
