# Codex workspace

This workspace uses `AGENTS.md`, repository skills in `.agents/skills/`, and the
local environment profile in `.codex/environments/environment.toml`.
Claude's files are independent and are not modified or linked.

## Environment and actions

The profile reuses the existing Python 3.15 `.venv`. It does not create another
venv, install packages, change global settings, or launch a server during setup.
In a fresh worktree without `.venv`, it falls back to the current Python and
reports missing tooling. Creating a venv and installing `.[dev]` require explicit
authorization; the status action only reports what is available.
It returns a nonzero exit status if Python is too old, a development dependency
is missing, or Node is unavailable for JavaScript syntax checks.

Available actions: **Check**, **Compare Offline**, **Dashboard**, and
**Environment Status**. The dashboard action starts a localhost server on port
8000 and runs until stopped. It can probe configured model backends; run it only
when requested. Check and compare use a temporary embedding cache, deleted on
normal exit. A forced process kill can leave the printed temporary path behind.

Run any action manually, for example:

```bash
python3 -B .codex/scripts/workspace.py doctor
python3 -B .codex/scripts/workspace.py check
python3 -B .codex/scripts/workspace.py compare sample.md
```

## Skills

Use `$check`, `$compare`, `$add-chunking-strategy`, `$add-backend`,
`$chunker-reviewer`, or `$public-mode-auditor`. The review skills perform reviews
in the current chat; they do not configure or launch subagents.

Claude slash commands are represented by skills. Claude's post-edit lint hook
is represented by explicit validation instructions, not an automatic hook.
Permissions remain controlled by the app and current task authorization.

Codex discovers repository skills in `.agents/skills`; restart the chat/app if
they do not appear. The environment file is provided for the app's local
environment selector; CLI checks alone do not prove the UI has selected it.

References: [local environments](https://learn.chatgpt.com/docs/environments/local-environment),
[repository skills](https://learn.chatgpt.com/docs/build-skills).
