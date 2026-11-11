"""Regenerate standalone Bash files after editing passive_pass_campaign.py."""
from pathlib import Path

folder = Path(__file__).resolve().parent
source = folder/'passive_pass_campaign.py'
payload = source.read_text(encoding='utf-8-sig').rstrip()
source.write_text(payload+'\n',encoding='utf-8',newline='\n')
for dataset in ('tictactoe','adult','dota2','mnist','imdb'):
    text = '''#!/usr/bin/env bash
set -Eeuo pipefail
# Standalone PASS-only launcher. Default: DRY_RUN=1 (print commands only).
# Full run: DRY_RUN=0 CONFIRM_QUALITATIVE_RUN=1 bash this_script.sh
# All alphas, two pure attacks once, two lazy attacks at gaps 3/6/10.
# One CSV per dataset after all tasks succeed. See README_passive_pass.md.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
exec "${MERGER_PYTHON_BIN:-${PYTHON_BIN:-python}}" - --project-root "${PROJECT_ROOT}" --dataset DATASET "$@" <<'PY_PASS_CAMPAIGN'
'''.replace('--dataset DATASET','--dataset '+dataset)
    (folder/f'run_{dataset}_passive_pass.sh').write_text(text+payload+'\nPY_PASS_CAMPAIGN\n',encoding='utf-8',newline='\n')
    print(f'generated run_{dataset}_passive_pass.sh')
