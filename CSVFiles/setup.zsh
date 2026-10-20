#!/usr/bin/env zsh
# ------------------------------------------------------------------
# One-step Python environment for the AO3 scraper + analysis scripts.
#
#   source setup_env.zsh    -> builds the venv AND leaves it active in this terminal
#   ./setup_env.zsh         -> builds the venv (then run: source .venv/bin/activate)
#
# Put this file in your FanficCode folder. Safe to re-run: it reuses the venv
# and only installs what is missing or outdated.
#
# Packages (none are in a default Python install; numpy arrives with pandas):
#   pandas, numpy        data tables and numbers (scraper + grapher)
#   matplotlib, seaborn  plotting (grapher)
#   scipy                Mann-Whitney tests and correlations (grapher)
#   statsmodels          the joint regression model (grapher)
#   beautifulsoup4       reads AO3 HTML, imported as bs4 (scraper + supply counts)
#   playwright           drives the real browser (scraper + supply counts)
# ------------------------------------------------------------------

SCRIPT_DIR="${0:A:h}"
VENV_DIR="${SCRIPT_DIR}/.venv"
PACKAGES=(pandas numpy matplotlib seaborn scipy statsmodels beautifulsoup4 playwright statsmodels)

setup_ao3_env() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found. Install it first (https://www.python.org/downloads/ or: brew install python)."
    return 1
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR" || return 1
  else
    echo "Reusing existing virtual environment in $VENV_DIR"
  fi

  source "$VENV_DIR/bin/activate" || return 1

  echo "Upgrading pip ..."
  python3 -m pip install --upgrade pip || return 1

  echo "Installing: ${PACKAGES[*]} ..."
  python3 -m pip install --upgrade "${PACKAGES[@]}" || return 1

  echo "Downloading Playwright's Chromium browser (used when Google Chrome isn't installed) ..."
  python3 -m playwright install chromium || return 1

  echo "Checking imports ..."
  python3 -c "import pandas, numpy, matplotlib, seaborn, scipy, statsmodels, bs4, playwright; print('All packages import correctly.')" || return 1

  echo ""
  echo "Done. Environment: $VENV_DIR"
  echo "In each new terminal, activate it with:  source \"$VENV_DIR/bin/activate\""
  echo "Then run, e.g.:  python CSVFiles/analyser_grapher.py CSVFiles/ao3_results.csv"
}

setup_ao3_env