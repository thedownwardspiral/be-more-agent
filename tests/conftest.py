"""Runs before any test module is imported. Importing agent.py reads config.json
and opens agent.log in the repo; on the Pi those belong to the running agent.
Empty overrides keep the tests on DEFAULT_CONFIG and logging to stdout only."""
import os

os.environ["BMO_CONFIG_FILE"] = ""
os.environ["BMO_LOG_FILE"] = ""
