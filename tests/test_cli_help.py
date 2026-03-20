# tests/test_cli_help.py
import subprocess

def test_cli_help_runs():
    result = subprocess.run(["swb2_stats", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Summarize SWB2 outputs" in result.stdout