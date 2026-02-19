#!/usr/bin/env bash

# ==========================================
# Script: delayed_claude_debug_ping.sh
# Description:
#   Waits DELAY seconds, then sends a fixed
#   debugging message to Claude Code via CLI.
# ==========================================

set -euo pipefail

# ---- Configurable delay (change this once) ----
DELAY=5600  # seconds

# ---- Fixed message ----
MESSAGE="You recently helped me implement a distribute=true/false kwarg into scidb.for_each(). Now, I want to make a MATLAB test pass to ensure that the functionality works. However, I'm having some difficulty: This test case in TestForEach:test_distribute_true_saves_to_multiple_rows is failing during scidb.for_each() execution: [Pasted text #5 +36 lines] With this MATLAB output: [_is_tabular_dict] FAIL: not dict or empty (type=float, len=N/A) [error] subject=1: cannot distribute ProcessedSignal: Python Error: BinderException: Binder Error: Table \"ProcessedSignal_data\" does not have a column with name \"value\". Please help me fix the distribute= functionality to make this test pass. Or maybe it's the test itself that needs fixing?"

echo "Waiting ${DELAY} seconds before pinging Claude..."
sleep "${DELAY}"

echo "Sending debugging message to Claude..."

claude "${MESSAGE}"

echo "Done."