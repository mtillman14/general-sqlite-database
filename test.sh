#!/usr/bin/env bash

# ==========================================
# Script: delayed_claude_ping.sh
# Description:
#   Waits DELAY seconds, then sends a prompt
#   to Claude Code via CLI.
# Usage:
#   ./delayed_claude_ping.sh "Your message here"
# ==========================================

set -euo pipefail

# ---- Configurable delay (change this once) ----
DELAY=14400  # seconds

# Default message if none provided
# MESSAGE="${1:-Say 'hello' only}"
MESSAGE="Error occurred in TestTableRoundTrip/test_save_onerow_scalar_string and it did not run to completion.
    ---------
    Error ID:
    ---------
    'MATLAB:Python:PyException'
    --------------
    Error Details:
    --------------
    Error using construction>_extract_index (line 667)
    Python Error: ValueError: If using all scalar values, you must pass
    an index
    
    Error in construction>dict_to_mgr (line 448)
    
    Error in frame>__init__ (line 782)
    
    Error in database>load_all (line 1766)
    
    Error in bridge>load_and_extract (line 437)
    
    Error in TestTableRoundTrip/test_save_onerow_scalar_string (line 110)
                result = TableVar().load('subject', 4, 'session', 'A');
                
What does this mean? This is the last piece of the puzzle to make DuckDB column types when saving/loading tables work. But Id on't want to force the user to use manual indexes for these very common scenarios. So please resolve this issue so that indexes aren't needed, and column types are consistent. Then please write the column type design to docs/claude."

echo "Waiting ${DELAY} seconds before pinging Claude..."
sleep "${DELAY}"

# claude --continue

echo "Sending message to Claude:"
echo "\"${MESSAGE}\""

# Send message to Claude CLI
claude "${MESSAGE}"

echo "Done."