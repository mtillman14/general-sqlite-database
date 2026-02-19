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
DELAY=25200  # seconds

# Default message if none provided
MESSAGE="${1:-Say 'hello' only}"

echo "Waiting ${DELAY} seconds before pinging Claude..."
sleep "${DELAY}"

echo "Sending message to Claude:"
echo "\"${MESSAGE}\""

# Send message to Claude CLI
claude "${MESSAGE}"

echo "Done."