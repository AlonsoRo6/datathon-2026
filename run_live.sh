#!/bin/bash
# Single-champion runner: one model version answers, forever, unbuffered, no sleep.
#
#   bash run_live.sh          # start (kills any other recommender first)
#   tail -f run.log           # watch
#
# The loop restarts the process if it dies; caffeinate keeps the Mac awake.
cd "$(dirname "$0")"

while true; do
  caffeinate -dimsu python3 -u starter/recommender.py >> run.log 2>&1
  code=$?
  echo "[runner] recommender exited (code $code) at $(date) — restarting in 2s" >> run.log
  sleep 2
done
