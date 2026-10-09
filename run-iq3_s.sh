#!/bin/sh
cd "/home/bossman/ai/strata"
export STRATA_IQ_MT_MIN=1
exec "/home/bossman/ai/strata/.venv/bin/python" "/home/bossman/ai/strata/serve/server.py" "--engine" "strata" "--config" "/home/bossman/ai/strata/strata-iq3_s.json" "--port" "8000" "--open"
