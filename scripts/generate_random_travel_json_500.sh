#!/usr/bin/env bash
set -euo pipefail

echo "Deprecated: use scripts/generate_random_travel_json_5000.sh" >&2
exec bash scripts/generate_random_travel_json_5000.sh "$@"
