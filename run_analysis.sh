#!/bin/bash
# Run all strategies on 10 test episodes each, then generate analysis plot.
# Skips strategies that already have a results file in the runs directory.
# Usage: bash run_analysis.sh [SUFFIX]

set -e

SUFFIX="${1:-analysis}"
EPISODES=10
STRATEGIES=("pushover" "full_price" "skilled_seller" "haggler" "random")
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs_${SUFFIX}"

echo "============================================"
echo "  Craigslist Shop — Strategy Evaluation"
echo "============================================"
echo "  Episodes per strategy: $EPISODES"
echo "  Strategies: ${STRATEGIES[*]}"
echo "  Output: runs_${SUFFIX}/"
echo ""

cd "$(dirname "$0")/craigslist_shop"

for strategy in "${STRATEGIES[@]}"; do
    # Check if this strategy already has a results file
    if ls "$RUNS_DIR"/${strategy}_* 1>/dev/null 2>&1; then
        echo ">>> Skipping $strategy (already completed)"
    else
        echo ">>> Running $strategy ($EPISODES episodes)..."
        python test_agent.py \
            --strategy "$strategy" \
            --episodes "$EPISODES" \
            --suffix "$SUFFIX"
    fi
    echo ""
done

echo ">>> Generating analysis plot..."
cd "$SCRIPT_DIR"
python plot_results.py --suffix "$SUFFIX"

echo ""
echo "Done! Results in runs_${SUFFIX}/ and analysis/strategy_evaluation.png"
