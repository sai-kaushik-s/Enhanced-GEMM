
#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-baseline}
N=${2:-1024}
P=${3:-4}

if [ "$MODE" = "baseline" ]; then
    echo "Running baseline python implementation"
    python3 baseline/gemm_baseline.py $N $P
elif [ "$MODE" = "optimized" ]; then
    echo "Running optimized binary"
    if [ ! -x optimized/gemm_opt ]; then
        echo "Optimized binary not found. Run make first."
        exit 1
    fi
    optimized/gemm_opt $N $P
elif [ "$MODE" = "compare" ]; then
    echo "Comparing baseline and optimized implementations"

    if [ ! -x optimized/gemm_opt ]; then
        echo "Optimized binary not found. Run make first."
        exit 1
    fi

    echo "Running baseline..."
    BASELINE_FULL_OUTPUT=$( { /usr/bin/time -p python3 baseline/gemm_baseline.py $N $P; } 2>&1 )
    echo "Running optimized..."
    OPTIMIZED_FULL_OUTPUT=$( { /usr/bin/time -p optimized/gemm_opt $N $P; } 2>&1 )

    echo "Parsing execution times..."

    BASELINE_TIME=$(echo "$BASELINE_FULL_OUTPUT" | awk -F'time=' '{print $2}' | awk '{print $1}')
    OPTIMIZED_TIME=$(echo "$OPTIMIZED_FULL_OUTPUT" | awk -F'time=' '{print $2}' | awk '{print $1}')

    BASELINE_REAL=$(echo "$BASELINE_FULL_OUTPUT" | awk '/^real/ {print $2}')
    OPTIMIZED_REAL=$(echo "$OPTIMIZED_FULL_OUTPUT" | awk '/^real/ {print $2}')

    if [ -z "$BASELINE_TIME" ] || [ -z "$OPTIMIZED_TIME" ]; then
        echo "Error: Could not parse internal time from program output."
        echo "Baseline output: $BASELINE_FULL_OUTPUT"
        echo "Optimized output: $OPTIMIZED_FULL_OUTPUT"
        exit 1
    fi

    if [ -z "$BASELINE_REAL" ] || [ -z "$OPTIMIZED_REAL" ]; then
        echo "Error: Could not parse real time from time command."
        echo "Baseline output: $BASELINE_FULL_OUTPUT"
        echo "Optimized output: $OPTIMIZED_FULL_OUTPUT"
        exit 1
    fi

    echo "Baseline internal time: ${BASELINE_TIME}s"
    echo "Optimized internal time: ${OPTIMIZED_TIME}s"
    echo "Baseline real time: ${BASELINE_REAL}s"
    echo "Optimized real time: ${OPTIMIZED_REAL}s"

    SPEEDUP_INTERNAL=$(awk "BEGIN {printf \"%.2f\", $BASELINE_TIME / $OPTIMIZED_TIME}")
    SPEEDUP_REAL=$(awk "BEGIN {printf \"%.2f\", $BASELINE_REAL / $OPTIMIZED_REAL}")

    echo "Speedup (internal timing): ${SPEEDUP_INTERNAL}x"
    echo "Speedup (overall real time): ${SPEEDUP_REAL}x"
else
    echo "Unknown mode: $MODE"
    exit 1
fi
