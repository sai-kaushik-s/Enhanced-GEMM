
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

    BASELINE_FULL_OUTPUT=$(python3 baseline/gemm_baseline.py $N $P)
    OPTIMIZED_FULL_OUTPUT=$(optimized/gemm_opt $N $P)

    echo "Parsing execution times..."
    
    BASELINE_TIME=$(echo "$BASELINE_FULL_OUTPUT" | awk -F'time=' '{print $2}' | awk '{print $1}')
    OPTIMIZED_TIME=$(echo "$OPTIMIZED_FULL_OUTPUT" | awk -F'time=' '{print $2}' | awk '{print $1}')

    if [ -z "$BASELINE_TIME" ] || [ -z "$OPTIMIZED_TIME" ]; then
        echo "Error: Could not parse time from program output."
        echo "Baseline output: $BASELINE_FULL_OUTPUT"
        echo "Optimized output: $OPTIMIZED_FULL_OUTPUT"
        exit 1
    fi
    
    echo "Baseline time: $BASELINE_TIME seconds"
    echo "Optimized time: $OPTIMIZED_TIME seconds"

    SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $BASELINE_TIME / $OPTIMIZED_TIME}")
    echo "Speedup (Optimized vs Baseline): ${SPEEDUP}x"
else
    echo "Unknown mode: $MODE"
    exit 1
fi
