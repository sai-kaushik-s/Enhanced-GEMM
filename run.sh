#!/usr/bin/env bash

MODE=${1:-baseline}
N=${2:-1024}
P=${3:-4}

if [ "$MODE" = "baseline" ]; then
    echo "Running baseline Python implementation"
    source .venv/bin/activate
    python3 baseline/gemm_baseline.py $N $P
elif [ "$MODE" = "optimized" ]; then
    echo "Running optimized binary"
    if [ ! -x optimized/gemm_opt ]; then
        echo "Optimized binary not found. Run make first."
        exit 1
    fi
    optimized/gemm_opt $N $P
elif [ "$MODE" = "compare" ]; then
    echo "Comparing baseline and optimized implementations for N=$N, P=$P"
    BASE_PERF_BIN="python3 baseline/gemm_baseline.py"
    BASE_PERF_FILE="baseline/gemm_baseline.py"
    BASE_CHECK_BIN="optimized/gemm_baseline"
    OPT_BIN="optimized/gemm_opt"

    if [ ! -f "$BASE_PERF_FILE" ]; then
        echo "Error: Baseline Python script not found: $BASE_PERF_FILE"
        exit 1
    fi
    if [ ! -x "$BASE_CHECK_BIN" ]; then
        echo "Error: Baseline binary not found or not executable: $BASE_CHECK_BIN"
        echo "Did you run 'make'?"
        exit 1
    fi
    if [ ! -x "$OPT_BIN" ]; then
        echo "Error: Optimized binary not found or not executable: $OPT_BIN"
        echo "Did you run 'make'?"
        exit 1
    fi

    echo "----------------------------------------"
    echo "1/3 Running Python baseline (for performance)..."
    BASE_PERF_OUT=$( { /usr/bin/time -p $BASE_PERF_BIN $N $P; } 2>&1 )

    echo "2/3 Running C baseline (for checksum verification)..."
    BASE_CHECK_OUT=$( { /usr/bin/time -p $BASE_CHECK_BIN $N $P; } 2>&1 )
    
    echo "3/3 Running Optimized binary (for performance & checksum)..."
    OPT_OUT=$( { /usr/bin/time -p $OPT_BIN $N $P; } 2>&1 )

    echo "----------------------------------------"
    echo "Analyzing results..."

    CHECK_BASE=$(echo "$BASE_CHECK_OUT" | awk -F'=' '/checksum=/ {print $2}' | awk '{print $1}')
    CHECK_OPT=$(echo "$OPT_OUT" | awk -F'=' '/checksum=/ {print $2}' | awk '{print $1}')

    TIME_BASE_INT=$(echo "$BASE_PERF_OUT" | awk -F'time=' '{print $2}' | awk '{print $1}')
    TIME_BASE_REAL=$(echo "$BASE_PERF_OUT" | awk '/^real/ {print $2}')
    TIME_OPT_INT=$(echo "$OPT_OUT" | awk -F'time=' '{print $2}' | awk '{print $1}')
    TIME_OPT_REAL=$(echo "$OPT_OUT" | awk '/^real/ {print $2}')

    if [ -z "$CHECK_BASE" ] || [ -z "$CHECK_OPT" ]; then
        echo "Error: Could not parse checksums."
        echo "Baseline (C) output: $BASE_CHECK_OUT"
        echo "Optimized output:    $OPT_OUT"
        exit 1
    fi

    if [ "$CHECK_BASE" != "$CHECK_OPT" ]; then
        echo "FAILED: Checksum mismatch!"
        echo "Baseline (C): $CHECK_BASE"
        echo "Optimized:    $CHECK_OPT"
        exit 1
    else
        echo "SUCCESS: Checksums match ($CHECK_BASE)"
    fi

    echo "----------------------------------------"
    echo "Performance Comparison..."
    
    if [ -z "$TIME_BASE_INT" ] || [ -z "$TIME_OPT_INT" ]; then
        echo "Error: Could not parse timing data."
        exit 1
    fi

    echo -e "\033[1;32mInternal Running Time:\033[0m"
    echo -e "  \033[1mPython\033[0m:    ${TIME_BASE_INT}s"
    echo -e "  \033[1mOptimized\033[0m: ${TIME_OPT_INT}s"
    echo -e "\033[1;34mTotal Running Time:\033[0m"
    echo -e "  \033[1mPython\033[0m:    ${TIME_BASE_REAL}s"
    echo -e "  \033[1mOptimized\033[0m: ${TIME_OPT_REAL}s"
    echo "----------------------------------------"

    SPEEDUP_INT=$(awk "BEGIN {if ($TIME_OPT_INT > 0) printf \"%.2f\", $TIME_BASE_INT / $TIME_OPT_INT; else print \"N/A\"}")
    SPEEDUP_REAL=$(awk "BEGIN {if ($TIME_OPT_REAL > 0) printf \"%.2f\", $TIME_BASE_REAL / $TIME_OPT_REAL; else print \"N/A\"}")

    echo -e "\033[1;33mSpeedup (Internal):\033[0m  ${SPEEDUP_INT}x"
    echo -e "\033[1;33mSpeedup (Real Time):\033[0m ${SPEEDUP_REAL}x"
elif [ "$MODE" = "scorer" ]; then
    echo "Running scorer.py for the baseline and optimized implementations for N=$N, P=$P"
    SCORER_BIN="python3 scorer.py"
    BASE_PERF_BIN="python3 baseline/gemm_baseline.py"
    BASE_PERF_FILE="baseline/gemm_baseline.py"
    OPT_BIN="optimized/gemm_opt"

    if [ ! -f "$BASE_PERF_FILE" ]; then
        echo "Error: Baseline Python script not found: $BASE_PERF_FILE"
        exit 1
    fi
    if [ ! -x "$OPT_BIN" ]; then
        echo "Error: Optimized binary not found or not executable: $OPT_BIN"
        echo "Did you run 'make'?"
        exit 1
    fi
    $SCORER_BIN \
        --baseline "$BASE_PERF_BIN $N $P" \
        --optimized "$OPT_BIN $N $P"
else
    echo "Unknown mode: $MODE"
    exit 1
fi
