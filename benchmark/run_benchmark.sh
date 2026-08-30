#!/bin/bash
# Run each benchmark config as a separate process for isolation
set -e
cd "$(dirname "$0")/.."
MEM_RATIO=0.40
RESULTS_FILE="$(dirname "$0")/results.json"
PYTHON="/home/hh/miniconda3/envs/lut_moe_cu124/bin/python3"

RESULTS=()
for cfg in 16bit_lossless 8bit_blocklut progressive_nestedlut; do
    echo ""
    echo "========================================"
    echo "  Running: $cfg (device_mem_ratio=$MEM_RATIO)"
    echo "========================================"
    echo ""
    sleep 5

    # Run isolated
    OUTPUT=$(timeout 600 $PYTHON benchmark/run_single.py $cfg $MEM_RATIO 2>&1)
    RET=$?

    if [ $RET -ne 0 ]; then
        echo "FAILED (exit=$RET): $cfg"
        echo "$OUTPUT" | tail -5
        RESULTS+=("{\"name\":\"$cfg\",\"failed\":true}")
        continue
    fi

    # Extract JSON result line
    JSON=$(echo "$OUTPUT" | grep '^{' | tail -1)
    if [ -n "$JSON" ]; then
        RESULTS+=("$JSON")
        echo "$JSON"
    fi

    # Clear GPU memory between runs
    $PYTHON -c "
import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()
print('GPU cleared')
" 2>/dev/null
    sleep 10
done

# Print summary
echo ""
echo "========================================"
echo "  SUMMARY"
echo "========================================"
for r in "${RESULTS[@]}"; do
    echo "$r" | $PYTHON -c "
import sys,json
try:
    d=json.load(sys.stdin)
    n=d.get('name','?')
    if d.get('failed'):
        print(f\"  {n:30s} FAILED\")
    else:
        ttft=d.get('avg_ttft_ms',0)
        tpot=d.get('avg_tpot_ms',0)
        print(f\"  {n:30s} TTFT={ttft:8.0f}ms  TPOT={tpot:8.0f}ms\")
except: pass
"
done

# Save combined
echo "${RESULTS[@]}" > "$RESULTS_FILE"
echo ""
echo "Saved to $RESULTS_FILE"
