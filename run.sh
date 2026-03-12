#!/bin/bash

set -euo pipefail

WALLET="${1:?Usage: $0 <wallet_address> [options...]}"
shift

PROCESS_NAME="./kalien"
DIR="."
MAX_JOBS=1
SALT="0x1"
NOSUBMIT=0
KALIEN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --process-name) PROCESS_NAME="$2"; shift 2 ;;
        --dir)          DIR="$2";          shift 2 ;;
        --jobs)         MAX_JOBS="$2";     shift 2 ;;
        --salt)         SALT="$2";         shift 2 ;;
        --nosubmit)     NOSUBMIT=1;        shift   ;;
        *)              KALIEN_ARGS+=("$1"); shift  ;;
    esac
done

API="https://kalien.xyz/api"
SUBMIT_LOG="$DIR/submit.log"

mkdir -p "$DIR"

if [[ "$PROCESS_NAME" != */* ]]; then
    PROCESS_NAME="./$PROCESS_NAME"
fi
if [ ! -x "$PROCESS_NAME" ]; then
    echo "[ERROR] Cannot find executable: $PROCESS_NAME" >&2
    exit 1
fi

declare -a JOB_PIDS=()
declare -a JOB_SEED_IDS=()
declare -a JOB_SEED_HEXES=()
declare -a JOB_SEED_DECS=()

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cleanup() {
    log "[STOP] Cleaning up all child processes..."
    trap '' INT TERM
    kill -- -$$ 2>/dev/null || true
    for pid in "${JOB_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    log "[STOP] Done."
}
trap cleanup EXIT INT TERM

submit_tape() {
    local tape="$1" seed_id="$2"
    local score
    score=$(basename "$tape" | sed 's/.*_\([0-9]*\)\.tape$/\1/') || { log "[SUBMIT] Cannot parse score from $tape"; return; }

    if [ "$NOSUBMIT" -eq 1 ]; then
        log "[SUBMIT] nosubmit mode — skipping $tape (score=$score)"
        return
    fi

    if grep -qF "OK     seed_id=$seed_id" "$SUBMIT_LOG" 2>/dev/null; then
        local prev_score
        prev_score=$(grep "OK     seed_id=$seed_id" "$SUBMIT_LOG" | sed 's/.*score=\([0-9]*\).*/\1/' | sort -n | tail -1)
        if [ "${score}" -le "${prev_score:-0}" ]; then
            log "[SUBMIT] Skipping $tape (score=$score already submitted or beaten)"
            return
        fi
    fi

    log "[SUBMIT] $tape  score=$score  seed_id=$seed_id"
    local response rc ts job_id
    response=$(curl -sf -X POST \
        "$API/proofs/jobs?claimant=${WALLET}&seed_id=${seed_id}" \
        -H "Content-Type: application/octet-stream" \
        --data-binary "@$tape" 2>&1) && rc=0 || rc=$?

    ts=$(date '+%Y-%m-%d %H:%M:%S')
    if [ $rc -eq 0 ]; then
        job_id=$(python3 -c "import json; d=json.loads('$response'); print(d.get('job',{}).get('jobId','?'))" 2>/dev/null || echo "?")
        log "[SUBMIT] OK  job=$job_id"
        echo "$ts  OK     seed_id=$seed_id  score=$score  job=$job_id  $tape" >> "$SUBMIT_LOG"
    else
        log "[SUBMIT] FAILED (rc=$rc): $response"
        echo "$ts  FAILED seed_id=$seed_id  score=$score  rc=$rc  $response" >> "$SUBMIT_LOG"
    fi
}

CURRENT_SEED_HEX=""
CURRENT_SEED_ID=""
CURRENT_SEED_DEC=""

fetch_seed() {
    local seed_json is_null
    seed_json=$(curl -sf "$API/seed/current") || return 1
    is_null=$(python3 -c "import json; d=json.loads('$seed_json'); print(not d.get('indexed', False) or d['seed'] is None)")
    if [ "$is_null" = "True" ]; then return 1; fi
    CURRENT_SEED_HEX=$(python3 -c "import json; d=json.loads('$seed_json'); print(hex(d['seed']))")
    CURRENT_SEED_ID=$(python3 -c  "import json; d=json.loads('$seed_json'); print(d['seed_id'])")
    CURRENT_SEED_DEC=$(python3 -c  "import json; d=json.loads('$seed_json'); print(d['seed'])")
}

reap_finished_jobs() {
    local -a alive_pids=() alive_seeds=() alive_hexes=() alive_decs=()
    for i in "${!JOB_PIDS[@]}"; do
        local pid="${JOB_PIDS[$i]}"
        local sid="${JOB_SEED_IDS[$i]}"
        local shex="${JOB_SEED_HEXES[$i]}"
        local sdec="${JOB_SEED_DECS[$i]}"
        if kill -0 "$pid" 2>/dev/null; then
            alive_pids+=("$pid")
            alive_seeds+=("$sid")
            alive_hexes+=("$shex")
            alive_decs+=("$sdec")
        else
            wait "$pid" 2>/dev/null || true
            log "[JOB] pid=$pid seed_id=$sid finished — scanning for best tape"
            local best_tape
            best_tape=$(ls "$DIR"/${sid}_${sdec}_*.tape 2>/dev/null | \
                awk -F'[_.]' '{print $(NF-1), $0}' | sort -n | tail -1 | awk '{print $2}') || true
            if [ -n "$best_tape" ]; then
                submit_tape "$best_tape" "$sid"
            else
                log "[JOB] No tape found for seed_id=$sid"
            fi
        fi
    done
    JOB_PIDS=("${alive_pids[@]+"${alive_pids[@]}"}")
    JOB_SEED_IDS=("${alive_seeds[@]+"${alive_seeds[@]}"}")
    JOB_SEED_HEXES=("${alive_hexes[@]+"${alive_hexes[@]}"}")
    JOB_SEED_DECS=("${alive_decs[@]+"${alive_decs[@]}"}")
}

seed_already_running() {
    local sid="$1"
    for s in "${JOB_SEED_IDS[@]+"${JOB_SEED_IDS[@]}"}"; do
        [ "$s" = "$sid" ] && return 0
    done
    return 1
}

launch_job() {
    local seed_hex="$1" seed_id="$2" seed_dec="$3"
    local logfile="$DIR/${seed_id}_${seed_dec}.log"
    local out_prefix="$DIR/${seed_id}_${seed_dec}"

    log "[JOB] Launching  seed_id=$seed_id  seed=$seed_dec  log=$logfile"
    log "[JOB] CMD: $PROCESS_NAME --seed $seed_hex --out $out_prefix --salt $SALT ${KALIEN_ARGS[*]+"${KALIEN_ARGS[*]}"}"

    (
        "$PROCESS_NAME" \
            --seed "$seed_hex" \
            --out  "$out_prefix" \
            --salt "$SALT" \
            "${KALIEN_ARGS[@]+"${KALIEN_ARGS[@]}"}" \
        2>&1 | tee -a "$logfile"
    ) &

    local pid=$!

    (
        local last_submitted=""
        while kill -0 "$pid" 2>/dev/null; do
            local best_tape
            best_tape=$(ls "$DIR"/${seed_id}_${seed_dec}_*.tape 2>/dev/null | \
                awk -F'[_.]' '{print $(NF-1), $0}' | sort -n | tail -1 | awk '{print $2}') || true
            if [ -n "$best_tape" ] && [ "$best_tape" != "$last_submitted" ]; then
                submit_tape "$best_tape" "$seed_id"
                last_submitted="$best_tape"
            fi
            sleep 5
        done
    ) &
    JOB_PIDS+=("$pid")
    JOB_SEED_IDS+=("$seed_id")
    JOB_SEED_HEXES+=("$seed_hex")
    JOB_SEED_DECS+=("$seed_dec")
    log "[JOB] Started  pid=$pid"
}

log "[START] wallet=$WALLET  jobs=$MAX_JOBS  dir=$DIR  nosubmit=$NOSUBMIT"
log "[START] process=$PROCESS_NAME  salt=$SALT  args: ${KALIEN_ARGS[*]+"${KALIEN_ARGS[*]}"}"

while true; do
    reap_finished_jobs
    if fetch_seed; then
        log "[SEED] seed=$CURRENT_SEED_HEX  seed_id=$CURRENT_SEED_ID  running_jobs=${#JOB_PIDS[@]}"

        if seed_already_running "$CURRENT_SEED_ID"; then
            : # busy
        elif [ "${#JOB_PIDS[@]}" -lt "$MAX_JOBS" ]; then
            launch_job "$CURRENT_SEED_HEX" "$CURRENT_SEED_ID" "$CURRENT_SEED_DEC"
        else
            log "[SCHED] All $MAX_JOBS job slots busy — waiting for a slot..."
        fi
    else
        log "[SEED] Could not fetch seed (null or network error), retrying..."
    fi

    sleep 60
done