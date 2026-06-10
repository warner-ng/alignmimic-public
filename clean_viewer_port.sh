#!/usr/bin/env bash
set -euo pipefail

ports=("$@")
if [[ ${#ports[@]} -eq 0 ]]; then
  ports=(8080-8099)
fi

kill_pids() {
  local pids="$1"
  local context="$2"

  if [[ -z "$pids" ]]; then
    return
  fi

  echo "[INFO] cleaning $context, PID(s): $(echo "$pids" | tr '\n' ' ')"
  echo "$pids" | xargs -r kill -15 || true
  sleep 1
}

find_port_pids() {
  local port="$1"

  if command -v lsof >/dev/null 2>&1; then
    lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
    return
  fi

  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :$port" 2>/dev/null \
      | awk 'NR>1 && $NF ~ /pid=/ { match($NF, /pid=([0-9]+)/, a); if (a[1] != "") print a[1] }' \
      | sort -u
    return
  fi

  echo "[ERROR] 需要 lsof 或 ss 来查找端口占用进程。" >&2
  exit 1
}

parse_ports_to_clean() {
  local input="$1"
  local start end

  if [[ "$input" == *-* ]]; then
    start="${input%-*}"
    end="${input#*-}"
    if ! [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ ]]; then
      echo "[ERROR] 非法端口范围: $input" >&2
      exit 1
    fi
    if (( start > end )); then
      echo "[ERROR] 端口范围无效: $input" >&2
      exit 1
    fi

    while (( start <= end )); do
      echo "$start"
      start=$((start + 1))
    done
    return
  fi

  if ! [[ "$input" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] 非法端口: $input" >&2
    exit 1
  fi
  echo "$input"
}

for port in "${ports[@]}"; do
  while IFS= read -r port; do
    pids="$(find_port_pids "$port")"
    if [[ -z "$pids" ]]; then
      echo "[OK] port $port is free."
      continue
    fi

    kill_pids "$pids" "port $port"
    pids="$(find_port_pids "$port")"
    if [[ -n "$pids" ]]; then
      echo "[WARN] port $port still busy, force killing PID(s): $(echo "$pids" | tr '\n' ' ')"
      echo "$pids" | xargs -r kill -9 || true
      sleep 1
    fi

    pids="$(find_port_pids "$port")"
    if [[ -n "$pids" ]]; then
      echo "[ERROR] port $port is still busy, PID(s): $(echo "$pids" | tr '\n' ' ')" >&2
      exit 1
    fi

    echo "[OK] port $port cleaned."
  done < <(parse_ports_to_clean "$port")
done
