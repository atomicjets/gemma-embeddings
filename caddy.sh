#!/usr/bin/env bash
set -euo pipefail
BIND=127.0.0.1
LB=18090

echo "[+] Checking for ss"
if ! command -v ss >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y >/dev/null 2>&1 || true
  apt-get install -y iproute2 >/dev/null 2>&1 || true
fi

echo "[+] Checking for Caddy"
if ! command -v caddy >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "https://caddyserver.com/api/download?os=linux&arch=amd64" -o /usr/local/bin/caddy
  elif command -v wget >/dev/null 2>&1; then
    wget -qO /usr/local/bin/caddy "https://caddyserver.com/api/download?os=linux&arch=amd64"
  else
    echo "[+] Installing curl"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y >/dev/null 2>&1 || true
    apt-get install -y curl >/dev/null 2>&1
    curl -fsSL "https://caddyserver.com/api/download?os=linux&arch=amd64" -o /usr/local/bin/caddy
  fi
  chmod +x /usr/local/bin/caddy
fi

echo "[+] Writing /tmp/Caddyfile"
cat >/tmp/Caddyfile <<'EOF'
http://127.0.0.1:18090 {
  bind 127.0.0.1

  reverse_proxy {
    lb_policy least_conn
    to 127.0.0.1:18080
    to 127.0.0.1:18081
    to 127.0.0.1:18082
    to 127.0.0.1:18083

    transport http {
      read_timeout 120s
      write_timeout 120s
      dial_timeout 10s
    }
  }
}
EOF

echo "[+] Starting Caddy on ${BIND}:${LB}"
nohup caddy run --config /tmp/Caddyfile --adapter caddyfile >/var/log/caddy.log 2>&1 &
sleep 2

echo "[+] Checking listener:"
if command -v ss >/dev/null 2>&1; then
  ss -ltnp | grep -F "${BIND}:${LB}" || echo "Caddy may still be starting… see /var/log/caddy.log"
else
  if command -v netstat >/dev/null 2>&1; then
    netstat -tlnp 2>/dev/null | grep -F "${BIND}:${LB}" || echo "Caddy may still be starting… see /var/log/caddy.log"
  else
    echo "Neither ss nor netstat available; tail /var/log/caddy.log for status."
  fi
fi