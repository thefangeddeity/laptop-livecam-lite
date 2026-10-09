#!/usr/bin/env bash
#
# laptop-livecam-lite -> node deployer.
#
# WHY THIS EXISTS: every artifact that actually runs tina used to live only on
# the filesystem. The viewer at /var/www/hls-livecam was hand-installed and was
# in no repo, so when vendor/hls.min.js went missing the viewer silently fell
# back to native HLS and there was no source of truth to compare against.
# This script is the one path from the tree to the box.
#
# Copies ONLY files this repo owns. It never uses --delete and never touches
# runtime state (broadcast.txt, buzz.txt, cams.json, *.bak-*), because those
# are written by the running system, not by the tree.
#
# NODE IDENTITY IS NOT IN THIS REPO. The repo is public; the tailnet name and
# the machines on it are not. Files ending .in are templates carrying
# @TAILNET_HOST@, filled from node.env (gitignored) at deploy time.
#
# /etc/hls-livecam/device.env is the ADMIN tier -- node-specific hardware
# settings, deliberately NOT deployed. device.env.example tracks its shape.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_ROOT=/var/www/hls-livecam
UNIT_DIR=/etc/systemd/system

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

if [[ ! -f "$HERE/node.env" ]]; then
  echo "FAIL  $HERE/node.env missing -- copy node.env.example and fill it in" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "$HERE/node.env"
: "${TAILNET_HOST:?node.env must set TAILNET_HOST}"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
CHANGED=0

say() { printf '  %-10s %s\n' "$1" "$2"; }
run() { if (( DRY )); then say DRY "$*"; else "$@"; fi; }

# install -C semantics by hand: only write when content differs, so an
# unchanged deploy is a true no-op and nginx/systemd are not bounced for free.
inst() { # inst <mode> <src> <dst>
  local mode=$1 src=$2 dst=$3
  [[ -f "$src" ]] || { say skip "missing $src"; return 0; }
  if cmp -s "$src" "$dst" 2>/dev/null; then say same "$dst"; return 0; fi
  say install "$dst"
  run sudo install -D -m "$mode" -o root -g root "$src" "$dst"
  CHANGED=1
}

# render <template.in> -> path under $TMP, with node identity substituted
render() {
  local src=$1 out="$TMP/$(basename "${1%.in}")"
  sed "s|@TAILNET_HOST@|$TAILNET_HOST|g" "$src" > "$out"
  if grep -q '@[A-Z_]\+@' "$out"; then
    echo "FAIL  unsubstituted placeholder in $(basename "$src"):" >&2
    grep -o '@[A-Z_]\+@' "$out" | sort -u >&2
    exit 3
  fi
  printf '%s' "$out"
}

echo "laptop-livecam-lite deploy -> $TAILNET_HOST  (dry-run=$DRY)"

echo "- api"
inst 0755 "$HERE/bin/broadcast-api"     /usr/local/bin/broadcast-api
inst 0755 "$HERE/bin/lightcv-audiocheck" /usr/local/bin/lightcv-audiocheck

echo "- web"
inst 0644 "$HERE/web/index.html"        "$WEB_ROOT/index.html"
inst 0644 "$HERE/web/vendor/hls.min.js" "$WEB_ROOT/vendor/hls.min.js"
inst 0644 "$HERE/web/ele.html"          "$WEB_ROOT/ele.html"
inst 0644 "$HERE/web/brand.png"         "$WEB_ROOT/brand.png"
inst 0644 "$HERE/web/dark.png"          "$WEB_ROOT/dark.png"
inst 0644 "$HERE/web/cams/cams.html"    "$WEB_ROOT/cams/cams.html"
# cams.json names real machines on a private tailnet, so it is runtime state
# on the box and is never shipped from here -- not even as a default.
if [[ -f "$WEB_ROOT/cams/cams.json" ]]; then
  say keep "$WEB_ROOT/cams/cams.json (runtime state, not tracked)"
else
  say WARN "$WEB_ROOT/cams/cams.json absent -- see web/cams/cams.example.json"
fi

echo "- nginx"
inst 0644 "$(render "$HERE/etc/nginx/hls-livecam.conf.in")" /etc/nginx/conf.d/hls-livecam.conf

echo "- systemd"
for u in "$HERE"/etc/systemd/*; do
  [[ -f "$u" ]] || continue
  case "$u" in
    *.in) inst 0644 "$(render "$u")" "$UNIT_DIR/$(basename "${u%.in}")" ;;
    *)    inst 0644 "$u"             "$UNIT_DIR/$(basename "$u")" ;;
  esac
done

if [[ -f /etc/hls-livecam/device.env ]]; then
  say keep "/etc/hls-livecam/device.env (admin tier, not deployed)"
else
  echo "- device.env MISSING -- seeding from example (REVIEW IT)"
  inst 0640 "$HERE/etc/hls-livecam/device.env.example" /etc/hls-livecam/device.env
  run sudo chgrp www-data /etc/hls-livecam/device.env
fi

if (( DRY )); then echo "dry run -- nothing applied"; exit 0; fi
if (( ! CHANGED )); then echo "nothing changed -- no restarts"; exit 0; fi

echo "- apply"
sudo nginx -t
sudo systemctl reload nginx          # reload, not restart: keeps live viewers up
sudo systemctl daemon-reload
sudo systemctl restart broadcast-api.service

echo "- verify"
sleep 4
base="https://$TAILNET_HOST:8443"
code=$(curl -ks -o /dev/null -w '%{http_code}' "$base/")
hls=$(curl -ks -o /dev/null -w '%{http_code}' "$base/vendor/hls.min.js")
echo "    viewer=$code  hls.min.js=$hls"
[[ "$code" == 200 && "$hls" == 200 ]] || { echo "VERIFY FAILED"; exit 1; }
systemctl is-active --quiet broadcast-api.service && echo "    broadcast-api active"
echo "done"
