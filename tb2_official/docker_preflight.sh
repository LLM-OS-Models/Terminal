#!/usr/bin/env bash
set -euo pipefail

if ! docker info >/dev/null 2>&1; then
  if [[ "${TB2_DOCKER_GROUP_REEXEC:-0}" != "1" ]] \
    && getent group docker | grep -Eq "[:,]${USER}(,|$)"; then
    exec sg docker -c "TB2_DOCKER_GROUP_REEXEC=1 $(printf '%q' "$0")"
  fi

  if sudo -n docker info >/dev/null 2>&1; then
    echo "Docker is reachable only through sudo, but Harbor runs Docker as the current user." >&2
    echo "Start a new login shell after adding ${USER} to the docker group," >&2
    echo "or run this command through: sg docker -c '$0'" >&2
    exit 3
  fi

  echo "Docker daemon is not reachable by the current user." >&2
  exit 1
fi

DOCKER=(docker)

"${DOCKER[@]}" info --format 'server={{.ServerVersion}} storage={{.Driver}}'
"${DOCKER[@]}" compose version

if sudo -n unshare -m true >/dev/null 2>&1; then
  echo "Local mount-namespace creation is permitted."
else
  echo "Note: local unshare is blocked; a host/remote Docker daemon may still work."
fi

if ! "${DOCKER[@]}" run --rm hello-world >/dev/null; then
  echo "Docker could not extract an image and run a container." >&2
  echo "For nested Docker, recreate the outer workload as privileged/unconfined." >&2
  echo "Alternatively, provide a working host Docker socket or a separate Docker runner." >&2
  exit 2
fi
echo "Docker preflight passed: daemon, compose, layer extraction, and container run work."
