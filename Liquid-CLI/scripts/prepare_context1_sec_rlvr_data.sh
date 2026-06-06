#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONTEXT_REPO="${CONTEXT_REPO:-context-1-data-gen}"
OUTPUT_DIR="/home/work/.data/harness1/sec/context1_output"
OUTPUT_JSONL="/home/work/.data/harness1/sec/context1_sec_rlvr.jsonl"
COLLECTION="sec_lfm25_rlvr_$(date -u +%Y%m%d%H%M%S)"
IDENTITY="${SEC_IDENTITY:-}"
MAX_WORKERS=8
EXTENSION_ROUNDS=1
ENV_FILE=""
NO_INDEX=0
SKIP_GENERATION=0
OVERWRITE_JSONL=0

while (($#)); do
  case "$1" in
    --context-repo)
      CONTEXT_REPO="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --output-jsonl)
      OUTPUT_JSONL="$2"; shift 2 ;;
    --collection)
      COLLECTION="$2"; shift 2 ;;
    --identity)
      IDENTITY="$2"; shift 2 ;;
    --max-workers)
      MAX_WORKERS="$2"; shift 2 ;;
    --extension-rounds)
      EXTENSION_ROUNDS="$2"; shift 2 ;;
    --env-file)
      ENV_FILE="$2"; shift 2 ;;
    --no-index)
      NO_INDEX=1; shift ;;
    --skip-generation)
      SKIP_GENERATION=1; shift ;;
    --overwrite-jsonl)
      OVERWRITE_JSONL=1; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

if [[ ! -d "$CONTEXT_REPO/.git" ]]; then
  echo "missing_context_repo=$CONTEXT_REPO" >&2
  echo "Clone it first: git clone https://github.com/chroma-core/context-1-data-gen.git $CONTEXT_REPO" >&2
  exit 1
fi

if [[ -n "$ENV_FILE" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "missing_env_file=$ENV_FILE" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [[ "$SKIP_GENERATION" != "1" ]]; then
  REQUIRED=(ANTHROPIC_API_KEY OPENAI_API_KEY CHROMA_API_KEY CHROMA_DATABASE BASETEN_API_KEY)
  MISSING=()
  for key in "${REQUIRED[@]}"; do
    if [[ -z "${!key:-}" ]]; then
      MISSING+=("$key")
    fi
  done
  if [[ -z "$IDENTITY" ]]; then
    MISSING+=(SEC_IDENTITY)
  fi
  if ((${#MISSING[@]})); then
    echo "missing_required_env=${MISSING[*]}" >&2
    echo "SEC generation is blocked. Provide --env-file with the required keys and --identity, or pass --skip-generation if output JSON already exists." >&2
    exit 2
  fi
fi

mkdir -p "$(dirname "$OUTPUT_JSONL")" "$OUTPUT_DIR"

if [[ "$SKIP_GENERATION" != "1" ]]; then
  echo "context_repo=$CONTEXT_REPO"
  echo "output_dir=$OUTPUT_DIR"
  echo "collection=$COLLECTION"
  echo "max_workers=$MAX_WORKERS"
  echo "extension_rounds=$EXTENSION_ROUNDS"

  (
    cd "$CONTEXT_REPO"
    if command -v uv >/dev/null 2>&1; then
      uv sync --extra sec-index --extra rerank
      UV_RUN=(uv run)
    else
      UV_RUN=()
    fi
    NO_INDEX_ARGS=()
    if [[ "$NO_INDEX" == "1" ]]; then
      NO_INDEX_ARGS+=(--no-index)
    fi
    "${UV_RUN[@]}" python -m agentic_search_data_gen.domains.sec \
      --output "$OUTPUT_DIR" \
      --collection "$COLLECTION" \
      --identity "$IDENTITY" \
      --max-workers "$MAX_WORKERS" \
      --extension-rounds "$EXTENSION_ROUNDS" \
      "${NO_INDEX_ARGS[@]}"
  )
fi

CONVERT_ARGS=()
if [[ "$OVERWRITE_JSONL" == "1" ]]; then
  CONVERT_ARGS+=(--overwrite)
fi

python Liquid-CLI/scripts/convert_context1_sec_to_rlvr_jsonl.py \
  --input-dir "$OUTPUT_DIR" \
  --output-jsonl "$OUTPUT_JSONL" \
  "${CONVERT_ARGS[@]}"

echo "sec_rlvr_jsonl=$OUTPUT_JSONL"
