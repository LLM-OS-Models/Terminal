#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HARNESS_REPO="${HARNESS_REPO:-/home/work/.projects/LLM-OS-Models/harness-1}"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/harness1}"
BROWSECOMP_REPO_URL="${BROWSECOMP_REPO_URL:-https://github.com/texttron/BrowseComp-Plus}"
BROWSECOMP_DIR="${BROWSECOMP_DIR:-$DATA_ROOT/external/BrowseComp-Plus}"
UPDATE=0
DECRYPT_BROWSECOMP=1

while (($#)); do
  case "$1" in
    --harness-repo)
      HARNESS_REPO="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      BROWSECOMP_DIR="$DATA_ROOT/external/BrowseComp-Plus"
      shift 2
      ;;
    --update)
      UPDATE=1
      shift
      ;;
    --no-decrypt-browsecomp)
      DECRYPT_BROWSECOMP=0
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$DATA_ROOT/external" "$DATA_ROOT/sft_data"

if [[ ! -f "$HARNESS_REPO/training/generate_sft_data.py" ]]; then
  echo "Harness-1 source tree not found: $HARNESS_REPO" >&2
  echo "Clone it first or set HARNESS_REPO." >&2
  exit 1
fi

if [[ -d "$BROWSECOMP_DIR/.git" ]]; then
  if [[ "$UPDATE" == "1" ]]; then
    git -C "$BROWSECOMP_DIR" pull --ff-only
  fi
else
  git clone --depth 1 "$BROWSECOMP_REPO_URL" "$BROWSECOMP_DIR"
fi

BROWSECOMP_ANSWERS="$BROWSECOMP_DIR/data/browsecomp_plus_decrypted.jsonl"
BROWSECOMP_QUERIES="$BROWSECOMP_DIR/topics-qrels/queries.tsv"
BROWSECOMP_QRELS_GOLD="$BROWSECOMP_DIR/topics-qrels/qrel_golds.txt"
BROWSECOMP_QRELS_EVIDENCE="$BROWSECOMP_DIR/topics-qrels/qrel_evidence.txt"

if [[ "$DECRYPT_BROWSECOMP" == "1" ]]; then
  if [[ -f "$BROWSECOMP_ANSWERS" && -f "$BROWSECOMP_QUERIES" ]]; then
    echo "browsecomp_decrypted=ready"
  else
    echo "browsecomp_decrypted=building"
    (
      cd "$BROWSECOMP_DIR"
      python scripts_build_index/decrypt_dataset.py \
        --output data/browsecomp_plus_decrypted.jsonl \
        --generate-tsv topics-qrels/queries.tsv
    )
  fi
fi

echo "harness_repo=$HARNESS_REPO"
echo "browsecomp_dir=$BROWSECOMP_DIR"
echo "sft_data_dir=$DATA_ROOT/sft_data"
echo "browsecomp_answers=$BROWSECOMP_ANSWERS"
echo "browsecomp_queries=$BROWSECOMP_QUERIES"
echo "browsecomp_qrels_gold=$BROWSECOMP_QRELS_GOLD"
echo "browsecomp_qrels_evidence=$BROWSECOMP_QRELS_EVIDENCE"

if [[ ! -f "$HARNESS_REPO/.env.local" ]]; then
  cat <<EOF
missing_env_local=$HARNESS_REPO/.env.local
Create it from $HARNESS_REPO/.env.example and point BrowseComp+ paths at:
  BROWSECOMPPLUS_QUERIES_PATH=$BROWSECOMP_QUERIES
  BROWSECOMPPLUS_QRELS_GOLD_PATH=$BROWSECOMP_QRELS_GOLD
  BROWSECOMPPLUS_QRELS_EVIDENCE_PATH=$BROWSECOMP_QRELS_EVIDENCE
  BROWSECOMPPLUS_ANSWERS_PATH=$BROWSECOMP_ANSWERS

Trajectory generation also needs OPENAI_API_KEY, CHROMA_API_KEY, CHROMA_DATABASE,
HUGGINGFACE_TOKEN, TINKER_API_KEY placeholders accepted by harness/config.py.
EOF
fi
