#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dotnet_channel="10.0"
cuda_arch="86-real;86-virtual"
install_dotnet=1
commit_push_results=0
git_username="Bench Runner"
git_email="bench-runner@users.noreply.github.com"

bench_args=()
run_choice="both"

usage() {
    cat <<'EOF'
Usage: scripts/run-bench-in-docker.sh [options] [-- <bench-main.py args...>]

Runs one-shot benchmark setup for fresh Docker clones:
1) Install .NET runtime/SDK (channel configurable)
2) Build CUDA benchmark binary
3) Run benchmark via: uv run python bench/main.py

Options:
  --dotnet-channel <channel>    .NET install channel (default: 10.0)
  --cuda-arch <arch>            CUDA arch for build script (default: 86-real;86-virtual)
  --skip-dotnet-install         Do not run dotnet installation
  --commit-push-results         Commit newest bench/results output and push
  --git-username <name>         Git user.name for commit (default: Bench Runner)
  --git-email <email>           Git user.email for commit
  -h, --help                    Show help

Examples:
  scripts/run-bench-in-docker.sh -- --samples 5 --warmup 1
  scripts/run-bench-in-docker.sh --commit-push-results -- --samples 10
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dotnet-channel)
            dotnet_channel="${2:?missing value for --dotnet-channel}"
            shift 2
            ;;
        --cuda-arch)
            cuda_arch="${2:?missing value for --cuda-arch}"
            shift 2
            ;;
        --run)
            # Forward run selection to bench script (cuda|csharp|both)
            run_choice="${2:?missing value for --run}"
            bench_args+=("--run" "${run_choice}")
            shift 2
            ;;
        --skip-dotnet-install)
            install_dotnet=0
            shift
            ;;
        --commit-push-results)
            commit_push_results=1
            shift
            ;;
        --git-username)
            git_username="${2:?missing value for --git-username}"
            shift 2
            ;;
        --git-email)
            git_email="${2:?missing value for --git-email}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            bench_args+=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ "${install_dotnet}" -eq 1 ]]; then
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "${tmp_dir}"' EXIT
    install_script="${tmp_dir}/dotnet-install.sh"

    echo "[1/3] Installing .NET runtime + SDK (channel: ${dotnet_channel})..."
    curl -fsSL https://dot.net/v1/dotnet-install.sh -o "${install_script}"
    bash "${install_script}" --channel "${dotnet_channel}" --runtime dotnet
    bash "${install_script}" --channel "${dotnet_channel}"

    export DOTNET_ROOT="${HOME}/.dotnet"
    export PATH="${DOTNET_ROOT}:${DOTNET_ROOT}/tools:${PATH}"
fi

if [[ "${run_choice}" == "csharp" ]]; then
    echo "[2/3] Skipping CUDA build (run_choice=csharp)..."
else
    echo "[2/3] Building CUDA benchmark binary..."
    bash "${repo_root}/scripts/build-cuda-standalone-test.sh" "${cuda_arch}"
fi

echo "[3/3] Running benchmark with uv..."
(
    cd "${repo_root}/bench"
    uv run python main.py --skip-cuda-build "${bench_args[@]}"
)

if [[ "${commit_push_results}" -eq 1 ]]; then
    newest_result_dir="$(ls -td "${repo_root}"/bench/results/* 2>/dev/null | head -n 1 || true)"
    if [[ -z "${newest_result_dir}" || ! -d "${newest_result_dir}" ]]; then
        echo "No bench result directory found under ${repo_root}/bench/results" >&2
        exit 3
    fi

    echo "Configuring git user as ${git_username} and committing ${newest_result_dir}..."
    git -C "${repo_root}" config user.name "${git_username}"
    git -C "${repo_root}" config user.email "${git_email}"
    git -C "${repo_root}" add "${newest_result_dir}"

    if git -C "${repo_root}" diff --cached --quiet; then
        echo "No new bench result changes to commit."
        exit 0
    fi

    commit_message="bench: add results $(basename "${newest_result_dir}")"
    git -C "${repo_root}" commit -m "${commit_message}"
    git -C "${repo_root}" push origin HEAD
fi
