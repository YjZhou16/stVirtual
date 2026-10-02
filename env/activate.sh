#!/usr/bin/env bash
# Source after activating the Python environment used for stVirtual.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Use: source env/activate.sh' >&2
    exit 1
fi
_stvirtual_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_stvirtual_prefix="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"
_stvirtual_cuda=""
for _stvirtual_candidate in "$_stvirtual_prefix/targets/x86_64-linux" "$_stvirtual_prefix" "${CUDA_PATH:-}" "${CUDA_HOME:-}" /usr/local/cuda; do
    if [[ -f "$_stvirtual_candidate/include/cuda.h" && -f "$_stvirtual_candidate/include/nvrtc.h" && -f "$_stvirtual_candidate/include/vector_functions.h" ]]; then
        _stvirtual_cuda="$_stvirtual_candidate"
        break
    fi
done
export PYTHONPATH="$_stvirtual_root/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "$_stvirtual_cuda" ]]; then
    export CUDA_PATH="$_stvirtual_cuda"
    export CUDA_HOME="${_stvirtual_cuda%/targets/x86_64-linux}"
    export CPATH="$_stvirtual_cuda/include${CPATH:+:$CPATH}"
    printf 'stVirtual ready; CUDA headers: %s\n' "$CUDA_PATH/include"
else
    echo 'stVirtual Python path ready. CUDA development headers were not found.' >&2
fi
unset _stvirtual_root _stvirtual_prefix _stvirtual_cuda _stvirtual_candidate
