#!/usr/bin/env bash
# Build the instrumented llama.cpp used by Argine2 (Windows x64, from Linux with MinGW-w64).
#
#   cd llama.cpp            # a clone at the pinned commit with patches 0002, 0001, 0003 applied
#   bash ../Argine2/tools/build_llama_research.sh [OUTDIR]
#
# Pinned llama.cpp commit: 254b177307b14d055bbe4c050c9c441a1e06d8a4 (2026-10-02 09:10 +0200),
# the last commit before `ggml : add alloc_buffer_n to buffer type interface` (#23671), which
# changes the struct the expert store's buffer type fills in.
#
# Flags that matter (Argine v1 README, "Compilare la build patchata"):
#   -DGGML_OPENMP=OFF         official Windows releases do not use OpenMP; the OpenMP build was 1.74x slower
#   static link               otherwise Anaconda's libstdc++ DLLs are picked up at run time
#   AVX/AVX2/FMA/F16C/BMI2    the reference CPU (i5-1340P) has no AVX-512
set -euo pipefail
OUT="${1:-../Argine2/llama-research}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cat > /tmp/mingw.cmake <<'TC'
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc-posix)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++-posix)
set(CMAKE_RC_COMPILER x86_64-w64-mingw32-windres)
set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
TC
FLAGS=(-DCMAKE_TOOLCHAIN_FILE=/tmp/mingw.cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
       -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON
       -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_OPENMP=OFF
       -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
       "-DCMAKE_EXE_LINKER_FLAGS=-static -static-libgcc -static-libstdc++")
cmake -B build-win -G Ninja "${FLAGS[@]}"
ninja -C build-win llama-server llama-perplexity llama-quantize llama-imatrix
mkdir -p "$OUT"
cp build-win/bin/llama-{server,perplexity,quantize,imatrix}.exe "$OUT/"
# only system DLLs may appear here
x86_64-w64-mingw32-objdump -p "$OUT/llama-server.exe" | grep "DLL Name"
python3 - "$OUT" "$(git rev-parse HEAD)" "$(x86_64-w64-mingw32-g++-posix --version | head -1)" "${FLAGS[*]}" <<'PY'
import hashlib, json, sys, datetime, pathlib
out, commit, cc, flags = sys.argv[1:5]
d = pathlib.Path(out)
info = {"llama_cpp_commit": commit, "llama_cpp_repo": "https://github.com/ggml-org/llama.cpp",
        "patches": ["0002-argine-expert-store.patch", "argine-store.{h,cpp}", "argine-experts.{h,cpp}",
                    "0001-argine-route-log.patch", "0003-argine-research-trace.patch", "argine-research.{h,cpp}"],
        "compiler": cc, "cmake_flags": flags,
        "built_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "binaries": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.exe"))}}
(d / "BUILD-INFO.json").write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info["binaries"], indent=2))
PY
