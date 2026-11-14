# syntax=docker/dockerfile:1
#
# medmcp-neuro-core — neuro-imaging tool stack as a fixed-environment MCP stdio server.
# GPU (all torch): HD-BET (skull_strip) + antspyx (registration) + FastSurfer
# (segment_brain) + ECLARE / SMORE (super_resolve). Launched by the core via
# `docker run -i --device nvidia.com/gpu=all`.
#
# segment_brain uses FastSurfer's seg-only pipeline (FastSurferVINN) — no FreeSurfer
# license required, torch-based, so it rides the same cu128 build and is arm64-capable.
#
# FreeSurfer 8.2.0 (run_recon_all, and FastSurfer's surface pipeline behind
# segment_brain(surfaces=True)) is built in the stages below and copied into the
# image as /opt/freesurfer. Those stages depend only on their own inputs -- the
# pinned versions and checksums here and the scripts in docker/ -- never on the
# Python environment or the source, so a build reuses them from the cache (CI keeps
# them in a registry cache of their own, see .github/workflows/images.yml) until
# FreeSurfer or its build recipe changes. Multi-arch:
#
#   linux/amd64  the official FreeSurfer 8.2.0 Ubuntu 24 package (sha256-pinned),
#                with talairach_avi added (the package lacks it, see
#                docker/talairach-avi.sh) and pruned to what recon-all needs.
#   linux/arm64  the same package with every program replaced by an aarch64 build
#                from the FreeSurfer v8.2.0 source (there is no official Linux arm64
#                build): ITK 5.4.0 and FreeSurfer are compiled here, a cold build of
#                about 30 min to an hour on 20 cores.
#
# This is a MODIFIED version of FreeSurfer and is marked as such (image label, NOTICE,
# /opt/freesurfer/MEDMCP-MODIFIED.txt), as the FreeSurfer licence requires. Both
# architectures download the 8.9 GB package once (a local copy can be served with
# --build-arg FS_DEB_URL=http://127.0.0.1:<port>/... and --network host; the
# SHA-256 is checked either way). No FreeSurfer licence key is, or may ever be, part
# of this image: the tools read the user's own key from the workspace at run time.
ARG BASE_IMAGE=medmcp-base:dev

ARG FS_VERSION=8.2.0
ARG FS_DEB_URL=https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/8.2.0/freesurfer_ubuntu24-8.2.0_amd64.deb
ARG FS_DEB_SHA256=a52996ed5147e5b7f1bd537c8405537cc2b133ae7075616b0c38975986a236a6
# The commit the official 8.2.0 package was built from (its build-stamp.txt).
ARG FS_GIT_TAG=v8.2.0
ARG FS_GIT_COMMIT=d932c45b7941662ea380a05efef580568b98d41a
ARG ITK_GIT_TAG=v5.4.0

# ── 1. The official package, extracted and pruned ───────────────────────────────
# Architecture-independent file work, so it runs on the build platform natively.
FROM --platform=$BUILDPLATFORM ubuntu:24.04 AS fs-package
ARG FS_DEB_URL
ARG FS_DEB_SHA256
ARG FS_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --chmod=0755 docker/prune-freesurfer.sh /usr/local/bin/
# Download, verify, then extract without the largest unused parts (the rest is
# pruned by the script). One layer, so the 8.9 GB package never lands in one.
RUN set -eux; \
    curl -fSL --retry 5 -o /tmp/fs.deb "$FS_DEB_URL"; \
    echo "$FS_DEB_SHA256  /tmp/fs.deb" | sha256sum -c -; \
    p="./usr/local/freesurfer/$FS_VERSION"; \
    mkdir /tmp/x; \
    dpkg-deb --fsys-tarfile /tmp/fs.deb | tar -x -C /tmp/x \
        --exclude="$p/models" --exclude="$p/python/lib" --exclude="$p/python/packages" \
        --exclude="$p/trctrain" --exclude="$p/average/mult-comp-cor" \
        --exclude="$p/average/samseg" --exclude="$p/subjects/cvs_avg35*" \
        --exclude="$p/mni-1.4" --exclude="$p/lib/tktools" \
        "$p"; \
    rm /tmp/fs.deb; \
    mv "/tmp/x/usr/local/freesurfer/$FS_VERSION" /opt/freesurfer; \
    rm -rf /tmp/x; \
    prune-freesurfer.sh /opt/freesurfer; \
    # recon-all records this version in every subject's logs: mark the distribution
    # as modified there too (the arm64 stage also corrects the architecture).
    grep -q "^setenv FS_RECON_VERSION '8.2.0 (freesurfer-linux-ubuntu24_x86_64-8.2.0-20260314-d932c45)'" \
        /opt/freesurfer/bin/recon-all; \
    sed -i "s/^\(setenv FS_RECON_VERSION '8.2.0 (freesurfer-linux-ubuntu24_x86_64-8.2.0-20260314-d932c45\))'/\1-medmcp-modified)'/" \
        /opt/freesurfer/bin/recon-all

# ── 2. talairach_avi, built from source (both architectures) ────────────────────
FROM ubuntu:24.04 AS talairach
ARG FS_GIT_TAG
ARG FS_GIT_COMMIT
ARG FS_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git gcc gfortran libc6-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "$FS_GIT_TAG" --filter=blob:none --sparse \
        https://github.com/freesurfer/freesurfer.git /src/freesurfer \
    && cd /src/freesurfer && git sparse-checkout set talairach_avi \
    && test "$(git rev-parse HEAD)" = "$FS_GIT_COMMIT"
COPY --chmod=0755 docker/talairach-avi.sh /usr/local/bin/
# The legacy Fortran compiles warn a lot; show the log only when the build fails.
RUN talairach-avi.sh /src/freesurfer /stage "$FS_VERSION-medmcp" > /tmp/talairach.log 2>&1 \
    || { grep -v -i -e warning -e '^ *|' /tmp/talairach.log | tail -60; exit 1; }

# ── 3a. amd64: the official programs ────────────────────────────────────────────
FROM scratch AS fs-amd64
COPY --from=fs-package /opt/freesurfer /opt/freesurfer
COPY --from=talairach /stage/ /opt/freesurfer/
COPY docker/MEDMCP-MODIFIED.amd64.txt /opt/freesurfer/MEDMCP-MODIFIED.txt
# recon-all prints this as its version and records it in every subject's logs.
COPY docker/build-stamp.amd64.txt /opt/freesurfer/build-stamp.txt

# ── 3b. arm64: ITK 5.4.0 + FreeSurfer v8.2.0 from source ────────────────────────
# As in the build notes (2026-11-13/14): ITK from source against the system zlib;
# FreeSurfer with MINIMAL=OFF and GUIs off, its macOS-arm64 switch (-DARM64), the
# x86-only -msse2 flags removed, the Fortran subprojects kept on with the aarch64
# libgfortran/BLAS/LAPACK paths and legacy-Fortran flags, the system Python
# standing in for the bundled fspython, and a Martinos-internal copy step disabled.
FROM ubuntu:24.04 AS build-arm64
ARG FS_GIT_TAG
ARG FS_GIT_COMMIT
ARG ITK_GIT_TAG
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran cmake git ca-certificates file xxd \
        python3 python3-dev python3-setuptools python3-numpy \
        zlib1g-dev libblas-dev liblapack-dev libgomp1 libx11-dev libxmu-dev \
        libglu1-mesa-dev libeigen3-dev tcsh bc perl libjpeg-dev libtiff-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch "$ITK_GIT_TAG" https://github.com/InsightSoftwareConsortium/ITK.git itk \
    && cmake -S itk -B itk-build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/opt/itk \
        -DITK_USE_SYSTEM_EIGEN=OFF -DITK_USE_SYSTEM_ZLIB=ON -DCMAKE_CXX_FLAGS="-Wno-error" \
    && (cmake --build itk-build -j"$(nproc)" > itk-make.log 2>&1 \
        || { grep -m 40 -i "error" itk-make.log; exit 1; }) \
    && cmake --install itk-build > /dev/null \
    && rm -rf itk-build itk itk-make.log
RUN git clone --depth 1 --branch "$FS_GIT_TAG" https://github.com/freesurfer/freesurfer.git \
    && test "$(git -C freesurfer rev-parse HEAD)" = "$FS_GIT_COMMIT"
WORKDIR /src/freesurfer
# Source changes for the aarch64 build (each one fails the build if it no longer applies).
RUN set -eux; \
    grep -q 'if(HOST_OS MATCHES "Ubuntu24")' python/fsbindings/CMakeLists.txt; \
    sed -i 's/if(HOST_OS MATCHES "Ubuntu24")/if(FALSE)/' python/fsbindings/CMakeLists.txt; \
    # x86-only flags hard-coded in gems/ and python/gems/ (and in dcm2niix's x86
    # branch). Each of these checks fails the build if it no longer holds.
    files=$(grep -rl --include=CMakeLists.txt -e "-msse2 -mfpmath=sse" .); \
    test -n "$files"; \
    echo "$files" | xargs sed -i 's/ -msse2 -mfpmath=sse//g'; \
    if grep -rq --include=CMakeLists.txt -e "-mfpmath=sse" .; then exit 1; fi; \
    python3 - <<'PY'
p = "CMakeLists.txt"
s = open(p).read()
old = """        ## while does not compile...
        set(BUILD_FORTRAN OFF)
     endif()
     ## otherwise using gcc 8"""
assert old in s, "Fortran switch-off block not found"
s = s.replace(old, """        ## arm64 build: Fortran kept on (talairach_avi)
     endif()
     ## otherwise using gcc 8""", 1)
open(p, "w").write(s)
PY
# The build looks for its bundled Python at a fixed path; point it at the system one.
RUN v=$(grep -oE 'set\(FSPYTHON_VERSION *"?[0-9.]+' CMakeLists.txt | grep -oE '[0-9.]+$' | head -1); \
    v=${v:-3.12}; mkdir -p /opt/fspkgs/fspython/$v/bin \
    && ln -sf /usr/bin/python3 /opt/fspkgs/fspython/$v/bin/python$v
ENV FS_PACKAGES_DIR=/opt/fspkgs
RUN mkdir build && cd build && cmake .. -DMINIMAL=OFF -DBUILD_GUIS=OFF \
        -DINSTALL_PYTHON_DEPENDENCIES=OFF -DDISTRIBUTE_FSPYTHON=OFF \
        -DCMAKE_INSTALL_PREFIX=/opt/freesurfer-src -DCMAKE_BUILD_TYPE=Release \
        -DITK_DIR=/opt/itk/lib/cmake/ITK-5.4 -DWARNING_AS_ERROR=OFF -DINTEGRATE_SAMSEG=OFF \
        -DGFORTRAN_LIBRARIES=/usr/lib/gcc/aarch64-linux-gnu/13/libgfortran.a \
        -DBLAS_LIBRARIES=/usr/lib/aarch64-linux-gnu/blas/libblas.a \
        -DLAPACK_LIBRARIES=/usr/lib/aarch64-linux-gnu/lapack/liblapack.a \
        -DCMAKE_Fortran_FLAGS='-std=legacy -fallow-invalid-boz -fallow-argument-mismatch' \
        -DCMAKE_C_FLAGS='-DARM64 -DPNG_ARM_NEON_OPT=0' \
        -DCMAKE_CXX_FLAGS='-DARM64 -DPNG_ARM_NEON_OPT=0' > /src/fs-cmake.log 2>&1 \
    || { tail -60 /src/fs-cmake.log; exit 1; }
RUN cd build && (make -j"$(nproc)" > /src/fs-make.log 2>&1 \
        || { grep -B2 -A6 "error" /src/fs-make.log | head -80; exit 1; })

FROM build-arm64 AS fs-arm64-assemble
COPY --from=fs-package /opt/freesurfer /opt/freesurfer
COPY --chmod=0755 docker/assemble-arm64.sh /usr/local/bin/
RUN assemble-arm64.sh /opt/freesurfer /src/freesurfer/build \
 && sed -i "s/ubuntu24_x86_64-8.2.0-20260314-d932c45-medmcp-modified/ubuntu24_aarch64-8.2.0-20260314-d932c45-medmcp-modified/" \
        /opt/freesurfer/bin/recon-all \
 && grep -q "ubuntu24_aarch64-8.2.0-20260314-d932c45-medmcp-modified" /opt/freesurfer/bin/recon-all
COPY --from=talairach /stage/ /opt/freesurfer/
COPY docker/MEDMCP-MODIFIED.arm64.txt /opt/freesurfer/MEDMCP-MODIFIED.txt
# recon-all prints this as its version and records it in every subject's logs.
COPY docker/build-stamp.arm64.txt /opt/freesurfer/build-stamp.txt
# Licences of what the arm64 programs link statically, beside FreeSurfer's own
# third-party licences in docs/: ITK 5.4.0 (Apache-2.0) and reference BLAS/LAPACK.
ADD --checksum=sha256:aac73b3148f6d1d7111dbca32099f68d26c644c6813ae1e4f05f6579aa2663fe \
    https://raw.githubusercontent.com/InsightSoftwareConsortium/ITK/v5.4.0/LICENSE \
    /opt/freesurfer/docs/license.itk.txt
ADD --checksum=sha256:3bb4f458523ef5a5df0dedf461638cab5d4416f3b91837b2e81c4067ea628807 \
    https://raw.githubusercontent.com/InsightSoftwareConsortium/ITK/v5.4.0/NOTICE \
    /opt/freesurfer/docs/notice.itk.txt
RUN cp /usr/share/doc/libblas-dev/copyright /opt/freesurfer/docs/license.blas.txt \
 && cp /usr/share/doc/liblapack-dev/copyright /opt/freesurfer/docs/license.lapack.txt \
 && chmod 0644 /opt/freesurfer/docs/license.itk.txt /opt/freesurfer/docs/notice.itk.txt

# Only the assembled tree, without the build environment beneath it, so the stage the
# image copies from (and CI caches) is the size of /opt/freesurfer.
FROM scratch AS fs-arm64
COPY --from=fs-arm64-assemble /opt/freesurfer /opt/freesurfer

# The FreeSurfer tree for this architecture. CI builds this target first, with its own
# registry cache (see .github/workflows/images.yml).
FROM fs-${TARGETARCH} AS freesurfer

# ── 4. The stack ────────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS runtime

# Stack metadata for one-click install/discovery (read via `docker inspect`).
#
# job: a core that runs stack jobs starts `medmcp-neuro-core-job call <tool>
# <approved arguments>` in a container of its own for the calls that take about an
# hour -- run_recon_all, and segment_brain with surfaces=True (FastSurfer's surface
# pipeline) -- so they outlive the chat and the backend pool. The tools ask for a
# job only for those calls; every other segment_brain call runs as before.
# watch_sec: how long the core watches a new job for an immediate failure.
LABEL org.medmcp.stack='{"name": "medmcp-neuro-core", "gpu": true, "tool_timeout_sec": 7200, "skills_path": "/app/src/medmcp_neuro_core/skills", "job": {"command": ["medmcp-neuro-core-job", "call"], "tools": ["run_recon_all", "segment_brain"], "watch_sec": 10}}'
LABEL org.medmcp.freesurfer.version="8.2.0" \
      org.medmcp.freesurfer.modified="true"

# torch is pinned to the CUDA 12.8 (cu128) build (see pyproject), so it runs
# natively on any host driver >= R570 (Turing through Blackwell) — no forward-compat
# shim needed. libgomp1 is needed by antspyx/torch OpenMP paths.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# antspyx publishes aarch64 wheels only for 0.4.2; every release from 0.5 on is
# x86_64/macos only. We keep 0.6.3 on both architectures rather than pinning an
# older ANTs on arm64 — differing ANTs versions per architecture would mean
# registration output that depends on the machine it ran on. So on arm64 antspyx
# is compiled from its sdist, which builds ITK+ANTs via CMake and needs a full
# C++ toolchain (without it: "CMAKE_CXX_COMPILER not set, after EnableLanguage").
# This makes arm64 builds substantially slower; amd64 still installs the wheel and
# is unaffected.
RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            build-essential cmake git python3-dev \
            zlib1g-dev libpng-dev libjpeg-dev libtiff-dev && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Trust extra CA certs at build time behind a TLS-intercepting (MITM) proxy so
# uv/pip/git fetch through it. Drop the proxy root CA as a *.crt into ./certs/
# (gitignored; empty = no-op — CI / non-proxied builds add nothing). UV_NATIVE_TLS
# makes uv use the system trust store. Runtime is offline, so no production impact.
COPY certs/ /usr/local/share/ca-certificates/medmcp-extra/
RUN update-ca-certificates
ENV UV_NATIVE_TLS=1

WORKDIR /app

# On arm64 this compiles antspyx (ITK+ANTs) from source, which by default runs one
# compile job per core. ITK's template-heavy translation units are memory-hungry, so
# on a small runner the peak exceeds RAM and the job is killed with no diagnostic at
# all — no failing step, no log. Leave it unset for local builds on a large machine
# (full parallelism); CI passes a cap on the arm64 leg only.
ARG CMAKE_BUILD_PARALLEL_LEVEL=""

# Frozen install from the committed lock (build-time network; runtime offline).
# Dependencies only, and before the source is copied. On arm64 this single layer
# is ~36 minutes of a ~40 minute build (antspyx compiles from its sdist), so its
# cache key has to be pyproject.toml + uv.lock and nothing else. With `COPY src`
# above it, editing one line of Python recompiled ITK and ANTs from scratch.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "${CMAKE_BUILD_PARALLEL_LEVEL}" ]; then \
        export CMAKE_BUILD_PARALLEL_LEVEL MAKEFLAGS="-j${CMAKE_BUILD_PARALLEL_LEVEL}"; \
        echo "capping build parallelism at ${CMAKE_BUILD_PARALLEL_LEVEL}"; \
    fi; \
    uv sync --frozen --no-dev --no-install-project \
 && find /app/.venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
 && find /app/.venv -name '*.a' -delete

# Python downloaders (HD-BET, FastSurfer checkpoints) use requests/urllib, which
# trust certifi's bundle — not the system store — so they also need pointing at the
# updated bundle to fetch through a MITM proxy. Harmless without a proxy CA, and the
# runtime is offline so there is no production impact.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Model weights come from third-party hosts (Zenodo for HD-BET, Deep-MI for
# FastSurfer) that intermittently answer 5xx. Unretried, a single bad response kills
# the entire image build — a Zenodo 504 on the HD-BET weights failed CI while the same
# URL served fine seconds later, and the sibling neuro-cancer stack hit the equivalent
# NVIDIA 504 three builds running. Retry with linear backoff.
RUN printf '%s\n' '#!/bin/sh' \
      'n=0' \
      'until "$@"; do' \
      '  n=$((n+1))' \
      '  [ "$n" -ge 5 ] && { echo "retry: failed after $n attempts: $*" >&2; exit 1; }' \
      '  echo "retry: attempt $n failed; sleeping $((n*15))s" >&2' \
      '  sleep $((n*15))' \
      'done' \
    > /usr/local/bin/retry && chmod +x /usr/local/bin/retry

# Bake HD-BET weights so skull_strip runs with --network none (no runtime download).
RUN retry /app/.venv/bin/python -c "from HD_BET.checkpoint_download import maybe_download_parameters; maybe_download_parameters()"

# ── FastSurfer (segment_brain) ────────────────────────────────────────────────
# Whole-brain segmentation via FastSurferVINN, seg-only (no FreeSurfer license).
# FastSurfer pins torch==2.7.1 — the SAME version /app/.venv uses (pinned in
# pyproject) — so its seg deps install into that ONE venv instead of a second one,
# sharing a single ~8 GB torch + CUDA stack. run_fastsurfer.sh runs with
# --py /app/.venv/bin/python (FASTSURFER_PYTHON below); subprocess-isolated either way.
ARG FASTSURFER_VERSION=v2.5.4
# Everything below in one layer so the cleanup (.git, bytecode, static libs) actually
# shrinks the image instead of just shadowing files in an earlier layer.
# seg-only: drop scikit-sparse — a recon_surf/surface-pipeline dep (needs SuiteSparse,
# and the pinned version isn't published) that is never imported on the seg path.
# Bake the FastSurferVINN (asegdkt) checkpoints (--vinn) and the corpus-callosum ones
# (--cc): FastSurfer 2.5's surface pipeline (segment_brain surfaces=True) requires the
# aseg with the corpus callosum that its CC module writes. CerebNet and HypVINN are
# never run (segment_brain passes --no_cereb --no_hypothal), so --all would download
# them for nothing. Offline at runtime.
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone --depth 1 --branch ${FASTSURFER_VERSION} \
        https://github.com/Deep-MI/FastSurfer.git /opt/FastSurfer \
 && grep -viE '^\s*scikit-sparse' /opt/FastSurfer/requirements.txt > /tmp/fs-seg-req.txt \
 && uv pip install --python /app/.venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match -r /tmp/fs-seg-req.txt \
 && retry env PYTHONPATH=/opt/FastSurfer /app/.venv/bin/python \
        /opt/FastSurfer/FastSurferCNN/download_checkpoints.py --vinn --cc \
 && rm -rf /opt/FastSurfer/.git \
 && find /app/.venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
 && find /app/.venv -name '*.a' -delete

# ── FreeSurfer 8.2.0 (run_recon_all; segment_brain surfaces) ──────────────────
# Late in the file so a FreeSurfer change never invalidates the long dependency
# layers above. What its scripts and programs need at run time: tcsh/csh for the
# scripts, bc, the OpenMP runtime, libtiff/libcrypt/libgfortran that the programs
# link, PyYAML for the rca-config helpers (fspython is the system Python 3), and on
# amd64 the system ITK 5.3 the official programs are linked against (the arm64 build
# links ITK statically).
ARG TARGETARCH
RUN set -eux; \
    pkgs="tcsh bc file libgomp1 libtiff6 libcrypt1 zlib1g libgfortran5 python3-yaml"; \
    if [ "$TARGETARCH" = amd64 ]; then pkgs="$pkgs libinsighttoolkit5.3"; fi; \
    apt-get update && apt-get install -y --no-install-recommends $pkgs \
    && rm -rf /var/lib/apt/lists/*
COPY --from=freesurfer /opt/freesurfer /opt/freesurfer
COPY licenses/FreeSurfer-LICENSE.txt /opt/freesurfer/LICENSE-FreeSurfer.txt
# FreeSurfer's environment is NOT set image-wide: FREESURFER_HOME and its bin/ on
# PATH would change what FastSurfer, ANTs and the other tools find. The tools that
# run FreeSurfer build it for their own child processes (tools/freesurfer/_env.py).
# Here: every program must find its libraries, on this architecture, now rather than
# in the middle of a one-hour run; recon-all must start; and the tree must never
# carry a licence key.
RUN set -eu; \
    missing=$(find /opt/freesurfer -type f -perm -u+x -exec sh -c \
        'for f; do file -b "$f" | grep -q "^ELF" && ldd "$f" 2>&1 | sed -n "s|^\t*\(.*not found\)|$f: \1|p"; done; true' _ {} +); \
    if [ -n "$missing" ]; then echo "$missing" | head -40; exit 1; fi; \
    for k in /opt/freesurfer/license.txt /opt/freesurfer/.license; do test ! -e "$k"; done; \
    test -s /opt/freesurfer/MEDMCP-MODIFIED.txt; \
    export FREESURFER_HOME=/opt/freesurfer FS_V8_XOPTS=0 PATH=/opt/freesurfer/bin:$PATH; \
    for p in recon-all talairach_avi imgreg_4dfp t4imgs_4dfp mri_convert mris_make_surfaces \
             mri_ca_register AntsN4BiasFieldCorrectionFs mri_mc mris_place_surface \
             mris_remesh mri_cc mris_sample_parc mri_segstats; do command -v "$p" > /dev/null; done; \
    recon-all -version

# The source, and the project itself, last: everything above depends only on the
# lock, so a code change now rebuilds seconds of work instead of the whole image.
# --inexact because FastSurfer's requirements were pip-installed into this venv
# and are deliberately absent from uv.lock; a plain sync would prune them.
COPY src ./src
# Third-party attributions (bundled weights, templates, atlases) travel with the image.
COPY NOTICE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --inexact

# ECLARE and SMORE (super_resolve) come from the lock like any other dependency —
# they are self-supervised, so there are no weights to bake. pyproject overrides
# their stale numpy/scipy pins (ECLARE's scipy==1.11.4, degrade's numpy==1.23.5) to
# run on the locked versions, so import their CLI entry points here to fail the
# build, not the first tool call, if that ever stops working. Importing matplotlib
# also builds its font cache now rather than on the first offline run.
RUN /app/.venv/bin/python -c \
    "import eclare.main, smore.train, smore.test, matplotlib.pyplot" \
 && test -x /app/.venv/bin/run-eclare \
 && test -x /app/.venv/bin/smore-train \
 && test -x /app/.venv/bin/smore-test

# Bake the MNI152 template so register_to_template runs with --network none. Both
# variants: the skull-stripped (desc-brain) one is what the registration skill uses
# after skull_strip, so omitting it would leave that path broken offline. Populated
# through the tool's own resolver, so the cache location and filenames cannot drift
# from the code that reads them.
RUN retry /app/.venv/bin/python -c \
    "from medmcp_neuro_core.tools._template import get_mni152_1mm; get_mni152_1mm(skull_stripped=False); get_mni152_1mm(skull_stripped=True)"

# Bake the lesion-load atlases (lesion_load_by_atlas) for --network none, through the
# same resolver the tool reads them with. Only atlases cleared for redistribution are
# registered (Pandora CC BY 4.0, HCP1065 CC BY-SA 4.0, Schaefer 2018 MIT); each is
# fetched from a pinned URL and SHA-256-checked, and lands with its ATTRIBUTION.txt and
# upstream licence file in /root/.medmcp_neuro_core/atlases/<name>/. Preparing Pandora
# splits its 4-D volume and briefly needs ~2.5 GB RAM.
RUN retry /app/.venv/bin/python -c \
    "from medmcp_neuro_core.tools._atlas import prepare_all; prepare_all()"

ENV PATH=/opt/FastSurfer:/app/.venv/bin:$PATH \
    FASTSURFER_HOME=/opt/FastSurfer \
    FASTSURFER_PYTHON=/app/.venv/bin/python \
    PYTHONPATH=/opt/FastSurfer \
    UV_NO_SYNC=1

# tini reaps processes and forwards signals; stdio passes through. The MCP server
# is the default COMMAND rather than part of the ENTRYPOINT so the same image runs
# the background jobs too:
#   docker run ... <image>                                       -> stdio MCP server
#   docker run ... <image> medmcp-neuro-core-job call <tool> '<arguments>'
ENTRYPOINT ["tini", "--"]
CMD ["medmcp-neuro-core"]
