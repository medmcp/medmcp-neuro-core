#!/usr/bin/env bash
# Turn the (pruned) official x86_64 FreeSurfer tree into an arm64 one.
#
# Usage: assemble-arm64.sh <FREESURFER_HOME> <freesurfer-build-dir>
#
# Every x86-64 program in the tree is replaced by the aarch64 program of the same
# name from the source build; one without an aarch64 build is removed (the prune
# step already removed the known ones, so a removal here is reported). Scripts and
# data stay as the official package has them. The build's own `make install` is not
# used: its layout differs from the package's, and the package is the reference.
set -euo pipefail
fs="$1"
build="$2"

replaced=0
removed=()
while IFS= read -r -d '' f; do
    file -b "$f" | grep -q '^ELF 64-bit.*x86-64' || continue
    name="$(basename "$f")"
    mapfile -t hits < <(find "$build" -path '*/CMakeFiles' -prune -o \
        -type f -name "$name" -perm -u+x -print)
    arm=""
    for h in "${hits[@]}"; do
        if file -b "$h" | grep -q '^ELF 64-bit.*aarch64'; then arm="$h"; break; fi
    done
    if [[ -n "$arm" ]]; then
        install -m 0755 "$arm" "$f"
        strip --strip-unneeded "$f"
        replaced=$((replaced + 1))
    else
        rm -f "$f"
        removed+=("${f#"$fs"/}")
    fi
done < <(find "$fs" -type f -print0)

echo "replaced $replaced x86-64 programs with aarch64 builds"
if ((${#removed[@]})); then
    echo "removed ${#removed[@]} x86-64 files without an aarch64 build:"
    printf '  %s\n' "${removed[@]}"
fi
# Nothing x86 may remain: it would fail with "exec format error" at run time.
if find "$fs" -type f -exec file -b {} + | grep -q 'x86-64'; then
    echo "x86-64 files remain in $fs" >&2
    exit 1
fi
