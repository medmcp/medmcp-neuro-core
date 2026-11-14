#!/usr/bin/env bash
# Build and stage FreeSurfer's talairach_avi tools from the FreeSurfer source tree.
#
# Usage: talairach-avi.sh <freesurfer-source-dir> <staging-prefix> <version-string>
#
# Why this exists: the official FreeSurfer 8.2.0 Ubuntu 24 package ships without
# talairach_avi, its 4dfp helpers and the 4dfp atlases. The Ubuntu 24 build turns
# FreeSurfer's Fortran subprojects off (GCC 13), and talairach_avi is one of them.
# recon-all's -talairach step runs talairach_avi in every stream, so neither the
# official amd64 package nor our arm64 build can run recon-all -all without it.
#
# This builds the same nine programs, with the same compiler flags, that
# talairach_avi/CMakeLists.txt in FreeSurfer v8.2.0 builds (plus the legacy-Fortran
# flags GCC 13 needs), installs the six scripts it configures, and the atlases
# FreeSurfer's own MINIMAL (recon-all only) install includes. The large atlas images
# live in FreeSurfer's public git-annex; each one is fetched by its annex key and
# checked against the SHA-256 and size the key itself names.
#
# The same script runs for both architectures, so the arm64 smoke test exercises
# exactly the code path the amd64 image uses.
set -euo pipefail

src="$1/talairach_avi"
prefix="$2"
version="$3"
annex="https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects"

[[ -f "$src/CMakeLists.txt" ]] || { echo "no talairach_avi sources in $src" >&2; exit 1; }
mkdir -p "$prefix/bin" "$prefix/average" "$prefix/docs/xml"

# ── atlases: git files, plus annex objects verified against their keys ──────────
# The MINIMAL list from talairach_avi/CMakeLists.txt (install_symlinks ... average).
atlases=(
    3T18yoSchwartzReactN32_as_orig.4dfp.hdr
    3T18yoSchwartzReactN32_as_orig.4dfp.ifh
    3T18yoSchwartzReactN32_as_orig.4dfp.img
    3T18yoSchwartzReactN32_as_orig.4dfp.img.rec
    3T18yoSchwartzReactN32_as_orig.4dfp.mat
    3T18yoSchwartzReactN32_as_orig.lst
    711-2B_as_mni_average_305_mask.4dfp.hdr
    711-2B_as_mni_average_305_mask.4dfp.ifh
    711-2B_as_mni_average_305_mask.4dfp.img
    711-2B_as_mni_average_305_mask.4dfp.img.rec
    711-2C_as_mni_average_305.4dfp.hdr
    711-2C_as_mni_average_305.4dfp.ifh
    711-2C_as_mni_average_305.4dfp.img
    711-2C_as_mni_average_305.4dfp.img.rec
    711-2C_as_mni_average_305.4dfp.mat
)
# The annexed ones, pinned: file -> annex key (SHA256-s<size>--<sha256>).
declare -A keys=(
    [3T18yoSchwartzReactN32_as_orig.4dfp.img]=SHA256-s67108864--47969a534cdaad9b2ba66129b3e26b3ef15c8c0b071e16d0f6a31f6205646346
    [3T18yoSchwartzReactN32_as_orig.4dfp.img.rec]=SHA256-s201738--f3a24ea5b52db87bf1ba7b8f3f7ed2b04d4f6c5bddf0808807720ffc31d6e101
    [711-2B_as_mni_average_305_mask.4dfp.img]=SHA256-s67108864--07fd8fd7f95492b2aed699864b0524241ca661bbe92cb302f365a1a0700f4fe7
    [711-2B_as_mni_average_305_mask.4dfp.img.rec]=SHA256-s2460--d6685a10f1d727478f0604e6537aab299993f016ed9ab002b11392c48dd10389
    [711-2C_as_mni_average_305.4dfp.img]=SHA256-s67108864--c3b831fa24ac7962f32437ce52c44eb495dd3583fafb287ea1970b3d195eed5d
    [711-2C_as_mni_average_305.4dfp.img.rec]=SHA256-s187031--93b23e81247a7b3dbd910df531c6053fd8427af56abc629d7c2e7e847097d834
)

for name in "${atlases[@]}"; do
    dest="$prefix/average/$name"
    if [[ -n "${keys[$name]:-}" ]]; then
        key="${keys[$name]}"
        # The source tree's symlink must point at the pinned key: a different
        # FreeSurfer checkout would otherwise get the wrong atlas silently.
        target="$(readlink "$src/$name" || true)"
        [[ "$(basename "$target")" == "$key" ]] \
            || { echo "$name: source links to '$target', expected key $key" >&2; exit 1; }
        size="${key#SHA256-s}"; size="${size%%--*}"
        sha="${key##*--}"
        h="$(printf %s "$key" | md5sum | cut -c1-6)"   # git-annex hashdirlower
        curl -fsSL --retry 5 -o "$dest" "$annex/${h:0:3}/${h:3:3}/$key/$key"
        [[ "$(stat -c %s "$dest")" == "$size" ]] || { echo "$name: size mismatch" >&2; exit 1; }
        echo "$sha  $dest" | sha256sum -c --quiet - || { echo "$name: SHA-256 mismatch" >&2; exit 1; }
    else
        [[ -f "$src/$name" && ! -L "$src/$name" ]] || { echo "$name missing from source" >&2; exit 1; }
        install -m 0644 "$src/$name" "$dest"
    fi
done

# ── scripts (install_configured: @FS_VERSION@ substituted) ───────────────────────
for s in talairach_avi mpr2mni305 brec brec.awk t4img_4dfp tal_QC_AZS; do
    sed "s/@FS_VERSION@/${version}/g" "$src/$s" > "$prefix/bin/$s"
    chmod 0755 "$prefix/bin/$s"
done
install -m 0644 "$src/talairach_avi.help.xml" "$prefix/docs/xml/"

# ── programs: CMakeLists.txt's targets and flags (Release: -O3 -DNDEBUG) ─────────
build="$(mktemp -d)"
defs="-DHAVE_GFORTRAN -DHAVE_OPENMP -DHAVE_ZLIB -DLinux"
cflags="-O3 -DNDEBUG -fopenmp $defs -I$src"
fflags="-O3 -DNDEBUG -fopenmp $defs -I$src -ffixed-line-length-132 \
-fno-second-underscore -std=legacy -fallow-invalid-boz -fallow-argument-mismatch"
# Linked by gfortran with a static libgfortran, as FreeSurfer links them.
ldflags="-fopenmp -static-libgfortran -Wl,--gc-sections"

trx="rec.c Getifh.c endianio.c"
rms="cgauss3d.c fftsol.f imgpad.f polfit.f matopr.f eigen.f param6opr.f"
lin="t4_sub.f t4scale.c"
img="fimgreg.f imgvalm.f imgvalx.f ffind.f"

program() {  # program <name> <sources...>
    local name="$1"; shift
    local objs=()
    mkdir -p "$build/$name"
    for f in "$@"; do
        local o="$build/$name/${f%.*}.o"
        case "$f" in
            *.c) gcc $cflags -c "$src/$f" -o "$o" ;;
            *.f) gfortran $fflags -c "$src/$f" -o "$o" ;;
        esac
        objs+=("$o")
    done
    gfortran $ldflags "${objs[@]}" -o "$prefix/bin/$name"
}

program analyzeto4dfp   analyzeto4dfp.c cflip.c $trx
program 4dfptoanalyze   4dfptoanalyze.c cflip.c Inithdr.c $trx
program flip_4dfp       flip_4dfp.c cflip.c $trx
program zero_lt_4dfp    zero_lt_4dfp.c $trx
program ifh2hdr         ifh2hdr.c Inithdr.c $trx
program gauss_4dfp      gauss_4dfp.c cgauss3dd.c conc.c $trx $rms
program imgreg_4dfp     imgreg_4dfp.c $trx $rms $lin $img cflip.c
program t4imgs_4dfp     t4imgs_4dfp.c $trx $rms $lin $img cflip.c spline3dvgh.f ft4imgo.f \
                        to_711-2B.f ft4imgn.f set_rnan.f ft4ixyz.f
program compute_vox2vox compute_vox2vox.c $trx $rms $lin $img cflip.c set_rnan.f ft4imgo.f
rm -rf "$build"
echo "talairach_avi staged in $prefix"
