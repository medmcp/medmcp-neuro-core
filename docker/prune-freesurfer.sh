#!/usr/bin/env bash
# Cut an extracted FreeSurfer 8.2.0 package down to what recon-all needs.
#
# Usage: prune-freesurfer.sh <FREESURFER_HOME>
#
# The package is 18 GB; recon-all's classic stream uses a small part of it. What
# goes, and why:
#   - GUIs and viewers (freeview, tk*), MATLAB-runtime tools (subfield and nuclei
#     segmentation), diffusion and tractography (dmri_*, trctrain), FS-FAST,
#     MATLAB sources, the bundled FSL/AFNI/DCMTK copies: not recon-all, and none of
#     them has an aarch64 build. Removing them on both architectures keeps the two
#     images' program sets identical.
#   - the bundled Python 3.8 environment and its packages (TensorFlow, PyTorch) and
#     the deep-learning models: used by the v8 stream (SynthSeg, SynthStrip,
#     SynthMorph) and other Synth* tools, which this stack does not run (it runs
#     recon-all -no-v8). The two Python helpers recon-all does call (rca-config,
#     rca-config2csh) need only PyYAML; fspython is pointed at the system Python.
#   - group-analysis and other non-recon-all atlases (mult-comp-cor, samseg, the
#     MNI152 2009c template, Yeo, mideface, thalamic-nuclei and claustrum atlases)
#     and every subject but fsaverage (recon-all maps the Brodmann-area labels from
#     fsaverage).
#   - the MNI/MINC tools (x86-only Perl+C; recon-all's bias correction uses ANTs
#     N4). Their one other use is recon-all's last-resort Talairach fallback
#     (mritotal), which therefore is not available: a scan whose talairach_avi
#     registration fails its QA check with both atlases stops recon-all.
set -euo pipefail
fs="$1"
[[ -x "$fs/bin/recon-all" ]] || { echo "$fs is not a FreeSurfer tree" >&2; exit 1; }
cd "$fs"

rm -rf diffusion fsfast matlab trctrain tktools models mni mni-1.4 \
       lib/tktools lib/tcl lib/images \
       python/lib python/packages python/include python/share python/requirements-*.txt

rm -rf average/mult-comp-cor average/samseg average/mni_icbm152_nlin_asym_09c \
       average/Yeo_Brainmap_MNI152 average/mideface-atlas average/ThalamicNuclei \
       average/claustrum_seg

find subjects -mindepth 1 -maxdepth 1 ! -name fsaverage ! -name README -exec rm -rf {} +

# Programs without a portable build (GUI toolkits, MATLAB runtime, VTK, bundled
# third-party tools): the list of 8.2.0 programs the aarch64 build does not produce.
for p in \
    3dvolreg.afni SegmentSubfieldsT1Longitudinal SegmentThalamicNuclei \
    SegmentThalamicNuclei_DTI connected_components dcmdjpeg.fs dcmdrle.fs \
    dissection_photo dmri_AnatomiCuts dmri_coloredFA dmri_extractSurfaceMeasurements \
    dmri_groupByEndpoints dmri_match dmri_neighboringRegions dmri_projectEndPoints \
    dmri_saveHistograms dmri_stats_ac fiducials_calibration fiducials_correction \
    flirt.fsl freeview fs_spmreg.glnxa64 fslmaths.fsl fslorient.fsl gdcmconv.fs \
    irepifitvol.glnx64 mask_extraction mri_make_bem_surfaces mri_probedicom \
    mri_vessel_segment mri_vol2label mris2rgb mris_curvature2image \
    mris_extract_values mris_image2vtk mris_mesh_subdivide mris_multimodal \
    mris_multimodal_surface_placement mris_profileClustering mris_surf2vtk \
    mris_volmask_vtk nmovie_qt retrospective_correction segmentNuclei segmentSubject \
    segmentSubjectT1T2_autoEstimateAlveusML segmentSubjectT1_autoEstimateAlveusML \
    segmentSubjectT2_autoEstimateAlveusML slicetimer.fsl stim_polar streamlineFilter \
    testOrientationPlanesFromParcellation tkmedit tkregister2 tksurfer trk_tools
do
    rm -f "bin/$p"
done

# fspython runs $FREESURFER_HOME/python/bin/python3: the system interpreter.
rm -rf python/bin
mkdir -p python/bin
ln -s /usr/bin/python3 python/bin/python3
du -sh "$fs"
