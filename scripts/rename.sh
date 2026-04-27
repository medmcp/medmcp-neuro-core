#!/usr/bin/env bash
# One-shot rename of the placeholder package name after scaffolding from the template.
#
# Usage:
#   ./scripts/rename.sh medmcp-dicom
#
# This replaces:
#   - the PyPI/distribution name  "medmcp-template"  -> "medmcp-dicom"
#   - the Python module name      "medmcp_template"  -> "medmcp_dicom"
#   - the src directory           src/medmcp_template -> src/medmcp_dicom
#
# After running, delete this script: `rm scripts/rename.sh`.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <new-package-name>" >&2
    echo "  e.g. $0 medmcp-dicom" >&2
    exit 1
fi

NEW_DIST="$1"

if [[ ! "$NEW_DIST" =~ ^[a-z][a-z0-9-]*$ ]]; then
    echo "error: package name must be lowercase letters, digits, and hyphens (got: $NEW_DIST)" >&2
    exit 1
fi

NEW_MOD="${NEW_DIST//-/_}"

OLD_DIST="medmcp-template"
OLD_MOD="medmcp_template"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

if [[ ! -d "src/$OLD_MOD" ]]; then
    echo "error: src/$OLD_MOD not found — did you already rename?" >&2
    exit 1
fi

echo "Renaming: $OLD_DIST -> $NEW_DIST  (module: $OLD_MOD -> $NEW_MOD)"

# Find all git-tracked text files and substitute in place. Exclude this script,
# LICENSE/CHANGELOG (historical text), and CONTRIBUTING.md (contains meta-references
# to medmcp-template as the origin template repo that must not be renamed).
mapfile -t files < <(
    git ls-files \
        "*.py" "*.toml" "*.yml" "*.yaml" "*.md" "justfile" "*.cfg" "*.ini" \
        | grep -v -E "^scripts/rename\.sh$|^CHANGELOG\.md$|^CONTRIBUTING\.md$"
)

for f in "${files[@]}"; do
    if grep -q -E "$OLD_DIST|$OLD_MOD" "$f"; then
        sed -i.bak \
            -e "s/$OLD_DIST/$NEW_DIST/g" \
            -e "s/$OLD_MOD/$NEW_MOD/g" \
            "$f"
        rm "$f.bak"
        echo "  updated: $f"
    fi
done

git mv "src/$OLD_MOD" "src/$NEW_MOD" 2>/dev/null || mv "src/$OLD_MOD" "src/$NEW_MOD"
echo "  moved:   src/$OLD_MOD -> src/$NEW_MOD"

# Rename the in-package skills subdirectory (directory name not covered by sed above).
if [[ -d "src/$NEW_MOD/skills/$OLD_DIST" ]]; then
    mv "src/$NEW_MOD/skills/$OLD_DIST" "src/$NEW_MOD/skills/$NEW_DIST"
    echo "  moved:   src/$NEW_MOD/skills/$OLD_DIST -> src/$NEW_MOD/skills/$NEW_DIST"
fi

echo
echo "Done. Next steps:"
echo "  1. rm scripts/rename.sh"
echo "  2. Edit pyproject.toml (description, keywords, URLs)"
echo "  3. Replace README.md with your project's own README"
echo "  4. Implement your tools in src/$NEW_MOD/tools/ and register them in server.py"
echo "     Include a _render key in each tool's return dict with display rules + NEXT ACTION"
echo "  5. Rename src/$NEW_MOD/skills/$NEW_DIST/ to a task name (e.g. explore-data),"
echo "     update the name: field in SKILL.md to match, then write workflow + gotchas only"
echo "  6. just setup && just check"
