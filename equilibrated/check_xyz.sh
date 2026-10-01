#!/bin/bash
# Usage: ./check_xyz.sh [root_dir]   (default: current directory)
# Checks that line 1 (atom count) + 2 == total number of lines in every .xyz/.extxyz file.

ROOT="${1:-.}"
ok=0
bad=0

while IFS= read -r -d '' f; do
    # awk counts a last line even without trailing newline; ignores whitespace on line 1
    read -r natoms total < <(awk 'NR==1{n=$1} END{print n, NR}' "$f")

    if ! [[ "$natoms" =~ ^[0-9]+$ ]]; then
        echo "BAD HEADER : $f (first line: '$natoms')"
        ((bad++))
        continue
    fi

    expected=$((natoms + 2))
    if [ "$total" -eq "$expected" ]; then
        ((ok++))
    else
        echo "MISMATCH   : $f  (natoms=$natoms, expected $expected lines, found $total)"
        ((bad++))
    fi
done < <(find "$ROOT" -type f \( -name "*.xyz" -o -name "*.extxyz" \) -print0 | sort -z)

echo "Done: $ok OK, $bad problem file(s)."
