#!/bin/bash

set -e

# Remove old NSight files if present
rm -f m1_profile.nsys-rep m1_profile.sqlite
rm -f m2_profile.nsys-rep m2_profile.sqlite
rm -f m3_profile.nsys-rep m3_profile.sqlite

BENCHES=("m1" "m2" "m3")

echo ""
echo "Running UVM Microbenchmarks"
echo "--------------------------------------------"

for b in "${BENCHES[@]}"; do
    echo ""
    echo "Running $b..."
    ./$b

    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-um-gpu-page-faults=true \
        --cuda-um-cpu-page-faults=true \
        -o ${b}_profile \
        --force-overwrite=true \
        ./$b > /dev/null

    echo ""    

    nsys stats ${b}_profile.nsys-rep > /dev/null

    echo "Page Fault Stats:"
    output=$(nsys stats --quiet -r um_total_sum ${b}_profile.sqlite 2>/dev/null)

    if [ -z "$output" ]; then
        echo "No Unified Memory page faults occurred"
    else
        echo "$output"
    fi

echo ""

done