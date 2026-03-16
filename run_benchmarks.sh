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
    run_output=$(./$b)
    echo "$run_output"

    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-um-gpu-page-faults=true \
        --cuda-um-cpu-page-faults=true \
        -o ${b}_profile \
        --force-overwrite=true \
        ./$b > /dev/null

    echo ""    

    nsys stats ${b}_profile.nsys-rep > /dev/null

    echo "Page Fault and Migration Information:"
    echo "--------------------------------------------"
    output=$(nsys stats --quiet -r um_total_sum ${b}_profile.sqlite 2>/dev/null)

    if [ -z "$output" ]; then
        echo "No Unified Memory page faults occurred"
    else
        numbers_line=$(echo "$output" | tail -n 1)
        nums=($(grep -Eo '[0-9]+(\.[0-9]+)?' <<< "$numbers_line"))
        count=${#nums[@]}
        
        if [ "$count" -eq 4 ]; then
            htod_mb=${nums[0]}
            dtoh_mb=${nums[1]}
            cpu_pfs=${nums[2]}
            gpu_pfs=${nums[3]}
        elif [ "$count" -eq 3 ]; then
            htod_mb=${nums[0]}
            dtoh_mb=0
            cpu_pfs=${nums[1]}
            gpu_pfs=${nums[2]}
        fi

        echo "CPU Page Faults: $cpu_pfs"
        echo "GPU Page Faults: $gpu_pfs"
        echo ""
    
        echo "HtoD migration size: $htod_mb MB"
        echo "DtoH migration size: $dtoh_mb MB"
        echo ""

        kernel_time=$(echo "$run_output" | grep "Kernel Time" | awk '{print $(NF-1)}')

        htod_bandwidth=$(echo "$htod_mb / 1024 / ($kernel_time / 1000)" | bc -l)
        dtoh_bandwidth=$(echo "$dtoh_mb / 1024 / ($kernel_time / 1000)" | bc -l)
        
        printf "Approx PCIe Bandwidth for HtoD Migrations: %.2f GB/s\n" "$htod_bandwidth"
        printf "Approx PCIe Bandwidth for DtoH Migrations: %.2f GB/s\n" "$dtoh_bandwidth"

    fi

echo ""

done

