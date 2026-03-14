# Single GPU LLM Inference using Unified Virtual Memory 

## Build
- make

## Run
- chmod +x run_benchmarks.sh 
- ./run_benchmarks.sh
       

## Microbenchmarks

|     | Configuration                                                                                                                                                                                                                                                               | What it is                                          |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| M1  | Data allocated via `cudaMallocManaged` on CPU; kernel accesses it w/o any prior migration.                                                                                                                                                                                  | Worst case. Full page fault latency on every access |
| M2  | Same allocation, but `cudaMemPrefetchAsync` called on the entire buffer before the kernel launches on a separate stream.                                                                                                                                                    | Best case prefetch                                  |
| M3  | Issue `cudaMemPrefetchAsync` on a dedicated non-blocking prefetch stream, then immediately launch the kernel on the compute stream without waiting for the prefetch to complete. A `cudaStreamWaitEvent` at the start of the kernel ensures correctness but allows overlap. | More realistic prefetch                             |

## Metrics
- Kernel execution time
- Page fault count
- PCIe read bandwidth

