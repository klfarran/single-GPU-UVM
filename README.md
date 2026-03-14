# Single GPU LLM Inference using Unified Virtual Memory 

-  Build: make 
- Run: ./microbench

- Profile for page faults: 
    - -- nsys profile --trace=cuda,osrt --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --capture-range=cudaProfilerApi ./microbench
    - nsys stats report.nsys-rep
