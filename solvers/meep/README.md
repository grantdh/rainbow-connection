# MEEP Configuration

- **FP64 install:** conda env meep_arm (v1.31.0)
- **FP32 install:** conda env meep-single (v1.31.0, source build)
- **Metal kernels:** ~/acme-meep/src/metal/meep_metal_fdtd.swift

## Performance (256³, M2 Max)
- MEEP FP64 CPU: 211 Mcells/s (bandwidth-saturated, OpenMP useless)
- MEEP FP32 CPU: 335 Mcells/s (1.59× over FP64)
- Metal GPU FP32: 2601 Mcells/s (7.8× over FP32 CPU)

## Key insight
One CPU core saturates DRAM bandwidth for FDTD stencils.
GPU wins via 6.7× more bandwidth, not more ALUs.
