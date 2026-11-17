# Upstream Project History and Citations

FlashGPU-Sim builds on GPGPU-Sim and retains the upstream copyright and
license notices in [COPYRIGHT](../../COPYRIGHT). Current build and workload
instructions are in the [FlashGPU-Sim README](../../README.md).

## Contributors

GPGPU-Sim was created by Tor Aamodt's research group at the University of
British Columbia. Many have directly contributed to development of GPGPU-Sim
including: Tor Aamodt, Wilson W.L. Fung, Ali Bakhoda, George Yuan, Ivan Sham,
Henry Wong, Henry Tran, Andrew Turner, Aaron Ariel, Inderpret Singh, Tim
Rogers, Jimmy Kwa, Andrew Boktor, Ayub Gubran Tayler Hetherington and others.

AccelWattch (introduced in GPGPU-Sim 4.2.0) was developed by researchers at
Northwestern University, Purdue University, and the University of British Columbia.
Contributors to AccelWattch include Nikos Hardavellas's research group at Northwestern University:
Vijay Kandiah; Tor Aamodt's research group at the University of British Columbia: Scott Peverelle;
and Timothy Rogers's research group at Purdue University: Mahmoud Khairy, Junrui Pan, and Amogh Manjunath.

The interconnection network includes BookSim, developed by Bill Dally's
research group at Stanford. AccelWattch incorporates the McPAT power model.

## Upstream Citation Guidance

The following citation guidance is retained from the upstream distribution;
use the citations relevant to the inherited components used in your work.

If you use GPGPU-Sim 4.0 in your research, please cite:

Mahmoud Khairy, Zhesheng Shen, Tor M. Aamodt, Timothy G Rogers.
Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling.
In proceedings of the 47th IEEE/ACM International Symposium on Computer Architecture (ISCA),
May 29 - June 3, 2020.

If you use CuDNN or PyTorch support (execution-driven simulation), checkpointing or our new debugging tool for functional
simulation errors in GPGPU-Sim for your research, please cite:

Jonathan Lew, Deval Shah, Suchita Pati, Shaylin Cattell, Mengchi Zhang, Amruth Sandhupatla,
Christopher Ng, Negar Goli, Matthew D. Sinclair, Timothy G. Rogers, Tor M. Aamodt
Analyzing Machine Learning Workloads Using a Detailed GPU Simulator, arXiv:1811.08933,
https://arxiv.org/abs/1811.08933

If you use the Tensor Core model in GPGPU-Sim or GPGPU-Sim's CUTLASS Library
for your research please cite:

Md Aamir Raihan, Negar Goli, Tor Aamodt,
Modeling Deep Learning Accelerator Enabled GPUs, arXiv:1811.08309,
https://arxiv.org/abs/1811.08309

If you use the AccelWattch power model in your research, please cite:

Vijay Kandiah, Scott Peverelle, Mahmoud Khairy, Junrui Pan, Amogh Manjunath, Timothy G. Rogers, Tor M. Aamodt, and Nikos Hardavellas. 2021.
AccelWattch: A Power Modeling Framework for Modern GPUs. In MICRO54: 54th Annual IEEE/ACM International Symposium on Microarchitecture
(MICRO ’21), October 18–22, 2021, Virtual Event, Greece.

If you use the support for CUDA dynamic parallelism in your research, please cite:

Jin Wang and Sudhakar Yalamanchili, Characterization and Analysis of Dynamic
Parallelism in Unstructured GPU Applications, 2014 IEEE International Symposium
on Workload Characterization (IISWC), November 2014.

If you use figures plotted using AerialVision in your publications, please cite:

Aaron Ariel, Wilson W. L. Fung, Andrew Turner, Tor M. Aamodt, Visualizing
Complex Dynamics in Many-Core Accelerator Architectures, In Proceedings of the
IEEE International Symposium on Performance Analysis of Systems and Software
(ISPASS), pp. 164-174, White Plains, NY, March 28-30, 2010.
