#pragma once

#ifdef FLASH_GPGPU_SIM_OMP

#ifndef FLASH_GPGPU_SIM_OMP_MAX_THREADS
#define FLASH_GPGPU_SIM_OMP_MAX_THREADS 32
#endif

#include <omp.h>
#include <array>
#include <memory>
#endif