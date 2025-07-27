// Copyright 2025 AutoCore, Inc.
//
// Portions of this code are derived from the BEVFormer TensorRT implementation by Derry Lin:
// https://github.com/DerryHub/BEVFormer_tensorrt
//
// Original code licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified by AutoCore, Inc. in 2025.
// Original creation by Derry Lin on 2022/10/22.

#ifndef TENSORRT_OPS_ROTATEKERNEL_H
#define TENSORRT_OPS_ROTATEKERNEL_H

#include "cuda_int8.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

enum class RotateInterpolation { Bilinear, Nearest };

template <typename T>
void rotate(T *output, const T *input, const T *angle, const T *center, const int64_t *input_dims,
            RotateInterpolation interp, cudaStream_t stream);

void rotate_h2(__half2 *output, const __half2 *input, const __half *angle, const __half *center,
               const int64_t *input_dims, RotateInterpolation interp,
               cudaStream_t stream);

template <typename T>
void rotate_int8(int8_4 *output, float scale_o, const int8_4 *input,
                 float scale_i, const T *angle, const T *center,
                 const int64_t *input_dims, RotateInterpolation interp,
                 cudaStream_t stream);

#endif // TENSORRT_OPS_ROTATEKERNEL_H