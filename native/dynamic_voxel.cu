#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

void check_cuda_float(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32.");
}

void check_cuda_long(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must be int64.");
}

__device__ __forceinline__ float softplus_stable(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return expf(x);
    }
    return log1pf(expf(x));
}

__device__ __forceinline__ float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(x);
    return z / (1.0f + z);
}

__device__ __forceinline__ bool ray_box_interval_xy(
    float base_x,
    float base_y,
    float dir_x,
    float dir_y,
    float* tmin_out,
    float* tmax_out) {
    float tmin = -1.0e20f;
    float tmax = 1.0e20f;
    const float eps = 1.0e-8f;

    if (fabsf(dir_x) < eps) {
        if (base_x < -1.0f || base_x > 1.0f) {
            return false;
        }
    } else {
        float t0 = (-1.0f - base_x) / dir_x;
        float t1 = (1.0f - base_x) / dir_x;
        if (t0 > t1) {
            const float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
    }

    if (fabsf(dir_y) < eps) {
        if (base_y < -1.0f || base_y > 1.0f) {
            return false;
        }
    } else {
        float t0 = (-1.0f - base_y) / dir_y;
        float t1 = (1.0f - base_y) / dir_y;
        if (t0 > t1) {
            const float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
    }

    if (tmax <= tmin) {
        return false;
    }
    *tmin_out = tmin;
    *tmax_out = tmax;
    return true;
}

__device__ __forceinline__ int level_resolution(int level, int res0, int res1, int res2) {
    if (level == 0) {
        return res0;
    }
    if (level == 1) {
        return res1;
    }
    return res2;
}

__device__ __forceinline__ const int64_t* level_index_ptr(
    int level,
    const int64_t* index0,
    const int64_t* index1,
    const int64_t* index2) {
    if (level == 0) {
        return index0;
    }
    if (level == 1) {
        return index1;
    }
    return index2;
}

__device__ __forceinline__ int64_t lookup_leaf(
    float x,
    float y,
    float z,
    int num_levels,
    int res0,
    int res1,
    int res2,
    const int64_t* index0,
    const int64_t* index1,
    const int64_t* index2,
    int* out_res,
    int* out_ix,
    int* out_iy,
    int* out_iz) {
    for (int level = num_levels - 1; level >= 0; --level) {
        const int res = level_resolution(level, res0, res1, res2);
        const float fx = (x + 1.0f) * 0.5f * static_cast<float>(res);
        const float fy = (y + 1.0f) * 0.5f * static_cast<float>(res);
        const float fz = (z + 1.0f) * 0.5f * static_cast<float>(res);
        const int ix = static_cast<int>(floorf(fx));
        const int iy = static_cast<int>(floorf(fy));
        const int iz = static_cast<int>(floorf(fz));
        if (ix < 0 || iy < 0 || iz < 0 || ix >= res || iy >= res || iz >= res) {
            continue;
        }
        const int64_t* index = level_index_ptr(level, index0, index1, index2);
        const int64_t leaf = index[(static_cast<int64_t>(ix) * res + iy) * res + iz];
        if (leaf >= 0) {
            *out_res = res;
            *out_ix = ix;
            *out_iy = iy;
            *out_iz = iz;
            return leaf;
        }
    }
    return -1;
}

template <bool AccumulateGrad>
__device__ __forceinline__ float traverse_dynamic_voxels(
    const float* __restrict__ leaf_logits,
    const int64_t* __restrict__ index0,
    const int64_t* __restrict__ index1,
    const int64_t* __restrict__ index2,
    float* __restrict__ grad_leaf_logits,
    float grad_output,
    float angle,
    int64_t row,
    int64_t col,
    int detector_h,
    int detector_w,
    int num_levels,
    int res0,
    int res1,
    int res2,
    float attenuation_shift) {
    const float ca = cosf(-angle);
    const float sa = sinf(-angle);
    const float dir_x = ca;
    const float dir_y = sa;
    const float u_x = -sa;
    const float u_y = ca;
    const float u = -1.0f + (static_cast<float>(col) + 0.5f) * 2.0f / static_cast<float>(detector_w);
    const float z = -1.0f + (static_cast<float>(row) + 0.5f) * 2.0f / static_cast<float>(detector_h);
    const float base_x = u_x * u;
    const float base_y = u_y * u;

    float tmin = 0.0f;
    float tmax = 0.0f;
    if (z < -1.0f || z > 1.0f || !ray_box_interval_xy(base_x, base_y, dir_x, dir_y, &tmin, &tmax)) {
        return 0.0f;
    }

    const int max_res = max(res0, max(res1, res2));
    const int max_steps = max(32, 8 * max_res + 16);
    const float eps = 1.0e-6f;
    float t = tmin;
    float accum = 0.0f;

    for (int step_id = 0; step_id < max_steps && t < tmax - eps; ++step_id) {
        const float remaining = tmax - t;
        const float probe_delta = fminf(fmaxf(eps, remaining * 1.0e-5f), remaining * 0.5f);
        const float probe_t = t + probe_delta;
        const float x = base_x + dir_x * probe_t;
        const float y = base_y + dir_y * probe_t;

        int res = 0;
        int ix = 0;
        int iy = 0;
        int iz = 0;
        const int64_t leaf = lookup_leaf(
            x,
            y,
            z,
            num_levels,
            res0,
            res1,
            res2,
            index0,
            index1,
            index2,
            &res,
            &ix,
            &iy,
            &iz);
        if (leaf < 0) {
            break;
        }

        float next_t = tmax;
        if (dir_x > eps) {
            const float x_hi = -1.0f + 2.0f * static_cast<float>(ix + 1) / static_cast<float>(res);
            const float tx = (x_hi - base_x) / dir_x;
            if (tx > t + eps) {
                next_t = fminf(next_t, tx);
            }
        } else if (dir_x < -eps) {
            const float x_lo = -1.0f + 2.0f * static_cast<float>(ix) / static_cast<float>(res);
            const float tx = (x_lo - base_x) / dir_x;
            if (tx > t + eps) {
                next_t = fminf(next_t, tx);
            }
        }
        if (dir_y > eps) {
            const float y_hi = -1.0f + 2.0f * static_cast<float>(iy + 1) / static_cast<float>(res);
            const float ty = (y_hi - base_y) / dir_y;
            if (ty > t + eps) {
                next_t = fminf(next_t, ty);
            }
        } else if (dir_y < -eps) {
            const float y_lo = -1.0f + 2.0f * static_cast<float>(iy) / static_cast<float>(res);
            const float ty = (y_lo - base_y) / dir_y;
            if (ty > t + eps) {
                next_t = fminf(next_t, ty);
            }
        }
        next_t = fminf(next_t, tmax);
        if (!(next_t > t)) {
            break;
        }

        const float segment = next_t - t;
        const float shifted = leaf_logits[leaf] + attenuation_shift;
        if (AccumulateGrad) {
            atomicAdd(&grad_leaf_logits[leaf], grad_output * segment * sigmoid_stable(shifted));
        } else {
            accum += softplus_stable(shifted) * segment;
        }
        t = next_t;
    }

    return accum;
}

__global__ void dynamic_voxel_integrate_forward_kernel(
    const float* __restrict__ leaf_logits,
    const int64_t* __restrict__ index0,
    const int64_t* __restrict__ index1,
    const int64_t* __restrict__ index2,
    const float* __restrict__ angles,
    const int64_t* __restrict__ rows,
    const int64_t* __restrict__ cols,
    float* __restrict__ output,
    int num_rays,
    int detector_h,
    int detector_w,
    int num_levels,
    int res0,
    int res1,
    int res2,
    float attenuation_shift) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays) {
        return;
    }
    output[idx] = traverse_dynamic_voxels<false>(
        leaf_logits,
        index0,
        index1,
        index2,
        nullptr,
        0.0f,
        angles[idx],
        rows[idx],
        cols[idx],
        detector_h,
        detector_w,
        num_levels,
        res0,
        res1,
        res2,
        attenuation_shift);
}

__global__ void dynamic_voxel_integrate_backward_kernel(
    const float* __restrict__ leaf_logits,
    const int64_t* __restrict__ index0,
    const int64_t* __restrict__ index1,
    const int64_t* __restrict__ index2,
    const float* __restrict__ angles,
    const int64_t* __restrict__ rows,
    const int64_t* __restrict__ cols,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_leaf_logits,
    int num_rays,
    int detector_h,
    int detector_w,
    int num_levels,
    int res0,
    int res1,
    int res2,
    float attenuation_shift) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays) {
        return;
    }
    traverse_dynamic_voxels<true>(
        leaf_logits,
        index0,
        index1,
        index2,
        grad_leaf_logits,
        grad_output[idx],
        angles[idx],
        rows[idx],
        cols[idx],
        detector_h,
        detector_w,
        num_levels,
        res0,
        res1,
        res2,
        attenuation_shift);
}

}  // namespace

at::Tensor dynamic_voxel_integrate_forward_cuda(
    at::Tensor leaf_logits,
    at::Tensor index0,
    at::Tensor index1,
    at::Tensor index2,
    at::Tensor angles,
    at::Tensor rows,
    at::Tensor cols,
    int64_t detector_h,
    int64_t detector_w,
    int64_t num_levels,
    int64_t res0,
    int64_t res1,
    int64_t res2,
    double attenuation_shift) {
    check_cuda_float(leaf_logits, "leaf_logits");
    check_cuda_long(index0, "index0");
    check_cuda_float(angles, "angles");
    check_cuda_long(rows, "rows");
    check_cuda_long(cols, "cols");
    if (num_levels > 1) {
        check_cuda_long(index1, "index1");
    }
    if (num_levels > 2) {
        check_cuda_long(index2, "index2");
    }
    TORCH_CHECK(leaf_logits.dim() == 1, "leaf_logits must have shape (N,).");
    TORCH_CHECK(angles.dim() == 1 && rows.dim() == 1 && cols.dim() == 1, "angles, rows, and cols must be vectors.");
    TORCH_CHECK(angles.size(0) == rows.size(0) && angles.size(0) == cols.size(0), "ray input vectors must have the same length.");
    TORCH_CHECK(num_levels >= 1 && num_levels <= 3, "dynamic CUDA integrator supports 1 to 3 levels.");

    c10::cuda::CUDAGuard guard(leaf_logits.device());
    auto output = at::empty({angles.size(0)}, leaf_logits.options());
    const int num_rays = static_cast<int>(angles.size(0));
    constexpr int threads = 256;
    const int blocks = static_cast<int>((num_rays + threads - 1) / threads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();
    dynamic_voxel_integrate_forward_kernel<<<blocks, threads, 0, stream>>>(
        leaf_logits.contiguous().data_ptr<float>(),
        index0.contiguous().data_ptr<int64_t>(),
        num_levels > 1 ? index1.contiguous().data_ptr<int64_t>() : nullptr,
        num_levels > 2 ? index2.contiguous().data_ptr<int64_t>() : nullptr,
        angles.contiguous().data_ptr<float>(),
        rows.contiguous().data_ptr<int64_t>(),
        cols.contiguous().data_ptr<int64_t>(),
        output.data_ptr<float>(),
        num_rays,
        static_cast<int>(detector_h),
        static_cast<int>(detector_w),
        static_cast<int>(num_levels),
        static_cast<int>(res0),
        static_cast<int>(res1),
        static_cast<int>(res2),
        static_cast<float>(attenuation_shift));
    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "dynamic voxel integrate forward kernel failed: ", cudaGetErrorString(error));
    return output;
}

at::Tensor dynamic_voxel_integrate_backward_cuda(
    at::Tensor leaf_logits,
    at::Tensor index0,
    at::Tensor index1,
    at::Tensor index2,
    at::Tensor angles,
    at::Tensor rows,
    at::Tensor cols,
    at::Tensor grad_output,
    int64_t detector_h,
    int64_t detector_w,
    int64_t num_levels,
    int64_t res0,
    int64_t res1,
    int64_t res2,
    double attenuation_shift) {
    check_cuda_float(leaf_logits, "leaf_logits");
    check_cuda_long(index0, "index0");
    check_cuda_float(angles, "angles");
    check_cuda_long(rows, "rows");
    check_cuda_long(cols, "cols");
    check_cuda_float(grad_output, "grad_output");
    if (num_levels > 1) {
        check_cuda_long(index1, "index1");
    }
    if (num_levels > 2) {
        check_cuda_long(index2, "index2");
    }
    TORCH_CHECK(grad_output.dim() == 1 && grad_output.size(0) == angles.size(0), "grad_output must match rays.");
    TORCH_CHECK(num_levels >= 1 && num_levels <= 3, "dynamic CUDA integrator supports 1 to 3 levels.");

    c10::cuda::CUDAGuard guard(leaf_logits.device());
    auto grad_leaf_logits = at::zeros_like(leaf_logits);
    const int num_rays = static_cast<int>(angles.size(0));
    constexpr int threads = 256;
    const int blocks = static_cast<int>((num_rays + threads - 1) / threads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();
    dynamic_voxel_integrate_backward_kernel<<<blocks, threads, 0, stream>>>(
        leaf_logits.contiguous().data_ptr<float>(),
        index0.contiguous().data_ptr<int64_t>(),
        num_levels > 1 ? index1.contiguous().data_ptr<int64_t>() : nullptr,
        num_levels > 2 ? index2.contiguous().data_ptr<int64_t>() : nullptr,
        angles.contiguous().data_ptr<float>(),
        rows.contiguous().data_ptr<int64_t>(),
        cols.contiguous().data_ptr<int64_t>(),
        grad_output.contiguous().data_ptr<float>(),
        grad_leaf_logits.data_ptr<float>(),
        num_rays,
        static_cast<int>(detector_h),
        static_cast<int>(detector_w),
        static_cast<int>(num_levels),
        static_cast<int>(res0),
        static_cast<int>(res1),
        static_cast<int>(res2),
        static_cast<float>(attenuation_shift));
    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "dynamic voxel integrate backward kernel failed: ", cudaGetErrorString(error));
    return grad_leaf_logits;
}
