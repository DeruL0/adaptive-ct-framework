#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>

namespace {

void check_cuda_float(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32.");
}

__device__ __forceinline__ float sample_volume_xyz(
    const float* volume,
    int nx,
    int ny,
    int nz,
    float x,
    float y,
    float z) {
    const float fx = (x + 1.0f) * 0.5f * static_cast<float>(nx) - 0.5f;
    const float fy = (y + 1.0f) * 0.5f * static_cast<float>(ny) - 0.5f;
    const float fz = (z + 1.0f) * 0.5f * static_cast<float>(nz) - 0.5f;

    if (fx < 0.0f || fy < 0.0f || fz < 0.0f ||
        fx > static_cast<float>(nx - 1) ||
        fy > static_cast<float>(ny - 1) ||
        fz > static_cast<float>(nz - 1)) {
        return 0.0f;
    }

    const int x0 = static_cast<int>(floorf(fx));
    const int y0 = static_cast<int>(floorf(fy));
    const int z0 = static_cast<int>(floorf(fz));
    const int x1 = min(x0 + 1, nx - 1);
    const int y1 = min(y0 + 1, ny - 1);
    const int z1 = min(z0 + 1, nz - 1);
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const float tz = fz - static_cast<float>(z0);

    const int stride_y = nz;
    const int stride_x = ny * nz;
    const auto at = [&](int ix, int iy, int iz) {
        return volume[ix * stride_x + iy * stride_y + iz];
    };

    const float c000 = at(x0, y0, z0);
    const float c001 = at(x0, y0, z1);
    const float c010 = at(x0, y1, z0);
    const float c011 = at(x0, y1, z1);
    const float c100 = at(x1, y0, z0);
    const float c101 = at(x1, y0, z1);
    const float c110 = at(x1, y1, z0);
    const float c111 = at(x1, y1, z1);

    const float c00 = c000 * (1.0f - tx) + c100 * tx;
    const float c01 = c001 * (1.0f - tx) + c101 * tx;
    const float c10 = c010 * (1.0f - tx) + c110 * tx;
    const float c11 = c011 * (1.0f - tx) + c111 * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
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

__global__ void project_dense_parallel_kernel(
    const float* __restrict__ volume,
    const float* __restrict__ angles,
    float* __restrict__ output,
    int num_angles,
    int nx,
    int ny,
    int nz,
    int detector_h,
    int detector_w,
    int samples_per_ray) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int rays_per_angle = detector_h * detector_w;
    const int total = num_angles * rays_per_angle;
    if (idx >= total) {
        return;
    }

    const int angle_idx = idx / rays_per_angle;
    const int pixel = idx - angle_idx * rays_per_angle;
    const int row = pixel / detector_w;
    const int col = pixel - row * detector_w;

    // R2/TIGRE prepared projections use the opposite in-plane rotation sign
    // relative to this normalized ray basis.
    const float angle = -angles[angle_idx];
    const float ca = cosf(angle);
    const float sa = sinf(angle);
    const float dir_x = ca;
    const float dir_y = sa;
    const float u_x = -sa;
    const float u_y = ca;
    const float u = -1.0f + (static_cast<float>(col) + 0.5f) * 2.0f / static_cast<float>(detector_w);
    const float z = -1.0f + (static_cast<float>(row) + 0.5f) * 2.0f / static_cast<float>(detector_h);

    float tmin = 0.0f;
    float tmax = 0.0f;
    const float base_x = u_x * u;
    const float base_y = u_y * u;
    if (z < -1.0f || z > 1.0f || !ray_box_interval_xy(base_x, base_y, dir_x, dir_y, &tmin, &tmax)) {
        output[idx] = 0.0f;
        return;
    }

    const int samples = max(samples_per_ray, 1);
    const float step = (tmax - tmin) / static_cast<float>(samples);
    float accum = 0.0f;
    for (int s = 0; s < samples; ++s) {
        const float t = tmin + (static_cast<float>(s) + 0.5f) * step;
        const float x = base_x + dir_x * t;
        const float y = base_y + dir_y * t;
        accum += sample_volume_xyz(volume, nx, ny, nz, x, y, z);
    }
    output[idx] = accum * step;
}

}  // namespace

at::Tensor project_dense_parallel_forward_cuda(
    at::Tensor volume,
    at::Tensor angles,
    int64_t detector_h,
    int64_t detector_w,
    int64_t samples_per_ray) {
    check_cuda_float(volume, "volume");
    check_cuda_float(angles, "angles");
    TORCH_CHECK(volume.dim() == 3, "volume must have shape (X, Y, Z).");
    TORCH_CHECK(angles.dim() == 1, "angles must have shape (A,).");

    c10::cuda::CUDAGuard guard(volume.device());
    auto out = at::empty(
        {angles.size(0), detector_h, detector_w},
        at::TensorOptions().device(volume.device()).dtype(at::kFloat));

    const int64_t total = angles.size(0) * detector_h * detector_w;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getDefaultCUDAStream().stream();
    project_dense_parallel_kernel<<<blocks, threads, 0, stream>>>(
        volume.contiguous().data_ptr<float>(),
        angles.contiguous().data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(angles.size(0)),
        static_cast<int>(volume.size(0)),
        static_cast<int>(volume.size(1)),
        static_cast<int>(volume.size(2)),
        static_cast<int>(detector_h),
        static_cast<int>(detector_w),
        static_cast<int>(samples_per_ray));
    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "project_dense_parallel kernel failed: ", cudaGetErrorString(error));
    return out;
}
