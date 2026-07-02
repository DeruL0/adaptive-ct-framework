#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

void check_cuda_float(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32.");
}

void check_cuda_long(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must be int64.");
}

void check_cuda_int(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32.");
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
    const float bases[2] = {base_x, base_y};
    const float directions[2] = {dir_x, dir_y};
    for (int axis = 0; axis < 2; ++axis) {
        const float base = bases[axis];
        const float direction = directions[axis];
        if (fabsf(direction) < eps) {
            if (base < -1.0f || base > 1.0f) {
                return false;
            }
            continue;
        }
        float t0 = (-1.0f - base) / direction;
        float t1 = (1.0f - base) / direction;
        if (t0 > t1) {
            const float temporary = t0;
            t0 = t1;
            t1 = temporary;
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

__device__ __forceinline__ int64_t lookup_leaf(
    float x,
    float y,
    float z,
    int num_levels,
    int root_x,
    int root_y,
    int root_z,
    const int32_t* node_child_base,
    const int32_t* node_leaf_id,
    int* out_rx,
    int* out_ry,
    int* out_rz,
    int* out_ix,
    int* out_iy,
    int* out_iz) {
    int rx = root_x;
    int ry = root_y;
    int rz = root_z;
    int ix = static_cast<int>(floorf((x + 1.0f) * 0.5f * static_cast<float>(rx)));
    int iy = static_cast<int>(floorf((y + 1.0f) * 0.5f * static_cast<float>(ry)));
    int iz = static_cast<int>(floorf((z + 1.0f) * 0.5f * static_cast<float>(rz)));
    if (ix < 0 || iy < 0 || iz < 0 || ix >= rx || iy >= ry || iz >= rz) {
        return -1;
    }
    int64_t node = (static_cast<int64_t>(ix) * ry + iy) * rz + iz;
    for (int level = 0; level < num_levels; ++level) {
        const int32_t leaf = node_leaf_id[node];
        if (leaf >= 0) {
            *out_rx = rx;
            *out_ry = ry;
            *out_rz = rz;
            *out_ix = ix;
            *out_iy = iy;
            *out_iz = iz;
            return static_cast<int64_t>(leaf);
        }
        if (level + 1 >= num_levels) {
            return -1;
        }
        const int32_t child_base = node_child_base[node];
        if (child_base < 0) {
            return -1;
        }
        rx *= 2;
        ry *= 2;
        rz *= 2;
        ix = static_cast<int>(floorf((x + 1.0f) * 0.5f * static_cast<float>(rx)));
        iy = static_cast<int>(floorf((y + 1.0f) * 0.5f * static_cast<float>(ry)));
        iz = static_cast<int>(floorf((z + 1.0f) * 0.5f * static_cast<float>(rz)));
        if (ix < 0 || iy < 0 || iz < 0 || ix >= rx || iy >= ry || iz >= rz) {
            return -1;
        }
        const int child_index = ((ix & 1) << 2) | ((iy & 1) << 1) | (iz & 1);
        node = static_cast<int64_t>(child_base) + child_index;
    }
    return -1;
}

__device__ __forceinline__ void bernstein_basis(int degree, float x, float* values) {
    x = fminf(fmaxf(x, 0.0f), 1.0f);
    const float one_minus_x = 1.0f - x;
    values[0] = 1.0f;
    values[1] = 0.0f;
    values[2] = 0.0f;
    values[3] = 0.0f;
    if (degree == 1) {
        values[0] = one_minus_x;
        values[1] = x;
    } else if (degree == 2) {
        values[0] = one_minus_x * one_minus_x;
        values[1] = 2.0f * x * one_minus_x;
        values[2] = x * x;
    } else if (degree == 3) {
        values[0] = one_minus_x * one_minus_x * one_minus_x;
        values[1] = 3.0f * x * one_minus_x * one_minus_x;
        values[2] = 3.0f * x * x * one_minus_x;
        values[3] = x * x * x;
    }
}

__device__ __forceinline__ void gauss_legendre(int order, int index, float* node, float* weight) {
    if (order <= 1) {
        *node = 0.0f;
        *weight = 2.0f;
    } else if (order == 2) {
        const float nodes[2] = {-0.5773502691896257f, 0.5773502691896257f};
        *node = nodes[index];
        *weight = 1.0f;
    } else if (order == 3) {
        const float nodes[3] = {-0.7745966692414834f, 0.0f, 0.7745966692414834f};
        const float weights[3] = {0.5555555555555556f, 0.8888888888888888f, 0.5555555555555556f};
        *node = nodes[index];
        *weight = weights[index];
    } else if (order == 4) {
        const float nodes[4] = {
            -0.8611363115940526f,
            -0.3399810435848563f,
            0.3399810435848563f,
            0.8611363115940526f};
        const float weights[4] = {
            0.3478548451374538f,
            0.6521451548625461f,
            0.6521451548625461f,
            0.3478548451374538f};
        *node = nodes[index];
        *weight = weights[index];
    } else {
        const float nodes[5] = {
            -0.9061798459386640f,
            -0.5384693101056831f,
            0.0f,
            0.5384693101056831f,
            0.9061798459386640f};
        const float weights[5] = {
            0.2369268850561891f,
            0.4786286704993665f,
            0.5688888888888889f,
            0.4786286704993665f,
            0.2369268850561891f};
        *node = nodes[index];
        *weight = weights[index];
    }
}

__device__ __forceinline__ float segment_integral(
    const float* coefficient_logits,
    const int64_t* leaf_degrees,
    const int64_t* coefficient_offsets,
    int64_t leaf,
    int rx,
    int ry,
    int rz,
    int ix,
    int iy,
    int iz,
    float base_x,
    float base_y,
    float dir_x,
    float dir_y,
    float z,
    float t0,
    float t1,
    float attenuation_shift) {
    const int px = static_cast<int>(leaf_degrees[leaf * 3 + 0]);
    const int py = static_cast<int>(leaf_degrees[leaf * 3 + 1]);
    const int pz = static_cast<int>(leaf_degrees[leaf * 3 + 2]);
    const int quadrature_order = (px + py + pz + 2) / 2;
    const int64_t coefficient_start = coefficient_offsets[leaf];
    const float half = 0.5f * (t1 - t0);
    const float centre = 0.5f * (t1 + t0);
    float result = 0.0f;
    for (int quadrature_id = 0; quadrature_id < quadrature_order; ++quadrature_id) {
        float node = 0.0f;
        float weight = 0.0f;
        gauss_legendre(quadrature_order, quadrature_id, &node, &weight);
        const float t = centre + half * node;
        const float x = base_x + dir_x * t;
        const float y = base_y + dir_y * t;
        const float local_x = (x + 1.0f) * 0.5f * static_cast<float>(rx) - static_cast<float>(ix);
        const float local_y = (y + 1.0f) * 0.5f * static_cast<float>(ry) - static_cast<float>(iy);
        const float local_z = (z + 1.0f) * 0.5f * static_cast<float>(rz) - static_cast<float>(iz);
        float bx[4], by[4], bz[4];
        bernstein_basis(px, local_x, bx);
        bernstein_basis(py, local_y, by);
        bernstein_basis(pz, local_z, bz);
        float attenuation = 0.0f;
        int local_id = 0;
        for (int i = 0; i <= px; ++i) {
            for (int j = 0; j <= py; ++j) {
                for (int k = 0; k <= pz; ++k, ++local_id) {
                    attenuation += softplus_stable(
                        coefficient_logits[coefficient_start + local_id] + attenuation_shift) * bx[i] * by[j] * bz[k];
                }
            }
        }
        result += weight * attenuation;
    }
    return half * result;
}

__device__ __forceinline__ void segment_backward(
    const float* coefficient_logits,
    const int64_t* leaf_degrees,
    const int64_t* coefficient_offsets,
    float* grad_coefficient_logits,
    float grad_output,
    int64_t leaf,
    int rx,
    int ry,
    int rz,
    int ix,
    int iy,
    int iz,
    float base_x,
    float base_y,
    float dir_x,
    float dir_y,
    float z,
    float t0,
    float t1,
    float attenuation_shift) {
    const int px = static_cast<int>(leaf_degrees[leaf * 3 + 0]);
    const int py = static_cast<int>(leaf_degrees[leaf * 3 + 1]);
    const int pz = static_cast<int>(leaf_degrees[leaf * 3 + 2]);
    const int quadrature_order = (px + py + pz + 2) / 2;
    const int64_t coefficient_start = coefficient_offsets[leaf];
    const float half = 0.5f * (t1 - t0);
    const float centre = 0.5f * (t1 + t0);
    int local_id = 0;
    for (int i = 0; i <= px; ++i) {
        for (int j = 0; j <= py; ++j) {
            for (int k = 0; k <= pz; ++k, ++local_id) {
                float integrated_basis = 0.0f;
                for (int quadrature_id = 0; quadrature_id < quadrature_order; ++quadrature_id) {
                    float node = 0.0f;
                    float weight = 0.0f;
                    gauss_legendre(quadrature_order, quadrature_id, &node, &weight);
                    const float t = centre + half * node;
                    const float x = base_x + dir_x * t;
                    const float y = base_y + dir_y * t;
                    const float local_x = (x + 1.0f) * 0.5f * static_cast<float>(rx) - static_cast<float>(ix);
                    const float local_y = (y + 1.0f) * 0.5f * static_cast<float>(ry) - static_cast<float>(iy);
                    const float local_z = (z + 1.0f) * 0.5f * static_cast<float>(rz) - static_cast<float>(iz);
                    float bx[4], by[4], bz[4];
                    bernstein_basis(px, local_x, bx);
                    bernstein_basis(py, local_y, by);
                    bernstein_basis(pz, local_z, bz);
                    integrated_basis += weight * bx[i] * by[j] * bz[k];
                }
                const float shifted = coefficient_logits[coefficient_start + local_id] + attenuation_shift;
                atomicAdd(
                    &grad_coefficient_logits[coefficient_start + local_id],
                    grad_output * half * integrated_basis * sigmoid_stable(shifted));
            }
        }
    }
}

__device__ __forceinline__ float traverse_bernstein_octree(
    const float* coefficient_logits,
    const int64_t* leaf_degrees,
    const int64_t* coefficient_offsets,
    const int32_t* node_child_base,
    const int32_t* node_leaf_id,
    float* grad_coefficient_logits,
    float grad_output,
    int64_t* segment_leaf_ids,
    float* segment_contributions,
    int max_segments,
    float angle,
    int64_t row,
    int64_t col,
    int detector_h,
    int detector_w,
    int num_levels,
    int root_x,
    int root_y,
    int root_z,
    float attenuation_shift) {
    const float ca = cosf(-angle);
    const float sa = sinf(-angle);
    const float dir_x = ca;
    const float dir_y = sa;
    const float u = -1.0f + (static_cast<float>(col) + 0.5f) * 2.0f / static_cast<float>(detector_w);
    const float z = -1.0f + (static_cast<float>(row) + 0.5f) * 2.0f / static_cast<float>(detector_h);
    const float base_x = -sa * u;
    const float base_y = ca * u;
    float tmin = 0.0f;
    float tmax = 0.0f;
    if (z < -1.0f || z > 1.0f || !ray_box_interval_xy(base_x, base_y, dir_x, dir_y, &tmin, &tmax)) {
        return 0.0f;
    }

    const float eps = 1.0e-6f;
    float t = tmin;
    float total = 0.0f;
    int segment_id = 0;
    for (; segment_id < max_segments && t < tmax - eps; ++segment_id) {
        const float remaining = tmax - t;
        // A one-ULP probe can remain numerically on the previous cell after a
        // shallow-angle boundary crossing. Use a small geometric nudge that is
        // still far below the narrowest supported leaf width.
        const float probe_t = t + fminf(fmaxf(1.0e-5f, remaining * 1.0e-5f), remaining * 0.5f);
        const float probe_x = base_x + dir_x * probe_t;
        const float probe_y = base_y + dir_y * probe_t;
        int rx = 0;
        int ry = 0;
        int rz = 0;
        int ix = 0;
        int iy = 0;
        int iz = 0;
        const int64_t leaf = lookup_leaf(
            probe_x,
            probe_y,
            z,
            num_levels,
            root_x,
            root_y,
            root_z,
            node_child_base,
            node_leaf_id,
            &rx,
            &ry,
            &rz,
            &ix,
            &iy,
            &iz);
        if (leaf < 0) {
            break;
        }

        float next_t = tmax;
        if (dir_x > eps) {
            const float boundary = -1.0f + 2.0f * static_cast<float>(ix + 1) / static_cast<float>(rx);
            const float crossing = (boundary - base_x) / dir_x;
            if (crossing > t + eps) {
                next_t = fminf(next_t, crossing);
            }
        } else if (dir_x < -eps) {
            const float boundary = -1.0f + 2.0f * static_cast<float>(ix) / static_cast<float>(rx);
            const float crossing = (boundary - base_x) / dir_x;
            if (crossing > t + eps) {
                next_t = fminf(next_t, crossing);
            }
        }
        if (dir_y > eps) {
            const float boundary = -1.0f + 2.0f * static_cast<float>(iy + 1) / static_cast<float>(ry);
            const float crossing = (boundary - base_y) / dir_y;
            if (crossing > t + eps) {
                next_t = fminf(next_t, crossing);
            }
        } else if (dir_y < -eps) {
            const float boundary = -1.0f + 2.0f * static_cast<float>(iy) / static_cast<float>(ry);
            const float crossing = (boundary - base_y) / dir_y;
            if (crossing > t + eps) {
                next_t = fminf(next_t, crossing);
            }
        }
        next_t = fminf(next_t, tmax);
        if (!(next_t > t)) {
            break;
        }

        if (grad_coefficient_logits != nullptr) {
            segment_backward(
                coefficient_logits,
                leaf_degrees,
                coefficient_offsets,
                grad_coefficient_logits,
                grad_output,
                leaf,
                rx,
                ry,
                rz,
                ix,
                iy,
                iz,
                base_x,
                base_y,
                dir_x,
                dir_y,
                z,
                t,
                next_t,
                attenuation_shift);
        } else {
            const float contribution = segment_integral(
                coefficient_logits,
                leaf_degrees,
                coefficient_offsets,
                leaf,
                rx,
                ry,
                rz,
                ix,
                iy,
                iz,
                base_x,
                base_y,
                dir_x,
                dir_y,
                z,
                t,
                next_t,
                attenuation_shift);
            total += contribution;
            if (segment_leaf_ids != nullptr && segment_contributions != nullptr) {
                segment_leaf_ids[segment_id] = leaf;
                segment_contributions[segment_id] = contribution;
            }
        }
        t = next_t;
    }
    return total;
}

__global__ void bernstein_forward_kernel(
    const float* coefficient_logits,
    const int64_t* leaf_degrees,
    const int64_t* coefficient_offsets,
    const int32_t* node_child_base,
    const int32_t* node_leaf_id,
    const float* angles,
    const int64_t* rows,
    const int64_t* cols,
    float* output,
    int num_rays,
    int detector_h,
    int detector_w,
    int num_levels,
    int root_x,
    int root_y,
    int root_z,
    int max_segments,
    float attenuation_shift) {
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= num_rays) {
        return;
    }
    output[ray_id] = traverse_bernstein_octree(
        coefficient_logits,
        leaf_degrees,
        coefficient_offsets,
        node_child_base,
        node_leaf_id,
        nullptr,
        0.0f,
        nullptr,
        nullptr,
        max_segments,
        angles[ray_id],
        rows[ray_id],
        cols[ray_id],
        detector_h,
        detector_w,
        num_levels,
        root_x,
        root_y,
        root_z,
        attenuation_shift);
}

__global__ void bernstein_backward_kernel(
    const float* coefficient_logits,
    const int64_t* leaf_degrees,
    const int64_t* coefficient_offsets,
    const int32_t* node_child_base,
    const int32_t* node_leaf_id,
    const float* angles,
    const int64_t* rows,
    const int64_t* cols,
    const float* grad_output,
    float* grad_coefficient_logits,
    int num_rays,
    int detector_h,
    int detector_w,
    int num_levels,
    int root_x,
    int root_y,
    int root_z,
    int max_segments,
    float attenuation_shift) {
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= num_rays) {
        return;
    }
    traverse_bernstein_octree(
        coefficient_logits,
        leaf_degrees,
        coefficient_offsets,
        node_child_base,
        node_leaf_id,
        grad_coefficient_logits,
        grad_output[ray_id],
        nullptr,
        nullptr,
        max_segments,
        angles[ray_id],
        rows[ray_id],
        cols[ray_id],
        detector_h,
        detector_w,
        num_levels,
        root_x,
        root_y,
        root_z,
        attenuation_shift);
}

__global__ void bernstein_segments_kernel(
    const float* coefficient_logits,
    const int64_t* leaf_degrees,
    const int64_t* coefficient_offsets,
    const int32_t* node_child_base,
    const int32_t* node_leaf_id,
    const float* angles,
    const int64_t* rows,
    const int64_t* cols,
    int64_t* leaf_ids,
    float* contributions,
    int num_rays,
    int detector_h,
    int detector_w,
    int num_levels,
    int root_x,
    int root_y,
    int root_z,
    int max_segments,
    float attenuation_shift) {
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= num_rays) {
        return;
    }
    traverse_bernstein_octree(
        coefficient_logits,
        leaf_degrees,
        coefficient_offsets,
        node_child_base,
        node_leaf_id,
        nullptr,
        0.0f,
        leaf_ids + static_cast<int64_t>(ray_id) * max_segments,
        contributions + static_cast<int64_t>(ray_id) * max_segments,
        max_segments,
        angles[ray_id],
        rows[ray_id],
        cols[ray_id],
        detector_h,
        detector_w,
        num_levels,
        root_x,
        root_y,
        root_z,
        attenuation_shift);
}

void validate_inputs(
    const at::Tensor& coefficient_logits,
    const at::Tensor& leaf_degrees,
    const at::Tensor& coefficient_offsets,
    const at::Tensor& node_child_base,
    const at::Tensor& node_leaf_id,
    const at::Tensor& angles,
    const at::Tensor& rows,
    const at::Tensor& cols,
    int64_t num_levels) {
    check_cuda_float(coefficient_logits, "coefficient_logits");
    check_cuda_long(leaf_degrees, "leaf_degrees");
    check_cuda_long(coefficient_offsets, "coefficient_offsets");
    check_cuda_int(node_child_base, "node_child_base");
    check_cuda_int(node_leaf_id, "node_leaf_id");
    check_cuda_float(angles, "angles");
    check_cuda_long(rows, "rows");
    check_cuda_long(cols, "cols");
    TORCH_CHECK(coefficient_logits.dim() == 1, "coefficient_logits must have shape (K,).");
    TORCH_CHECK(leaf_degrees.dim() == 2 && leaf_degrees.size(1) == 3, "leaf_degrees must have shape (N, 3).");
    TORCH_CHECK(coefficient_offsets.dim() == 1 && coefficient_offsets.size(0) == leaf_degrees.size(0) + 1,
        "coefficient_offsets must have shape (N + 1,).");
    TORCH_CHECK(angles.dim() == 1 && rows.dim() == 1 && cols.dim() == 1, "ray inputs must be vectors.");
    TORCH_CHECK(angles.size(0) == rows.size(0) && angles.size(0) == cols.size(0), "ray vectors must have equal length.");
    TORCH_CHECK(node_child_base.dim() == 1 && node_leaf_id.dim() == 1,
        "packed node arrays must be vectors.");
    TORCH_CHECK(node_child_base.size(0) == node_leaf_id.size(0),
        "packed node arrays must have equal length.");
    TORCH_CHECK(num_levels >= 1 && num_levels <= 16, "Bernstein CUDA integrator supports 1 to 16 levels.");
    // Degree bounds are part of the Python model contract. Avoid reducing this
    // GPU tensor on every forward/backward launch, which would synchronize the
    // hot training path.
}

int max_segments_for_resolutions(int64_t root_x, int64_t root_y, int64_t num_levels) {
    const int64_t maximum_x = root_x << (num_levels - 1);
    const int64_t maximum_y = root_y << (num_levels - 1);
    TORCH_CHECK(maximum_x <= 1000000000 && maximum_y <= 1000000000,
        "Finest Bernstein resolution is too large.");
    return static_cast<int>(maximum_x + maximum_y + 8);
}

}  // namespace

at::Tensor bernstein_octree_integrate_forward_cuda(
    at::Tensor coefficient_logits,
    at::Tensor leaf_degrees,
    at::Tensor coefficient_offsets,
    at::Tensor node_child_base,
    at::Tensor node_leaf_id,
    at::Tensor angles,
    at::Tensor rows,
    at::Tensor cols,
    int64_t detector_h,
    int64_t detector_w,
    int64_t num_levels,
    int64_t root_x,
    int64_t root_y,
    int64_t root_z,
    double attenuation_shift) {
    validate_inputs(coefficient_logits, leaf_degrees, coefficient_offsets, node_child_base, node_leaf_id, angles, rows, cols, num_levels);
    c10::cuda::CUDAGuard guard(coefficient_logits.device());
    auto output = at::empty({angles.size(0)}, coefficient_logits.options());
    const int num_rays = static_cast<int>(angles.size(0));
    constexpr int threads = 128;
    const int blocks = (num_rays + threads - 1) / threads;
    const int max_segments = max_segments_for_resolutions(root_x, root_y, num_levels);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bernstein_forward_kernel<<<blocks, threads, 0, stream>>>(
        coefficient_logits.contiguous().data_ptr<float>(),
        leaf_degrees.contiguous().data_ptr<int64_t>(),
        coefficient_offsets.contiguous().data_ptr<int64_t>(),
        node_child_base.contiguous().data_ptr<int32_t>(),
        node_leaf_id.contiguous().data_ptr<int32_t>(),
        angles.contiguous().data_ptr<float>(),
        rows.contiguous().data_ptr<int64_t>(),
        cols.contiguous().data_ptr<int64_t>(),
        output.data_ptr<float>(),
        num_rays,
        static_cast<int>(detector_h),
        static_cast<int>(detector_w),
        static_cast<int>(num_levels),
        static_cast<int>(root_x),
        static_cast<int>(root_y),
        static_cast<int>(root_z),
        max_segments,
        static_cast<float>(attenuation_shift));
    const auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "Bernstein integrate forward kernel failed: ", cudaGetErrorString(error));
    return output;
}

at::Tensor bernstein_octree_integrate_backward_cuda(
    at::Tensor coefficient_logits,
    at::Tensor leaf_degrees,
    at::Tensor coefficient_offsets,
    at::Tensor node_child_base,
    at::Tensor node_leaf_id,
    at::Tensor angles,
    at::Tensor rows,
    at::Tensor cols,
    at::Tensor grad_output,
    int64_t detector_h,
    int64_t detector_w,
    int64_t num_levels,
    int64_t root_x,
    int64_t root_y,
    int64_t root_z,
    double attenuation_shift) {
    validate_inputs(coefficient_logits, leaf_degrees, coefficient_offsets, node_child_base, node_leaf_id, angles, rows, cols, num_levels);
    check_cuda_float(grad_output, "grad_output");
    TORCH_CHECK(grad_output.dim() == 1 && grad_output.size(0) == angles.size(0), "grad_output must match rays.");
    c10::cuda::CUDAGuard guard(coefficient_logits.device());
    auto grad_coefficients = at::zeros_like(coefficient_logits);
    const int num_rays = static_cast<int>(angles.size(0));
    constexpr int threads = 128;
    const int blocks = (num_rays + threads - 1) / threads;
    const int max_segments = max_segments_for_resolutions(root_x, root_y, num_levels);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bernstein_backward_kernel<<<blocks, threads, 0, stream>>>(
        coefficient_logits.contiguous().data_ptr<float>(),
        leaf_degrees.contiguous().data_ptr<int64_t>(),
        coefficient_offsets.contiguous().data_ptr<int64_t>(),
        node_child_base.contiguous().data_ptr<int32_t>(),
        node_leaf_id.contiguous().data_ptr<int32_t>(),
        angles.contiguous().data_ptr<float>(),
        rows.contiguous().data_ptr<int64_t>(),
        cols.contiguous().data_ptr<int64_t>(),
        grad_output.contiguous().data_ptr<float>(),
        grad_coefficients.data_ptr<float>(),
        num_rays,
        static_cast<int>(detector_h),
        static_cast<int>(detector_w),
        static_cast<int>(num_levels),
        static_cast<int>(root_x),
        static_cast<int>(root_y),
        static_cast<int>(root_z),
        max_segments,
        static_cast<float>(attenuation_shift));
    const auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "Bernstein integrate backward kernel failed: ", cudaGetErrorString(error));
    return grad_coefficients;
}

std::vector<at::Tensor> bernstein_octree_segments_forward_cuda(
    at::Tensor coefficient_logits,
    at::Tensor leaf_degrees,
    at::Tensor coefficient_offsets,
    at::Tensor node_child_base,
    at::Tensor node_leaf_id,
    at::Tensor angles,
    at::Tensor rows,
    at::Tensor cols,
    int64_t detector_h,
    int64_t detector_w,
    int64_t num_levels,
    int64_t root_x,
    int64_t root_y,
    int64_t root_z,
    double attenuation_shift) {
    validate_inputs(coefficient_logits, leaf_degrees, coefficient_offsets, node_child_base, node_leaf_id, angles, rows, cols, num_levels);
    c10::cuda::CUDAGuard guard(coefficient_logits.device());
    const int num_rays = static_cast<int>(angles.size(0));
    const int max_segments = max_segments_for_resolutions(root_x, root_y, num_levels);
    auto leaf_ids = at::full({num_rays, max_segments}, -1, leaf_degrees.options());
    auto contributions = at::zeros({num_rays, max_segments}, coefficient_logits.options());
    constexpr int threads = 128;
    const int blocks = (num_rays + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bernstein_segments_kernel<<<blocks, threads, 0, stream>>>(
        coefficient_logits.contiguous().data_ptr<float>(),
        leaf_degrees.contiguous().data_ptr<int64_t>(),
        coefficient_offsets.contiguous().data_ptr<int64_t>(),
        node_child_base.contiguous().data_ptr<int32_t>(),
        node_leaf_id.contiguous().data_ptr<int32_t>(),
        angles.contiguous().data_ptr<float>(),
        rows.contiguous().data_ptr<int64_t>(),
        cols.contiguous().data_ptr<int64_t>(),
        leaf_ids.data_ptr<int64_t>(),
        contributions.data_ptr<float>(),
        num_rays,
        static_cast<int>(detector_h),
        static_cast<int>(detector_w),
        static_cast<int>(num_levels),
        static_cast<int>(root_x),
        static_cast<int>(root_y),
        static_cast<int>(root_z),
        max_segments,
        static_cast<float>(attenuation_shift));
    const auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "Bernstein segments kernel failed: ", cudaGetErrorString(error));
    return {leaf_ids, contributions};
}
