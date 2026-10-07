// Exact line integrals of a typed-leaf octree along parallel rays (layout: adaptive_ct/representation/octree.py).
//   leaves [0, n_tri) trilinear: values[8 i .. 8 i + 7] (32-byte aligned block, read as two float4)
//   leaves [n_tri, n_tri + n_step) step: values[8 n_tri + 2 j], [+1], j = i - n_tri; geometry row j = (n, d, w, pad x 3)
//   other leaves constant: values[8 n_tri + 2 n_step + k], k = i - n_tri - n_step
//   tree: root[cell] / children[node * 8 + slot] int32; >= 0 leaf index, < 0 internal node -1 - value
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cmath>
#include <cstdint>

namespace {

namespace cg = cooperative_groups;

struct Tree {
    const float* values;
    const float* geometry;
    const int32_t* root;
    const int32_t* children;
    int64_t n_tri, n_step;
    int levels, rx, ry, rz;
};

struct Rays {
    const float* angles;
    const int64_t* rows;
    const int64_t* cols;
    int count, detector_h, detector_w, max_segments;
};

// One ray segment inside one leaf: the leaf's cell (resolution r, index c) and the ray x = bx + dx t, y = by + dy t, z.
struct Segment {
    int64_t leaf;
    int rx, ry, rz, ix, iy, iz;
    float bx, by, dx, dy, z, t0, t1;
    __device__ float ux(float t) const { return (bx + dx * t + 1.0f) * 0.5f * rx - ix; }
    __device__ float uy(float t) const { return (by + dy * t + 1.0f) * 0.5f * ry - iy; }
    __device__ float uz() const { return (z + 1.0f) * 0.5f * rz - iz; }
};

__device__ __forceinline__ float softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ float sigmoid_stable(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

__device__ __forceinline__ int leaf_kind(const Tree& T, int64_t leaf) {  // type id: 0 constant, 1 trilinear, 2 step
    return leaf < T.n_tri ? 1 : (leaf < T.n_tri + T.n_step ? 2 : 0);
}

__device__ __forceinline__ int64_t value_offset(const Tree& T, int64_t leaf) {
    if (leaf < T.n_tri) return 8 * leaf;
    if (leaf < T.n_tri + T.n_step) return 8 * T.n_tri + 2 * (leaf - T.n_tri);
    return 8 * T.n_tri + 2 * T.n_step + (leaf - T.n_tri - T.n_step);
}

__device__ int64_t lookup_leaf(const Tree& T, float x, float y, float z, int* r, int* c) {
    int rx = T.rx, ry = T.ry, rz = T.rz;
    int ix = static_cast<int>(floorf((x + 1.0f) * 0.5f * rx));
    int iy = static_cast<int>(floorf((y + 1.0f) * 0.5f * ry));
    int iz = static_cast<int>(floorf((z + 1.0f) * 0.5f * rz));
    if (ix < 0 || iy < 0 || iz < 0 || ix >= rx || iy >= ry || iz >= rz) return -1;
    int32_t v = T.root[(static_cast<int64_t>(ix) * ry + iy) * rz + iz];
    for (int level = 1; v < 0 && level < T.levels; ++level) {
        rx *= 2; ry *= 2; rz *= 2;
        ix = static_cast<int>(floorf((x + 1.0f) * 0.5f * rx));
        iy = static_cast<int>(floorf((y + 1.0f) * 0.5f * ry));
        iz = static_cast<int>(floorf((z + 1.0f) * 0.5f * rz));
        if (ix < 0 || iy < 0 || iz < 0 || ix >= rx || iy >= ry || iz >= rz) return -1;
        v = T.children[static_cast<int64_t>(-1 - v) * 8 + (((ix & 1) << 2) | ((iy & 1) << 1) | (iz & 1))];
    }
    if (v < 0) return -1;
    r[0] = rx; r[1] = ry; r[2] = rz; c[0] = ix; c[1] = iy; c[2] = iz;
    return v;
}

// Integral of H = sigmoid((n . (u - 1/2) - d) / w) over the segment (w = 0: hard step).  s is linear along the
// segment, so it is exact: L (softplus(s1 / w) - softplus(s0 / w)) / ((s1 - s0) / w).
__device__ float step_integral(const float* row, const Segment& s) {
    const float4 g = reinterpret_cast<const float4*>(row)[0];
    const float w = row[4];
    const float uz = s.uz() - 0.5f;
    const float s0 = g.x * (s.ux(s.t0) - 0.5f) + g.y * (s.uy(s.t0) - 0.5f) + g.z * uz - g.w;
    const float s1 = g.x * (s.ux(s.t1) - 0.5f) + g.y * (s.uy(s.t1) - 0.5f) + g.z * uz - g.w;
    const float length = s.t1 - s.t0;
    if (w <= 0.0f) {
        if (s0 > 0.0f && s1 > 0.0f) return length;
        if (s0 <= 0.0f && s1 <= 0.0f) return 0.0f;
        return length * fmaxf(s0, s1) / fabsf(s1 - s0);
    }
    const float a0 = s0 / w, a1 = s1 / w;
    if (fabsf(a1 - a0) < 1.0e-3f) return length * sigmoid_stable(0.5f * (a0 + a1));
    return length * (softplus_stable(a1) - softplus_stable(a0)) / (a1 - a0);
}

// Basis integrals of the 8 trilinear corner weights (corner 4 i + 2 j + k): 2-point Gauss is exact (z is constant
// along the ray, so the weights are quadratic in t).
__device__ void trilinear_integrals(const Segment& s, float* w) {
    const float length = s.t1 - s.t0;
    const float uz = fminf(fmaxf(s.uz(), 0.0f), 1.0f);
    for (int k = 0; k < 8; ++k) w[k] = 0.0f;
    for (int q = 0; q < 2; ++q) {
        const float t = 0.5f * (s.t0 + s.t1) + 0.5f * length * (q == 0 ? -0.5773502691896257f : 0.5773502691896257f);
        const float ux = fminf(fmaxf(s.ux(t), 0.0f), 1.0f), uy = fminf(fmaxf(s.uy(t), 0.0f), 1.0f);
        const float bx[2] = {1.0f - ux, ux}, by[2] = {1.0f - uy, uy}, bz[2] = {1.0f - uz, uz};
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) w[(i * 2 + j) * 2 + k] += 0.5f * length * bx[i] * by[j] * bz[k];
    }
}

// Integrals of the leaf's basis functions along the segment; returns their count, *offset = first value.
__device__ int segment_weights(const Tree& T, const Segment& s, float* w, int64_t* offset) {
    *offset = value_offset(T, s.leaf);
    const int kind = leaf_kind(T, s.leaf);
    if (kind == 1) {
        trilinear_integrals(s, w);
        return 8;
    }
    if (kind == 2) {
        const float I = step_integral(T.geometry + 8 * (s.leaf - T.n_tri), s);
        w[0] = s.t1 - s.t0 - I;
        w[1] = I;
        return 2;
    }
    w[0] = s.t1 - s.t0;
    return 1;
}

__device__ float segment_value(const Tree& T, const Segment& s) {
    float w[8];
    int64_t off;
    const int n = segment_weights(T, s, w, &off);
    if (n == 8) {
        const float4* p = reinterpret_cast<const float4*>(T.values + off);
        const float4 a = p[0], b = p[1];
        return w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
    }
    if (n == 2) {
        const float2 c = reinterpret_cast<const float2*>(T.values + off)[0];
        return w[0] * c.x + w[1] * c.y;
    }
    return w[0] * T.values[off];
}

// Neighbouring rays of a warp usually cross the same leaf: sum within the warp, one atomic add per group.
__device__ __forceinline__ void warp_add(const cg::coalesced_group& peers, float* slot, float value) {
    if (peers.size() > 1) value = cg::reduce(peers, value, cg::plus<float>());
    if (peers.thread_rank() == 0) atomicAdd(slot, value);
}

template <typename Visit>
__device__ void traverse(const Tree& T, const Rays& R, int ray, Visit&& visit) {
    const float a = -R.angles[ray];
    const float dx = cosf(a), dy = sinf(a);
    const float u = -1.0f + (static_cast<float>(R.cols[ray]) + 0.5f) * 2.0f / R.detector_w;
    const float z = -1.0f + (static_cast<float>(R.rows[ray]) + 0.5f) * 2.0f / R.detector_h;
    const float bx = -dy * u, by = dx * u;
    float tmin = -1.0e20f, tmax = 1.0e20f;
    const float b2[2] = {bx, by}, d2[2] = {dx, dy};
    for (int axis = 0; axis < 2; ++axis) {
        if (fabsf(d2[axis]) < 1.0e-8f) {
            if (b2[axis] < -1.0f || b2[axis] > 1.0f) return;
            continue;
        }
        const float t0 = (-1.0f - b2[axis]) / d2[axis], t1 = (1.0f - b2[axis]) / d2[axis];
        tmin = fmaxf(tmin, fminf(t0, t1));
        tmax = fminf(tmax, fmaxf(t0, t1));
    }
    if (z < -1.0f || z > 1.0f || tmax <= tmin) return;
    const float eps = 1.0e-6f;
    float t = tmin;
    for (int n = 0; n < R.max_segments && t < tmax - eps; ++n) {
        // A one-ULP probe can stay in the previous cell after a shallow crossing: nudge it, far below a leaf width.
        const float remaining = tmax - t;
        const float probe = t + fminf(fmaxf(1.0e-5f, remaining * 1.0e-5f), remaining * 0.5f);
        int r[3], c[3];
        const int64_t leaf = lookup_leaf(T, bx + dx * probe, by + dy * probe, z, r, c);
        if (leaf < 0) return;
        float next = tmax;
        if (dx > eps) next = fminf(next, (-1.0f + 2.0f * (c[0] + 1) / r[0] - bx) / dx);
        else if (dx < -eps) next = fminf(next, (-1.0f + 2.0f * c[0] / r[0] - bx) / dx);
        if (dy > eps) next = fminf(next, (-1.0f + 2.0f * (c[1] + 1) / r[1] - by) / dy);
        else if (dy < -eps) next = fminf(next, (-1.0f + 2.0f * c[1] / r[1] - by) / dy);
        if (!(next > t)) return;
        visit(Segment{leaf, r[0], r[1], r[2], c[0], c[1], c[2], bx, by, dx, dy, z, t, next});
        t = next;
    }
}

__global__ void forward_kernel(Tree T, Rays R, float* out) {
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= R.count) return;
    float total = 0.0f;
    traverse(T, R, ray, [&](const Segment& s) { total += segment_value(T, s); });
    out[ray] = total;
}

__global__ void backward_kernel(Tree T, Rays R, const float* grad_out, float* grad_values) {
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= R.count) return;
    const float g = grad_out[ray];
    traverse(T, R, ray, [&](const Segment& s) {
        float w[8];
        int64_t off;
        const int n = segment_weights(T, s, w, &off);
        const cg::coalesced_group peers = cg::labeled_partition(cg::coalesced_threads(), static_cast<unsigned int>(s.leaf));
        for (int k = 0; k < n; ++k) warp_add(peers, grad_values + off + k, g * w[k]);
    });
}

// Decision statistics (refinement/decision.py): per (ray, leaf) segment the basis integrals of every option
//   constant: L · trilinear: 8 corner basis integrals · step: (L - I, I) with the candidate geometry ·
//   split: length inside each of the 8 children (current function + one constant per child)
// and b = (this leaf's current contribution) - (ray residual), the data seen by a refit of this leaf with all
// other leaves fixed.  Accumulated per leaf and ray half (fold) into DECISION_STRIDE floats:
//   0 L*L, 1 L*b | 2..37 trilinear G (upper, row major), 38..45 trilinear*b | 46..48 step G (00, 01, 11),
//   49..50 step*b | 51..86 split G (upper), 87..94 split*(-residual) | 95 b*b | 96 residual^2
constexpr int DECISION_STRIDE = 97;

__device__ void segment_decision(const Tree& T, const Segment& s, const float* candidate, float* accum, int64_t leaf_count,
                                 int fold, float residual) {
    const float length = s.t1 - s.t0;
    const float b = segment_value(T, s) - residual;
    float tri[8];
    trilinear_integrals(s, tri);
    const float I = step_integral(candidate + 8 * s.leaf, s);
    const float st[2] = {length - I, I};
    // Child lengths: cut the segment at the leaf's mid-planes in x and y (z is constant along the ray).
    float cuts[4];
    int n = 0;
    cuts[n++] = s.t0;
    if (fabsf(s.dx) > 1.0e-8f) {
        const float tc = (-1.0f + 2.0f * (s.ix + 0.5f) / s.rx - s.bx) / s.dx;
        if (tc > s.t0 && tc < s.t1) cuts[n++] = tc;
    }
    if (fabsf(s.dy) > 1.0e-8f) {
        const float tc = (-1.0f + 2.0f * (s.iy + 0.5f) / s.ry - s.by) / s.dy;
        if (tc > s.t0 && tc < s.t1) cuts[n++] = tc;
    }
    cuts[n++] = s.t1;
    if (n == 4 && cuts[1] > cuts[2]) {
        const float tmp = cuts[1];
        cuts[1] = cuts[2];
        cuts[2] = tmp;
    }
    float child[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int cz = s.uz() > 0.5f ? 1 : 0;
    for (int c = 0; c + 1 < n; ++c) {
        const float tm = 0.5f * (cuts[c] + cuts[c + 1]);
        child[(s.ux(tm) > 0.5f ? 4 : 0) + (s.uy(tm) > 0.5f ? 2 : 0) + cz] += cuts[c + 1] - cuts[c];
    }
    const cg::coalesced_group peers = cg::labeled_partition(cg::coalesced_threads(), static_cast<unsigned int>(s.leaf * 2 + fold));
    float* slot = accum + (static_cast<int64_t>(fold) * leaf_count + s.leaf) * DECISION_STRIDE;
    int k = 0;
    warp_add(peers, slot + k++, length * length);
    warp_add(peers, slot + k++, length * b);
    for (int i = 0; i < 8; ++i)
        for (int j = i; j < 8; ++j) warp_add(peers, slot + k++, tri[i] * tri[j]);
    for (int i = 0; i < 8; ++i) warp_add(peers, slot + k++, tri[i] * b);
    warp_add(peers, slot + k++, st[0] * st[0]);
    warp_add(peers, slot + k++, st[0] * st[1]);
    warp_add(peers, slot + k++, st[1] * st[1]);
    warp_add(peers, slot + k++, st[0] * b);
    warp_add(peers, slot + k++, st[1] * b);
    for (int i = 0; i < 8; ++i)
        for (int j = i; j < 8; ++j) warp_add(peers, slot + k++, child[i] * child[j]);
    for (int i = 0; i < 8; ++i) warp_add(peers, slot + k++, -child[i] * residual);
    warp_add(peers, slot + k++, b * b);
    warp_add(peers, slot + k++, residual * residual);
}

__global__ void decision_kernel(Tree T, Rays R, const float* candidate, const float* residual, const int32_t* fold, float* accum,
                                int64_t leaf_count) {
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= R.count) return;
    traverse(T, R, ray, [&](const Segment& s) { segment_decision(T, s, candidate, accum, leaf_count, fold[ray], residual[ray]); });
}

void check(const at::Tensor& t, at::ScalarType type, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == type && t.is_contiguous(), name, " must be a contiguous CUDA tensor of type ", type);
}

Tree make_tree(const at::Tensor& values, const at::Tensor& geometry, const at::Tensor& root, const at::Tensor& children,
               int64_t n_tri, int64_t n_step, int64_t levels, int64_t rx, int64_t ry, int64_t rz) {
    check(values, at::kFloat, "values");
    check(geometry, at::kFloat, "step_geometry");
    check(root, at::kInt, "root");
    check(children, at::kInt, "children");
    TORCH_CHECK(geometry.dim() == 2 && geometry.size(0) == n_step && geometry.size(1) == 8, "step_geometry must be (n_step, 8).");
    TORCH_CHECK(root.numel() == rx * ry * rz && children.dim() == 2 && children.size(1) == 8, "root / children shapes.");
    TORCH_CHECK(values.numel() >= 8 * n_tri + 2 * n_step, "values too short for the type counts.");
    TORCH_CHECK(levels >= 1 && levels <= 16, "1 to 16 octree levels are supported.");
    TORCH_CHECK((rx << (levels - 1)) <= 1000000 && (ry << (levels - 1)) <= 1000000, "finest resolution too large.");
    return Tree{values.data_ptr<float>(), geometry.data_ptr<float>(), root.data_ptr<int32_t>(), children.data_ptr<int32_t>(),
                n_tri, n_step, static_cast<int>(levels), static_cast<int>(rx), static_cast<int>(ry), static_cast<int>(rz)};
}

Rays make_rays(const at::Tensor& angles, const at::Tensor& rows, const at::Tensor& cols, int64_t dh, int64_t dw, const Tree& T) {
    check(angles, at::kFloat, "angles");
    check(rows, at::kLong, "rows");
    check(cols, at::kLong, "cols");
    TORCH_CHECK(angles.dim() == 1 && rows.numel() == angles.numel() && cols.numel() == angles.numel(), "ray vectors must match.");
    const int max_segments = (T.rx << (T.levels - 1)) + (T.ry << (T.levels - 1)) + 8;
    return Rays{angles.data_ptr<float>(), rows.data_ptr<int64_t>(), cols.data_ptr<int64_t>(), static_cast<int>(angles.numel()),
                static_cast<int>(dh), static_cast<int>(dw), max_segments};
}

constexpr int THREADS = 128;

void launched(const char* name) {
    const auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, name, " failed: ", cudaGetErrorString(error));
}

}  // namespace

#define TREE_ARGS at::Tensor values, at::Tensor geometry, at::Tensor root, at::Tensor children, int64_t n_tri, int64_t n_step, \
    int64_t levels, int64_t rx, int64_t ry, int64_t rz, at::Tensor angles, at::Tensor rows, at::Tensor cols, int64_t dh, int64_t dw
#define MAKE_TREE_RAYS \
    c10::cuda::CUDAGuard guard(values.device()); \
    const Tree T = make_tree(values, geometry, root, children, n_tri, n_step, levels, rx, ry, rz); \
    const Rays R = make_rays(angles, rows, cols, dh, dw, T); \
    const int blocks = (R.count + THREADS - 1) / THREADS; \
    auto stream = at::cuda::getCurrentCUDAStream().stream();

at::Tensor bernstein_octree_integrate_forward_cuda(TREE_ARGS) {
    MAKE_TREE_RAYS
    auto out = at::empty({angles.size(0)}, values.options());
    if (R.count) forward_kernel<<<blocks, THREADS, 0, stream>>>(T, R, out.data_ptr<float>());
    launched("octree forward kernel");
    return out;
}

at::Tensor bernstein_octree_integrate_backward_cuda(TREE_ARGS, at::Tensor grad_output) {
    MAKE_TREE_RAYS
    check(grad_output, at::kFloat, "grad_output");
    TORCH_CHECK(grad_output.numel() == R.count, "grad_output must match the rays.");
    auto grad = at::zeros_like(values);
    if (R.count) backward_kernel<<<blocks, THREADS, 0, stream>>>(T, R, grad_output.data_ptr<float>(), grad.data_ptr<float>());
    launched("octree backward kernel");
    return grad;
}

at::Tensor bernstein_octree_decision_accumulate_cuda(TREE_ARGS, at::Tensor candidate, at::Tensor residual, at::Tensor fold, at::Tensor accum,
                                                     int64_t leaf_count) {
    MAKE_TREE_RAYS
    check(candidate, at::kFloat, "candidate_geometry");
    check(residual, at::kFloat, "residual");
    check(fold, at::kInt, "fold");
    check(accum, at::kFloat, "accum");
    TORCH_CHECK(candidate.dim() == 2 && candidate.size(0) == leaf_count && candidate.size(1) == 8, "candidate_geometry must be (N, 8).");
    TORCH_CHECK(residual.numel() == R.count && fold.numel() == R.count, "residual / fold must match the rays.");
    TORCH_CHECK(accum.numel() == 2 * leaf_count * DECISION_STRIDE, "accum must be (2, N, 97).");
    if (R.count)
        decision_kernel<<<blocks, THREADS, 0, stream>>>(T, R, candidate.data_ptr<float>(), residual.data_ptr<float>(),
                                                        fold.data_ptr<int32_t>(), accum.data_ptr<float>(), leaf_count);
    launched("octree decision kernel");
    return accum;
}
