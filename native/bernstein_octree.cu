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

constexpr int THREADS = 128;  // threads per block (the decision kernel uses DECISION_THREADS)

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
// Terms that are zero for the whole group are skipped (most of the split option's child products: a segment
// crosses 1 to 3 of the 8 children).
__device__ __forceinline__ void warp_add(const cg::coalesced_group& peers, float* slot, float value) {
    if (!peers.any(value != 0.0f)) return;
    if (peers.size() > 1) value = cg::reduce(peers, value, cg::plus<float>());
    if (peers.thread_rank() == 0) atomicAdd(slot, value);
}

// Threads of the warp at segments with the same key (leaf) split the n columns of their rows (shared memory, row
// stride `stride`, odd to avoid bank conflicts) among themselves: each sums its columns over the group's rows and
// adds them once.  Far fewer operations than one warp reduction per column (decision statistics 300 -> 210 ms).
__device__ void group_add(const float* shared_rows, int n, int stride, unsigned long long key, float* slot) {
    const unsigned active = __activemask();
    const unsigned group = __match_any_sync(active, key);
    __syncwarp(active);
    const int lane = threadIdx.x & 31;
    const int rank = __popc(group & ((1u << lane) - 1u)), size = __popc(group);
    const float* warp_rows = shared_rows + (threadIdx.x - lane) * stride;
    for (int e = rank; e < n; e += size) {
        float sum = 0.0f;
        for (unsigned m = group; m; m &= m - 1) sum += warp_rows[(__ffs(m) - 1) * stride + e];
        if (sum != 0.0f) atomicAdd(slot + e, sum);
    }
    __syncwarp(active);
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
    bool push[2] = {false, false};  // the segment starts on a cell face of this axis
    for (int axis = 0; axis < 2; ++axis) {
        if (fabsf(d2[axis]) < 1.0e-8f) {
            if (b2[axis] < -1.0f || b2[axis] > 1.0f) return;
            continue;
        }
        const float t0 = (-1.0f - b2[axis]) / d2[axis], t1 = (1.0f - b2[axis]) / d2[axis];
        if (fminf(t0, t1) > tmin) {
            tmin = fminf(t0, t1);
            push[0] = push[1] = false;
            push[axis] = true;
        }
        tmax = fminf(tmax, fmaxf(t0, t1));
    }
    if (z < -1.0f || z > 1.0f || tmax <= tmin) return;
    const float eps = 1.0e-6f;
    // The cell after a crossing is found from a probe point pushed PUSH across the face just crossed (along that
    // axis only): nudging along the ray instead fails at shallow angles, where the slow coordinate moves less than
    // its float resolution (seen at 0.47 degrees: segments assigned to the cell before the crossing).
    const float PUSH = 1.0e-5f;  // far below the finest cell (2/768), far above float resolution near 1
    float t = tmin;
    for (int n = 0; n < R.max_segments && t < tmax - eps; ++n) {
        // The probe can still sit on a face of the other axis (rounding): then push that axis as well and look again.
        int r[3], c[3];
        int64_t leaf = -1;
        float cx = 1.0e30f, cy = 1.0e30f;
        for (int attempt = 0; attempt < 3; ++attempt) {
            const float px = bx + dx * t + (push[0] ? copysignf(PUSH, dx) : 0.0f);
            const float py = by + dy * t + (push[1] ? copysignf(PUSH, dy) : 0.0f);
            leaf = lookup_leaf(T, px, py, z, r, c);
            if (leaf < 0) return;
            cx = dx > eps ? (-1.0f + 2.0f * (c[0] + 1) / r[0] - bx) / dx : (dx < -eps ? (-1.0f + 2.0f * c[0] / r[0] - bx) / dx : 1.0e30f);
            cy = dy > eps ? (-1.0f + 2.0f * (c[1] + 1) / r[1] - by) / dy : (dy < -eps ? (-1.0f + 2.0f * c[1] / r[1] - by) / dy : 1.0e30f);
            const bool again_x = cx <= t + eps && !push[0], again_y = cy <= t + eps && !push[1];
            if (!again_x && !again_y) break;
            push[0] |= again_x;
            push[1] |= again_y;
        }
        const float next = fminf(tmax, fminf(cx > t + eps ? cx : 1.0e30f, cy > t + eps ? cy : 1.0e30f));
        if (!(next > t)) return;
        push[0] = cx == next;
        push[1] = cy == next;
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
    __shared__ float rows[THREADS * 9];  // one row per thread: the segment's up to 8 gradient terms (odd stride)
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= R.count) return;
    const float g = grad_out[ray];
    traverse(T, R, ray, [&](const Segment& s) {
        float w[8];
        int64_t off;
        const int n = segment_weights(T, s, w, &off);
        for (int k = 0; k < n; ++k) rows[threadIdx.x * 9 + k] = g * w[k];
        group_add(rows, n, 9, static_cast<unsigned long long>(s.leaf), grad_values + off);
    });
}

// Decision statistics (refinement/decision.py): per (ray, leaf) segment the basis integrals of every option
//   constant: L · trilinear: 8 corner basis integrals · step: (L - I, I) with the candidate geometry ·
//   split: length inside each of the 8 children (current function + one constant per child)
// and b = (this leaf's current contribution) - (ray residual), the data seen by a refit of this leaf with all
// other leaves fixed.  Accumulated per leaf into DECISION_STRIDE floats (one view per call; the caller adds the
// views up in double: statistics that are subtracted later lose their digits when summed in float):
//   0 L*L, 1 L*b | 2..37 trilinear G (upper, row major), 38..45 trilinear*b | 46..48 step G (00, 01, 11),
//   49..50 step*b | 51..86 split G (upper), 87..94 split*(-residual) | 95 b*b | 96 residual^2 |
//   97..99 current-geometry step G (00, 01, 11), 100..101 current step*b (zero for leaves that are not steps)
constexpr int DECISION_STRIDE = 102;
constexpr int DECISION_THREADS = 64;
// The 102 columns go through shared memory 35 at a time (odd row stride: no bank conflicts), so a block needs 9 KB
// instead of 26 KB: 3 x more blocks per SM (63 -> 96 registers, no spill).  Measured on a 1.0 M-leaf state: decision
// kernel 133-160 -> 66 ms per view, same statistics.
constexpr int DECISION_WINDOW = 35;

__device__ void segment_decision(const Tree& T, const Segment& s, const float* candidate, float* accum, int64_t leaf_count,
                                 float residual, float* shared_rows, const int* slot) {
    const int out = slot[s.leaf];  // row of this leaf in accum; < 0: no statistics (group_add pairs the active threads only)
    if (out < 0) return;
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
    float cur[2] = {0.0f, 0.0f};  // the leaf's own step with its current geometry: the baseline of a step leaf's options
    if (leaf_kind(T, s.leaf) == 2) {
        const float Ic = step_integral(T.geometry + 8 * (s.leaf - T.n_tri), s);
        cur[0] = length - Ic;
        cur[1] = Ic;
    }
    float* row = shared_rows + threadIdx.x * DECISION_WINDOW;  // this thread's columns lo..hi, then summed per leaf
    for (int lo = 0; lo < DECISION_STRIDE; lo += DECISION_WINDOW) {
        const int hi = min(lo + DECISION_WINDOW, DECISION_STRIDE);
        int k = 0;
        auto put = [&](float v) {  // the k-th column, written only when it falls in this window
            if (k >= lo && k < hi) row[k - lo] = v;
            ++k;
        };
        put(length * length);
        put(length * b);
        for (int i = 0; i < 8; ++i)
            for (int j = i; j < 8; ++j) put(tri[i] * tri[j]);
        for (int i = 0; i < 8; ++i) put(tri[i] * b);
        put(st[0] * st[0]);
        put(st[0] * st[1]);
        put(st[1] * st[1]);
        put(st[0] * b);
        put(st[1] * b);
        for (int i = 0; i < 8; ++i)
            for (int j = i; j < 8; ++j) put(child[i] * child[j]);
        for (int i = 0; i < 8; ++i) put(-child[i] * residual);
        put(b * b);
        put(residual * residual);
        put(cur[0] * cur[0]);
        put(cur[0] * cur[1]);
        put(cur[1] * cur[1]);
        put(cur[0] * b);
        put(cur[1] * b);
        group_add(shared_rows, hi - lo, DECISION_WINDOW, static_cast<unsigned long long>(s.leaf),
                  accum + static_cast<int64_t>(out) * DECISION_STRIDE + lo);
    }
}

__global__ void decision_kernel(Tree T, Rays R, const float* candidate, const float* residual, float* accum,
                                int64_t leaf_count, const int* slot) {
    __shared__ float rows[DECISION_THREADS * DECISION_WINDOW];
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= R.count) return;
    traverse(T, R, ray, [&](const Segment& s) { segment_decision(T, s, candidate, accum, leaf_count, residual[ray], rows, slot); });
}

// Step geometry normal equations (training/geometry.py): for every segment in a step leaf the derivative of the
// segment's contribution c0 (L - I) + c1 I with respect to the geometry (n (3), d, w),
//   D = (c1 - c0) dI/dtheta,  I = L F(a0, a1),  F = (softplus(a1) - softplus(a0)) / (a1 - a0),  a = (n . v - d) / w,
// v = u - 1/2 at the segment ends.  Per step leaf: GEOMETRY_STATS = 5 gradient (sum residual * D) + 15 (upper
// triangle of sum D D^T, row major).  Hard steps (w = 0) have no derivative and are skipped.
constexpr int GEOMETRY_STATS = 20;
constexpr int GEOMETRY_ROW = 21;  // shared-memory row stride (odd: no bank conflicts)

__device__ void segment_geometry(const Tree& T, const Segment& s, float residual, float* accum, float* shared_rows) {
    const int64_t j = s.leaf - T.n_tri;
    const float* row = T.geometry + 8 * j;
    const float4 g = reinterpret_cast<const float4*>(row)[0];
    const float w = row[4];
    if (w <= 0.0f) return;
    const float2 c = reinterpret_cast<const float2*>(T.values + 8 * T.n_tri + 2 * j)[0];
    const float v0[3] = {s.ux(s.t0) - 0.5f, s.uy(s.t0) - 0.5f, s.uz() - 0.5f};
    const float v1[3] = {s.ux(s.t1) - 0.5f, s.uy(s.t1) - 0.5f, v0[2]};
    const float a0 = (g.x * v0[0] + g.y * v0[1] + g.z * v0[2] - g.w) / w;
    const float a1 = (g.x * v1[0] + g.y * v1[1] + g.z * v1[2] - g.w) / w;
    float F0, F1;  // dF/da0, dF/da1
    if (fabsf(a1 - a0) < 1.0e-2f) {
        const float m = sigmoid_stable(0.5f * (a0 + a1));
        F0 = F1 = 0.5f * m * (1.0f - m);
    } else {
        const float F = (softplus_stable(a1) - softplus_stable(a0)) / (a1 - a0);
        F0 = (F - sigmoid_stable(a0)) / (a1 - a0);
        F1 = (sigmoid_stable(a1) - F) / (a1 - a0);
    }
    const float k = (c.y - c.x) * (s.t1 - s.t0) / w;
    const float D[5] = {k * (F0 * v0[0] + F1 * v1[0]), k * (F0 * v0[1] + F1 * v1[1]), k * (F0 * v0[2] + F1 * v1[2]),
                        -k * (F0 + F1), -k * (F0 * a0 + F1 * a1)};
    float* out = shared_rows + threadIdx.x * GEOMETRY_ROW;
    for (int i = 0; i < 5; ++i) out[i] = residual * D[i];
    int n = 5;
    for (int i = 0; i < 5; ++i)
        for (int l = i; l < 5; ++l) out[n++] = D[i] * D[l];
    group_add(shared_rows, GEOMETRY_STATS, GEOMETRY_ROW, static_cast<unsigned long long>(j), accum + j * GEOMETRY_STATS);
}

__global__ void geometry_kernel(Tree T, Rays R, const float* residual, float* accum) {
    __shared__ float rows[THREADS * GEOMETRY_ROW];
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= R.count) return;
    const float r = residual[ray];
    traverse(T, R, ray, [&](const Segment& s) {
        if (leaf_kind(T, s.leaf) == 2) segment_geometry(T, s, r, accum, rows);
    });
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

at::Tensor bernstein_octree_decision_accumulate_cuda(TREE_ARGS, at::Tensor candidate, at::Tensor residual, at::Tensor accum,
                                                     int64_t leaf_count, at::Tensor slot) {
    MAKE_TREE_RAYS
    check(candidate, at::kFloat, "candidate_geometry");
    check(residual, at::kFloat, "residual");
    check(accum, at::kFloat, "accum");
    TORCH_CHECK(candidate.dim() == 2 && candidate.size(0) == leaf_count && candidate.size(1) == 8, "candidate_geometry must be (N, 8).");
    TORCH_CHECK(residual.numel() == R.count, "residual must match the rays.");
    check(slot, at::kInt, "slot");
    TORCH_CHECK(slot.numel() == leaf_count, "slot must be (N,).");
    TORCH_CHECK(accum.dim() == 2 && accum.size(1) == DECISION_STRIDE, "accum must be (rows, 102).");
    if (R.count)
        decision_kernel<<<(R.count + DECISION_THREADS - 1) / DECISION_THREADS, DECISION_THREADS, 0, stream>>>(T, R, candidate.data_ptr<float>(), residual.data_ptr<float>(),
                                                        accum.data_ptr<float>(), leaf_count, slot.data_ptr<int>());
    launched("octree decision kernel");
    return accum;
}

at::Tensor bernstein_octree_geometry_accumulate_cuda(TREE_ARGS, at::Tensor residual, at::Tensor accum) {
    MAKE_TREE_RAYS
    check(residual, at::kFloat, "residual");
    check(accum, at::kFloat, "accum");
    TORCH_CHECK(residual.numel() == R.count, "residual must match the rays.");
    TORCH_CHECK(accum.numel() == n_step * GEOMETRY_STATS, "accum must be (n_step, 20).");
    if (R.count && n_step)
        geometry_kernel<<<blocks, THREADS, 0, stream>>>(T, R, residual.data_ptr<float>(), accum.data_ptr<float>());
    launched("octree geometry kernel");
    return accum;
}
