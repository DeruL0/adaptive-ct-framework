#include <torch/extension.h>
#include <vector>

at::Tensor project_dense_parallel_forward_cuda(
    at::Tensor volume,
    at::Tensor angles,
    int64_t detector_h,
    int64_t detector_w,
    int64_t samples_per_ray);

#define TREE_ARGS at::Tensor values, at::Tensor geometry, at::Tensor root, at::Tensor children, int64_t n_tri, int64_t n_step,     int64_t levels, int64_t rx, int64_t ry, int64_t rz, at::Tensor angles, at::Tensor rows, at::Tensor cols, int64_t dh, int64_t dw
at::Tensor bernstein_octree_integrate_forward_cuda(TREE_ARGS);
at::Tensor bernstein_octree_integrate_backward_cuda(TREE_ARGS, at::Tensor grad_output);
at::Tensor bernstein_octree_decision_accumulate_cuda(TREE_ARGS, at::Tensor candidate, at::Tensor residual, at::Tensor accum,
                                                     int64_t leaf_count, at::Tensor active);

at::Tensor bernstein_octree_geometry_accumulate_cuda(TREE_ARGS, at::Tensor residual, at::Tensor accum);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "project_dense_parallel_forward",
        &project_dense_parallel_forward_cuda,
        "Project a dense volume with normalized parallel-beam geometry (CUDA)");
    m.def(
        "bernstein_octree_integrate_forward",
        &bernstein_octree_integrate_forward_cuda,
        "Integrate constant / trilinear / step octree leaves along parallel rays (CUDA)");
    m.def(
        "bernstein_octree_integrate_backward",
        &bernstein_octree_integrate_backward_cuda,
        "Backpropagate octree ray integration (CUDA)");
    m.def(
        "bernstein_octree_decision_accumulate",
        &bernstein_octree_decision_accumulate_cuda,
        "Accumulate per-leaf refit statistics of every decision option along parallel rays (CUDA)");
    m.def(
        "bernstein_octree_geometry_accumulate",
        &bernstein_octree_geometry_accumulate_cuda,
        "Accumulate per-step-leaf Gauss-Newton statistics of the step geometry along parallel rays (CUDA)");
}
