#include <torch/extension.h>
#include <vector>

at::Tensor project_dense_parallel_forward_cuda(
    at::Tensor volume,
    at::Tensor angles,
    int64_t detector_h,
    int64_t detector_w,
    int64_t samples_per_ray);

at::Tensor bernstein_octree_integrate_forward_cuda(
    at::Tensor coefficient_logits,
    at::Tensor leaf_degrees,
    at::Tensor coefficient_offsets,
    at::Tensor step_geometry,
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
    double attenuation_shift);

at::Tensor bernstein_octree_integrate_backward_cuda(
    at::Tensor coefficient_logits,
    at::Tensor leaf_degrees,
    at::Tensor coefficient_offsets,
    at::Tensor step_geometry,
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
    double attenuation_shift);

std::vector<at::Tensor> bernstein_octree_segments_forward_cuda(
    at::Tensor coefficient_logits,
    at::Tensor leaf_degrees,
    at::Tensor coefficient_offsets,
    at::Tensor step_geometry,
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
    double attenuation_shift);

at::Tensor bernstein_octree_decision_accumulate_cuda(
    at::Tensor coefficient_logits,
    at::Tensor leaf_degrees,
    at::Tensor coefficient_offsets,
    at::Tensor step_geometry,
    at::Tensor candidate_geometry,
    at::Tensor node_child_base,
    at::Tensor node_leaf_id,
    at::Tensor angles,
    at::Tensor rows,
    at::Tensor cols,
    at::Tensor residual,
    at::Tensor fold,
    at::Tensor accum,
    int64_t detector_h,
    int64_t detector_w,
    int64_t num_levels,
    int64_t root_x,
    int64_t root_y,
    int64_t root_z,
    double attenuation_shift);

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
        "bernstein_octree_segments_forward",
        &bernstein_octree_segments_forward_cuda,
        "Return per-ray Bernstein octree leaf contributions (CUDA)");
    m.def(
        "bernstein_octree_decision_accumulate",
        &bernstein_octree_decision_accumulate_cuda,
        "Accumulate per-leaf refit statistics of every decision option along parallel rays (CUDA)");
}
