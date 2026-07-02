#include <torch/extension.h>
#include <vector>

at::Tensor project_dense_parallel_forward_cuda(
    at::Tensor volume,
    at::Tensor angles,
    int64_t detector_h,
    int64_t detector_w,
    int64_t samples_per_ray);

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
    double attenuation_shift);

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
    double attenuation_shift);

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
    double attenuation_shift);

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
    double attenuation_shift);

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
    double attenuation_shift);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "project_dense_parallel_forward",
        &project_dense_parallel_forward_cuda,
        "Project a dense volume with normalized parallel-beam geometry (CUDA)");
    m.def(
        "dynamic_voxel_integrate_forward",
        &dynamic_voxel_integrate_forward_cuda,
        "Integrate dynamic leaf voxels along normalized parallel rays (CUDA)");
    m.def(
        "dynamic_voxel_integrate_backward",
        &dynamic_voxel_integrate_backward_cuda,
        "Backpropagate dynamic leaf-voxel ray integration (CUDA)");
    m.def(
        "bernstein_octree_integrate_forward",
        &bernstein_octree_integrate_forward_cuda,
        "Integrate anisotropic Bernstein octree leaves along parallel rays (CUDA)");
    m.def(
        "bernstein_octree_integrate_backward",
        &bernstein_octree_integrate_backward_cuda,
        "Backpropagate anisotropic Bernstein octree ray integration (CUDA)");
    m.def(
        "bernstein_octree_segments_forward",
        &bernstein_octree_segments_forward_cuda,
        "Return per-ray Bernstein octree leaf contributions (CUDA)");
}
