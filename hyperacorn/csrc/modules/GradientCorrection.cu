
#include "saiga/cuda/cuda.h"
#include "saiga/vision/torch/CudaHelper.h"
#include "saiga/vision/torch/EigenTensor.h"

#include "GradientCorrection.h"

#include <torch/autograd.h>

#include <torch/csrc/autograd/custom_function.h>

using namespace Saiga;

#undef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#undef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit - 1), MAX(in, 0))
#define WITHIN_BOUNDS(x, y, z, D, H, W) (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D)


static __global__ void GradientCorrectionBackwardCUDAKernel(StaticDeviceTensor<float, 2> sample_position,
                                                            StaticDeviceTensor<float, 2> ray_direction,
                                                            StaticDeviceTensor<long, 1> index,
                                                            StaticDeviceTensor<float, 2> grad_sample_position,
                                                            StaticDeviceTensor<float, 2> grad_output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= index.sizes[0]) return;

    vec3 direction;
    vec3 grad;

    long ray_id = index(i);

    for (int j = 0; j < 3; ++j)
    {
        direction(j) = ray_direction((int)ray_id, j);
        grad(j)      = grad_output(i, j);
    }

    grad = grad - grad.dot(direction) * direction;
    for (int j = 0; j < 3; ++j)
    {
        grad_sample_position(i, j) = grad(j);
    }
}
static void GradientCorrectionBackwardCUDA(torch::Tensor sample_position, torch::Tensor ray_direction,
                                           torch::Tensor index, torch::Tensor& grad_sample_position,
                                           torch::Tensor grad_output)
{
    int num_samples = sample_position.size(0);
    if (num_samples > 0)
    {
        GradientCorrectionBackwardCUDAKernel<<<iDivUp(num_samples, 128), 128>>>(sample_position, ray_direction, index,
                                                                                grad_sample_position, grad_output);
    }
}


namespace torch::autograd
{
std::vector<torch::Tensor> GradientCorrection::forward(AutogradContext* ctx, torch::Tensor sample_position,
                                                       torch::Tensor ray_direction, torch::Tensor index)
{
    std::vector<torch::Tensor> input;
    input.push_back(sample_position);
    input.push_back(ray_direction);

    if (index.defined())
    {
        input.push_back(index);
    }
    ctx->save_for_backward(input);


    std::vector<torch::Tensor> result;
    result.push_back(sample_position);
    return result;
}

std::vector<torch::Tensor> GradientCorrection::backward(AutogradContext* ctx, std::vector<torch::Tensor> grad_output)
{
    std::vector<torch::Tensor> input = ctx->get_saved_variables();
    auto sample_position             = input[0];
    auto ray_direction               = input[1];
    auto index                       = input[2];

    CHECK_EQ(grad_output.size(), 1);

    auto grad_sample_position = torch::zeros_like(sample_position);
    grad_sample_position.set_requires_grad(1);

    if (sample_position.is_cpu())
    {
        CHECK(false);
    }
    else if (sample_position.is_cuda())
    {
        GradientCorrectionBackwardCUDA(sample_position, ray_direction, index, grad_sample_position, grad_output[0]);
        CUDA_SYNC_CHECK_ERROR();
    }
    else
    {
        CHECK(false);
    }
    // std::cout << "backward output" << std::endl;
    // PrintTensorInfo(grad_multi_grid);
    // PrintTensorInfo(grad_uv);

    return {grad_sample_position, torch::Tensor(), torch::Tensor()};
}
}  // namespace torch::autograd

torch::Tensor GradientCorrection(torch::Tensor sample_position, torch::Tensor ray_direction, torch::Tensor index)
{
    // std::cout << "IndirectGridSample3D" << std::endl;
    // PrintTensorInfo(multi_grid);
    // PrintTensorInfo(index);
    // PrintTensorInfo(uv);
    auto result = torch::autograd::GradientCorrection::apply(sample_position, ray_direction, index);
    CHECK_EQ(result.size(), 1);
    return result.front();
}