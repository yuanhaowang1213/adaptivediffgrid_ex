#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/VisionTypes.h"

#include <torch/torch.h>


namespace torch::autograd
{
struct GradientCorrection : public Function<GradientCorrection>
{
    // returns a tensor for every layer
    static std::vector<torch::Tensor> forward(AutogradContext* ctx, torch::Tensor sample_position,
                                              torch::Tensor ray_direction, torch::Tensor index);

    static std::vector<torch::Tensor> backward(AutogradContext* ctx, std::vector<torch::Tensor> grad_output);
};
}  // namespace torch::autograd

// Input:
//      sample_position: [..., 3]
torch::Tensor GradientCorrection(torch::Tensor sample_position, torch::Tensor ray_direction, torch::Tensor index);
