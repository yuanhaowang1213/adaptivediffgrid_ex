#pragma once
#include <torch/torch.h>
namespace F = torch::nn::functional;
class Laplacian2DImpl : public torch::nn::Module
// class Laplacian2DImpl
{
    public:
    Laplacian2DImpl(int size, int channel)
    {
        channel_in = channel;
        float sig = size/(4. * std::sqrt(2 * std::log(2)));
//        torch::Tensor sigma = torch::from_blob({sig}, {1}, torch::TensorOptions(torch::kFloat32));
//        at::Tensor sigma = torch::tensor({sig});
//        gaussian_filter = torch::zeros({channel, 1, size, size}, torch::TensorOptions(torch::kFloat));
        auto x_ind = torch::arange(- (size-1)/2., (size+1)/2., 1.);
//        std::cout << "x ind is " ;
        // std::cout << x_ind << std::endl;
        std::vector<torch::Tensor> mesh2d = torch::meshgrid({x_ind, x_ind});
        torch::Tensor x = mesh2d[0].contiguous();
        torch::Tensor y = mesh2d[1].contiguous();
//        torch::Tensor gaussian_filter = torch::exp(- (x * x /2/sigma^2 + y * y/2/sigma^2  ));
        gaussian_filter = torch::exp(- (x * x + y * y)/(2 * sig * sig));
        gaussian_filter = gaussian_filter/gaussian_filter.sum();
        gaussian_filter = gaussian_filter.repeat({channel, 1, 1,1});
        // std::cout << gaussian_filter << std::endl;
        // gaussian_weights = register_parameter("gaussian_weights", gaussian_weights);
    }

    torch::Tensor downsample(torch::Tensor img)
    {
        img = F::pad(img, F::PadFuncOptions({2,2,2,2}).mode(torch::kReplicate));
        auto out = F::conv2d(img, gaussian_filter, F::Conv2dFuncOptions().stride(2).groups(img.size(1)));
        return out;
    }

    torch::Tensor upsample(torch::Tensor img)
    {
        auto cc = torch::cat({img, torch::zeros({img.size(0), img.size(1), img.size(2), img.size(3)})},3);
        cc = cc.view({img.size(0), img.size(1), img.size(2)*2, img.size(3)});
        cc = cc.permute({0,1,3,2});
        cc = torch::cat({cc, torch::zeros({img.size(0), img.size(1), img.size(3), img.size(2) *2})}, 3);
        cc = cc.view({img.size(0),img.size(1), img.size(3)*2, img.size(2)*2});
        cc = cc.permute({0,1,3,2});
        cc = F::pad(cc, F::PadFuncOptions({2,2,2,2}).mode(torch::kReplicate));
        return F::conv2d(cc, 4*gaussian_filter, F::Conv2dFuncOptions().groups(img.size(1)));
    }

    torch::Tensor gaussian_filter;
    int channel_in;
};
TORCH_MODULE(Laplacian2D);
