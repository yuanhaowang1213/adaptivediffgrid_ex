#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/torch/TorchHelper.h"
#include <torch/script.h>
#include <torch/torch.h>

// torch::nn::Conv3d conv3d1x1(
//     int in,
//     int out,
//     int stride,
//     int groups){
//     torch::nn::Conv3dOptions O(in, out, 1);
    
    
// }

class NonLocalImpl : public torch::nn::Module
{
    public:
    NonLocalImpl(int in_features)
        : in_features(in_features)
    {
        // Currently only support Gaussian
        inter_channels = in_features/2;
        if (inter_channels == 0)
        inter_channels = 1;

        // ConvOptions(int64_t in_channels, int64_t out_channels, ExpandingArray<D> kernel_size)
        //g = torch::nn::Conv3d(Conv3dOptions(3, 2, 3).stride(1).bias(false));
	//g.push_back(torch::nn::Conv3d(
	    // g->push_back();
        conv_g = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_features, inter_channels ,1));
        register_module("conv_g", conv_g);
        conv_theta=torch::nn::Conv3d(torch::nn::Conv3dOptions(in_features, inter_channels ,1));
        conv_phi = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_features, inter_channels ,1));
        register_module("conv_theta", conv_theta);
        register_module("conv_phi", conv_phi);

        conv_mask = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_features, inter_channels ,1));
	register_module("conv_mask", conv_mask);
        // auto convlayer = torch::nn::Conv3d(torch::nn::Conv3dOptions(in_features, inter_channels ,1));
        // W_z->push_back(convlayer);
        // auto batchlayer = torch::nn::BatchNorm3d(torch::nn::BatchNorm3dOptions(in_features));
        // torch::nn::init::constant_(batchlayer->weight, 0.);
        // torch::nn::init::constant_(batchlayer->bias, 0.);
        // W_z->push_back(batchlayer);

        // register_module("W_z", W_z);



    }


    at::Tensor forward(at::Tensor x)
    {           
        // CHECK_EQ(in_features, x.size(-1));
        // int batch_size = x.size(0);
        // auto g_x = g->forward(x).view({batch_size, inter_channels,-1});
        // g_x = g_x.permute({0, 2, 1});

        // auto theta_x = x.view({batch_size, in_features, -1});
        // auto phi_x   = x.view({batch_size, in_features, -1});
        // theta_x = theta_x.permute({0,2,1});
        // auto f = torch::matmul(theta_x, phi_x);

        // f = torch::softmax(f, -1);

        // auto y = torch::matmul(f, g_x);
        // y = y.permute({0, 2, 1}).contiguous();
        // // y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        // y = y.view({batch_size, inter_channels, x.size(2), x.size(3), x.size(4)});

        // auto W_y = W_z->forward(y);
        // auto z = W_y + x;
        
        // x in put [b c D W H]

   	int b = x.size(0);
        int c = x.size(1);
	int half_c;
	if(c == 1)
	{
		half_c = 1;
	}
	else
	{
		half_c = c/2;
	}
	
        int D = x.size(2);
        int W = x.size(3);
        int H = x.size(4);

        auto x_phi   = conv_phi(x).view({b, half_c, -1});
        auto x_theta = conv_theta(x).view({b, half_c, -1}).permute({0,2,1}).contiguous();
        // auto x_g     = conv_g(x),view({b, c/2, -1}).permute({0,2,1}).contiguous();

        auto multi_  = torch::matmul(x_theta, x_phi);
        multi_       = torch::softmax(multi_, 1);

        x_phi        = conv_g(x).view({b,half_c, -1}).permute({0,2,1}).contiguous();

        // multi_       = torch.matmul(mult_, x_g);
        multi_       = torch::matmul(multi_, x_phi);

        multi_        = multi_.permute({0,2,1}).contiguous().view({b,inter_channels, D, W, H });

        multi_        = conv_mask(multi_);



        return multi_;


    }

    

    int in_features, inter_channels;
    // torch::nn::Sequential g;
    // torch::nn::Conv3d g, theta, phi;
    // torch::nn::Sequential g, theta, phi;
    torch::nn::Conv3d conv_g{nullptr},conv_theta{nullptr}, conv_phi{nullptr}, conv_mask{nullptr};
    // torch::nn::Softmax softmax{nullptr};
    // torch::nn::Sequential W_z;
};
TORCH_MODULE(NonLocal);

class NonLocal2DImpl : torch::nn::Module
{
	public:
	NonLocal2DImpl(int in_features)
		: in_features(in_features)
	{
		inter_channels = in_features/2;
		if(inter_channels == 0)
		{
			inter_channels = 1;
		}
		conv_g = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_features, inter_channels, 1));
		register_module("conv_g", conv_g);
		conv_theta = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_features, inter_channels,1));
		conv_phi   = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_features, inter_channels,1));
		register_module("conv_theta", conv_theta);
		register_module("conv_phi", conv_phi);
		conv_mask  = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_features, inter_channels,1));
		register_module("conv_mask", conv_mask);
	}

	at::Tensor forward(at::Tensor x)
	{
		int b = x.size(0);
		int c = x.size(1);
		int half_c;
		if(c==1)
		{
			half_c = 1;
		}
		else
		{
			half_c = c/2;
		}
		int W = x.size(2);
		int H = x.size(3);
		auto x_phi = conv_phi(x).view({b, half_c, -1});
		auto x_theta = conv_theta(x).view({b, half_c,-1}).permute({0,2,1}).contiguous();

		auto multi_ = torch::matmul(x_theta, x_phi);
		multi_	    = torch::softmax(multi_, 1);

		x_phi	    = conv_g(x).view({b,half_c, -1}).permute({0,2,1}).contiguous();

		multi_	    = torch::matmul(multi_, x_phi);
		multi_	    = multi_.permute({0,2,1}).contiguous().view({b,inter_channels, W, H});
		multi_	    = conv_mask(multi_);
		return multi_;	
	}
	int in_features, inter_channels;
	torch::nn::Conv2d conv_g{nullptr}, conv_theta{nullptr}, conv_phi{nullptr}, conv_mask{nullptr};
};
TORCH_MODULE(NonLocal2D);
// Refering https://github.com/Deep-Imaging-Group/DRGAN-OCT/blob/main/models/base_blocks.py
// class NonLocalBlock(nn.Module):
//     def __init__(self, channel):
//         super(NonLocalBlock, self).__init__()
//         self.inter_channel = channel // 2
//         self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
//         self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
//         self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
//         self.softmax = nn.Softmax(dim=1)
//         self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

//     def forward(self, x):
//         # [N, C, H , W]
//         b, c, h, w = x.size()
//         # [N, C/2, H * W]
//         x_phi = self.conv_phi(x).view(b, c // 2, -1)
//         # [N, H * W, C/2]
//         x_theta = self.conv_theta(x).view(b, c // 2, -1).permute(0, 2, 1).contiguous()
//         x_g = self.conv_g(x).view(b, c // 2, -1).permute(0, 2, 1).contiguous()
//         # [N, H * W, H * W]
//         mul_theta_phi = torch.matmul(x_theta, x_phi)
//         mul_theta_phi = self.softmax(mul_theta_phi)
//         # [N, H * W, C/2]
//         mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
//         # [N, C/2, H, W]
//         mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
//         # [N, C, H , W]
//         mask = self.conv_mask(mul_theta_phi_g)
//         #out = mask + x
//         return mask

class NonLocallLoss
{
    public:
    // torch::Tensor forward(torch::Tensor grid)
    // {
    //     int num_channels = grid.size(-1);
    //     int num_node     = grid.size(0);
    //     torch::Tensor total_loss;
    //     for(int i = 0; i < num_node; ++i)
    //     {
    //         // printf("test here\n");
    //         auto local_grid = grid.slice(0,i,i+1).squeeze(0);
    //         // PrintTensorInfo(local_grid);
    //         for(int j = 0; j < num_channels; ++j)
    //         {
    //             auto local_grid_c = local_grid.slice(-1,j, j+1);
    //             // PrintTensorInfo(local_grid_c);

    //             auto local_grid_c_phi = local_grid_c.permute({1,0}).contiguous();
    //             // PrintTensorInfo(local_grid_c_phi);

    //             auto multi_           = torch::matmul(local_grid_c, local_grid_c_phi);
    //             // PrintTensorInfo(multi_);
    //             multi_                = torch::softmax(multi_, 1);
    //             // PrintTensorInfo(multi_);

    //             multi_                = torch::matmul(multi_, local_grid_c).permute({1,0}).contiguous();
    //             // PrintTensorInfo(multi_);

    //             auto loss = (multi_ - local_grid_c_phi).abs().mean();
    //             if (total_loss.defined())
    //             {
    //                 total_loss += loss;
    //             }
    //             else
    //             {
    //                 total_loss = loss;
    //             }
    //         }
    //     }
    //     return total_loss;
    // }


    torch::Tensor forward(torch::Tensor grid,torch::Tensor weight = {})
    {

        int num_channels = grid.size(-1);
        int num_node     = grid.size(0);
        torch::Tensor total_loss;
        for(int i = 0; i < num_node; ++i)
        {
            // printf("test here\n");
            // [grid_size, feature_size]
            auto local_grid = grid.slice(0,i,i+1).squeeze(0);

            auto local_grid_c_phi = local_grid.permute({1,0}).contiguous();
                // PrintTensorInfo(local_grid_c_phi);

            auto multi_           = torch::matmul(local_grid, local_grid_c_phi);
            // PrintTensorInfo(multi_);
            multi_                = torch::softmax(multi_, -1);

            multi_                = torch::matmul(multi_, local_grid_c_phi.permute({1,0})).contiguous();
            // multi_                = torch::matmul(multi_, local_grid).permute({1,0}).contiguous();
            // PrintTensorInfo(multi_);
            torch::Tensor loss= (multi_ - local_grid).abs().mean();
            if(weight.defined())
            {
                auto weighhere = weight.slice(0,i,i+1).squeeze(0);
                loss *= weighhere;
            }
            // PrintTensorInfo(loss);
            if (total_loss.defined())
            {
                total_loss += loss;
            }
            else
            {
                total_loss = loss;
            }

            // test here
            // local_grid_c_phi Tensor [6, 216] float cuda:0 Min/Max -0.05355 0.0402973 Mean -0.00347707 Sum -4.50629 sdev 0.0109905 req-grad 1
            // multi_ Tensor [216, 216] float cuda:0 Min/Max -0.00201114 0.00347553 Mean 0.000189949 Sum 8.86224 sdev 0.00032186 req-grad 1
            // multi_ Tensor [216, 6] float cuda:0 Min/Max -0.0108283 0.00300696 Mean -0.00347748 Sum -4.50681 sdev 0.00442572 req-grad 1
            // loss Scalar Tensor float cuda:0 req-grad 1 Value: 0.0441197

        }
        // PrintTensorInfo(loss);
        return total_loss * num_channels;
    }

    torch::Tensor forward_cross(torch::Tensor grid, std::vector<long> in_roi_node_vec,torch::Tensor weight = {})
    {
        // PrintTensorInfo(in_roi_index);
        int num_channels = grid.size(-1);
        int num_node     = grid.size(0);
        torch::Tensor total_loss;
        for(int i = 0; i < num_node; ++i)
        {
            // auto weighhere = in_roi_index.slice(0,i,i+1).squeeze(0);
            // if(!torch::equal(weighhere,zero_tensor ) )
            {
                // PrintTensorInfo(weighhere);
            // printf("test here\n");
                // [grid_size, feature_size]
                auto local_grid = grid.slice(0,i,i+1).squeeze(0);
                // PrintTensorInfo(local_grid);
                // auto local_grid_c_phi = local_grid.permute({1,0}).contiguous();
                // std::cout << "slice number " << in_roi_node_vec[i] << std::endl;
                auto local_grid_c_phi = grid.slice(0, (int)in_roi_node_vec[i], (int)in_roi_node_vec[i]+1).squeeze(0);
                // PrintTensorInfo(local_grid_c_phi);
                local_grid_c_phi = local_grid_c_phi.permute({1,0}).contiguous();
                    // PrintTensorInfo(local_grid_c_phi);

                auto multi_           = torch::matmul(local_grid, local_grid_c_phi);
                // PrintTensorInfo(multi_);
                multi_                = torch::softmax(multi_, -1);

                multi_                = torch::matmul(multi_, local_grid_c_phi.permute({1,0})).contiguous();
                // multi_                = torch::matmul(multi_, local_grid).permute({1,0}).contiguous();
                // PrintTensorInfo(multi_);
                torch::Tensor loss= (multi_ - local_grid).abs().mean();
                if(weight.defined())
                {
                    auto weighhere = weight.slice(0,i,i+1).squeeze(0);
                    loss *= weighhere;
                }
                // PrintTensorInfo(loss);
                if (total_loss.defined())
                {
                    total_loss += loss;
                }
                else
                {
                    total_loss = loss;
                }

            }
            // else{
            //     printf("not in roi\n");
            //     PrintTensorInfo(weighhere);
            // }
            // test here
            // local_grid_c_phi Tensor [6, 216] float cuda:0 Min/Max -0.05355 0.0402973 Mean -0.00347707 Sum -4.50629 sdev 0.0109905 req-grad 1
            // multi_ Tensor [216, 216] float cuda:0 Min/Max -0.00201114 0.00347553 Mean 0.000189949 Sum 8.86224 sdev 0.00032186 req-grad 1
            // multi_ Tensor [216, 6] float cuda:0 Min/Max -0.0108283 0.00300696 Mean -0.00347748 Sum -4.50681 sdev 0.00442572 req-grad 1
            // loss Scalar Tensor float cuda:0 req-grad 1 Value: 0.0441197

        }
        // PrintTensorInfo(loss);
        return total_loss * num_channels;
    }
    torch::Tensor forward_total(torch::Tensor grid)
    {
        int num_channels = grid.size(-1);
        int num_node     = grid.size(0);
        torch::Tensor total_loss;
        for(int i = 0; i < num_node; ++i)
        {   
            auto local_grid = grid.slice(0,i,i+1).squeeze(0);
            for(int j = i; j < num_node; ++j)
            {
                auto local_grid_c_phi = grid.slice(0,j,j+1).squeeze(0).permute({1,0}).contiguous();

                // auto multi_           = torch::matmul(local_grid, local_grid_c_phi);

                // multi_                = torch::softmax(multi_, -1);

                // multi_                = torch::matmul(multi_, local_grid_c_phi.permute({1,0})).contiguous();


                auto multi_                = torch::matmul(torch::softmax(torch::matmul(local_grid, local_grid_c_phi), -1), local_grid_c_phi.permute({1,0})).contiguous();

                torch::Tensor loss    = (multi_ - local_grid).abs().mean();

                if(total_loss.defined())
                {
                    total_loss += loss;
                }
                else
                {
                    total_loss = loss;
                }
            }

        }
        return total_loss * num_channels;
    }

    torch::Tensor zero_tensor = torch::zeros({1}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA) );
};
