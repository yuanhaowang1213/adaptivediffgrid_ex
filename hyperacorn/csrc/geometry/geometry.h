#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/assert.h"
#include "saiga/cuda/imgui_cuda.h"

#include "ImplicitNet.h"
#include "../data/SceneBase.h"

// #include "fourierlayer.h"
// #include "dncnn.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

// #include "randomfourierfeature.h"



class TVLoss
{
   public:
    // TV Loss for N-dimensional input.
    //
    // Input
    //      grid [num_batches, num_channels, x, y, z, ...]
    //      weight [num_batches]
    torch::Tensor forward(torch::Tensor grid, torch::Tensor weight = {})
    {
        int num_channels = grid.size(1);
        int D            = grid.dim() - 2;
        // int num_batches  = grid.size(0);
        // int num_channels = grid.size(1);


        torch::Tensor total_loss;
        for (int i = 0; i < D; ++i)
        {
            int d     = i + 2;
            int size  = grid.size(d);
            auto loss = (grid.slice(d, 0, size - 1) - grid.slice(d, 1, size)).abs().mean({1, 2, 3, 4});

            if (weight.defined())
            {
                loss *= weight;
            }

            loss = loss.mean();

            if (total_loss.defined())
            {
                total_loss += loss;
            }
            else
            {
                total_loss = loss;
            }
        }
        CHECK(total_loss.defined());

        return total_loss * num_channels;
    }
};


class TVLoss_global
{
    public:
    // Input grid[x,y,z]
    torch::Tensor forward(torch::Tensor grid)
    {
        torch::Tensor total_loss;
        int D = grid.dim();
        for(int i = 0; i < D;++i)
        {
            int size = grid.size(i);
            auto loss = (grid.slice(i, 0, size-1) - grid.slice(i, 1, size)).abs().mean();
            if(total_loss.defined())
            {
                total_loss += loss;
            }
            else
            {
                total_loss = loss;
            }
        }
        CHECK(total_loss.defined());
        return total_loss;

    }
};




class NeuralGeometry : public torch::nn::Module
{
   public:
    NeuralGeometry(int num_channels, int D, std::shared_ptr<CombinedParams> params)
        : num_channels(num_channels), D(D), params(params)
    {
    }

    virtual void train(int epoch_id, bool on)
    {
        torch::nn::Module::train(on);
        c10::cuda::CUDACachingAllocator::emptyCache();
        if (on)
        {
            if (!optimizer_adam && !optimizer_sgd)
            {
                CreateGeometryOptimizer();
            }
            if (optimizer_adam) optimizer_adam->zero_grad();
            if (optimizer_sgd) optimizer_sgd->zero_grad();
            if (optimizer_rms) optimizer_rms->zero_grad();
            if (optimizer_decoder) optimizer_decoder->zero_grad();
        }
    }

    void ResetGeometryOptimizer() { CreateGeometryOptimizer(); }

    void CreateGeometryOptimizer()
    {
        optimizer_adam =
            std::make_shared<torch::optim::Adam>(std::vector<torch::Tensor>(), torch::optim::AdamOptions().lr(10));

        optimizer_rms = std::make_shared<torch::optim::RMSprop>(std::vector<torch::Tensor>(),
                                                                torch::optim::RMSpropOptions().lr(10));

        optimizer_sgd = std::make_shared<torch::optim::SGD>(std::vector<torch::Tensor>(), torch::optim::SGDOptions(10));
        AddParametersToOptimizer();
    }



    virtual void PrintInfo() {}
    virtual void PrintGradInfo(int epoch_id, TensorBoardLogger* logger) {}


    void OptimizerStep(int epoch_id)
    {
        if (optimizer_sgd)
        {
            optimizer_sgd->step();
            optimizer_sgd->zero_grad();
        }
        if (optimizer_adam)
        {
            optimizer_adam->step();
            optimizer_adam->zero_grad();
        }
        if (optimizer_rms)
        {
            optimizer_rms->step();
            optimizer_rms->zero_grad();
        }
        if (optimizer_decoder)
        {
            optimizer_decoder->step();
            optimizer_decoder->zero_grad();
        }
    }

    void UpdateLearningRate(double factor)
    {
        if (optimizer_adam) UpdateLR(optimizer_adam.get(), factor);
        if (optimizer_sgd) UpdateLR(optimizer_sgd.get(), factor);
        if (optimizer_rms) UpdateLR(optimizer_rms.get(), factor);
        if (optimizer_decoder) UpdateLR(optimizer_decoder.get(), factor);
    }


    // Compute the 'simple' integral by just adding each sample value to the given ray index.
    // The ordering of the samples is not considered.
    //
    // Computes:
    //      sample_integral[ray_index[i]] += sample_value[i]
    //
    // Input:
    //      sample_value [num_groups, group_size, num_channels]
    //      ray_index [N]
    //
    // Output:
    //      sample_integral [num_channels, num_rays]
    //
    torch::Tensor IntegrateSamplesXRay(torch::Tensor sample_values, torch::Tensor integration_weight,
                                       torch::Tensor ray_index, int num_channels, int num_rays);


    // Blends the samples front-to-back using alpha blending. This is used for a RGB-camera model (non xray) and the
    // implementation follows the raw2outputs function of NeRF. However, in our case it is more complicated because
    // each ray can have a different number of samples. The computation is done in the following steps:
    //
    //  1. Sort the sample_values into a matrix of shape: [num_rays, max_samples_per_ray, num_channels]
    //     Each row, is also ordered correctly in a front to back fashion. If a ray has less than max_samples_per_ray
    //     samples, the remaining elements are filled with zero.
    //
    //
    // Input:
    //      sample_value [any_shape, num_channels]
    //      ray_index [any_shape]
    //
    //      // The local id of each sample in the ray. This is used for sorting!
    //      sample_index_in_ray [any_shape]
    //
    // Output:
    //      sample_integral [num_channels, num_rays]
    //
    torch::Tensor IntegrateSamplesAlphaBlending(torch::Tensor sample_values, torch::Tensor integration_weight,
                                                torch::Tensor ray_index, torch::Tensor sample_index_in_ray,
                                                int num_channels, int num_rays, int max_samples_per_ray);



   protected:
    std::shared_ptr<torch::optim::Adam> optimizer_decoder;

    std::shared_ptr<torch::optim::Adam> optimizer_adam;
    std::shared_ptr<torch::optim::SGD> optimizer_sgd;
    std::shared_ptr<torch::optim::RMSprop> optimizer_rms;

    int num_channels;
    int D;
    std::shared_ptr<CombinedParams> params;

    virtual void AddParametersToOptimizer() = 0;
};

class HierarchicalNeuralGeometry : public NeuralGeometry
{
   public:
    HierarchicalNeuralGeometry(int num_channels, int D, std::shared_ptr<CombinedParams> params, HyperTreeBase tree);

    std::pair<torch::Tensor, torch::Tensor> AccumulateSampleLossPerNode(const NodeBatchedSamples& combined_samples,
                                                                        torch::Tensor per_ray_loss);
    std::pair<torch::Tensor, torch::Tensor> AccumulateSampleLossPerNode(const SampleList& combined_samples,
                                                                        torch::Tensor per_ray_loss);

    virtual torch::Tensor VolumeRegularizer() { return torch::Tensor(); }

    virtual void SampleVolumeTest(std::string output_vol_file) {}

    HyperTreeBase tree = nullptr;

    torch::Tensor ComputeImage(SampleList all_samples, int num_channels, int num_pixels, bool use_decoder );

    // Input:
    //   global_coordinate [num_samples, D]
    //   node_id [num_samples]
    virtual torch::Tensor SampleVolumeIndirect(torch::Tensor global_coordinate, torch::Tensor node_id) { return {}; }

    // Output:
    //      value [num_groups, group_size, num_channels]
    torch::Tensor SampleVolumeBatched(torch::Tensor global_coordinate, torch::Tensor sample_mask, torch::Tensor node_id, bool use_decoder );


    virtual void to(torch::Device device, bool non_blocking = false) override
    {
        NeuralGeometry::to(device, non_blocking);
    }

    // Returns [volume_density, volume_node_index, volume_valid]
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> UniformSampledVolume(std::vector<long> shape,
                                                                                 int num_channels ,vec3 roi_min, vec3 roi_max, bool is_final );

    // slice_dim 0,1,2 for z,y,x
    void SaveVolume(TensorBoardLogger* tblogger, std::string tb_name, std::string out_dir, int num_channels,
                    float intensity_scale, int size, int slice_dim, vec3 roi_min, vec3 roi_max );

    // FINN decoder                              = nullptr;

    // if(0)
    // {
    //     // static MultiscaleBacon shared_fourier_layer;
    //     MultiscaleBacon fourier_layer          = nullptr;
    // }
    // Evaluates the octree at the inactive-node's feature positions and sets the respective feature vectors.
    // This should be called before changing the octree structure, because then some inactive nodes will become active.
    // The newly active nodes will have a good initialization after this method.
    virtual void InterpolateInactiveNodes(HyperTreeBase old_tree) {}

    void setup_tv(torch::Tensor tv_losshere) {local_tv_loss = tv_losshere;};

    void setup_nlm(torch::Tensor nlm_loss) {local_nlm_loss = nlm_loss;};

    torch::Tensor local_nlm_loss;
    torch::Tensor local_tv_loss;

   protected:
    // Takes the sample locations (and the corresponding tree-node-ids) and retrieves the values from the
    // hierarchical data structure. The input samples must be 'grouped' by the corresponding node-id.
    // The per-sample weight is multiplied to the raw sample output.
    //
    // Input:
    //      global_coordinate [num_groups, group_size, 3]
    //      weight            [num_groups, group_size]
    //      node_id           [num_groups]
    //
    // Output:
    //      value [num_groups, group_size, num_features]
    virtual torch::Tensor SampleVolume(torch::Tensor global_coordinate, torch::Tensor node_id) = 0;



    virtual void AddParametersToOptimizer();
    virtual torch::Tensor DecodeFeatures(torch::Tensor neural_features);

    // Input:
    //      neural_features [num_groups, group_size, num_channels]
    // if(0)
    // {
    //     torch::Tensor FourierProcess(torch::Tensor position, torch::Tensor neural_features);
    // }

   public:
    Saiga::CUDA::CudaTimerSystem* timer = nullptr;
};
