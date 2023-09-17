#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include "ImplicitNet.h"
#include "Settings.h"
#include "data/SceneBase.h"
#include "geometry.h"
// #include "fourierlayer.h"
#include "utils/cimg_wrapper.h"
// #include "multiplefilterlayer.h"
#include "non_local.h"
// #include "PositionalEncoding.h"
class GeometryOCT : public HierarchicalNeuralGeometry
{
   public:
    GeometryOCT(int num_channels, int D, HyperTreeBase tree, std::shared_ptr<CombinedParams> params);

    bool in_roi(float * node_min_ptr, float * node_max_ptr, float * node_mid_ptr, vec3 roi_min, vec3 roi_max, int node_id)
    {
        if( ((node_min_ptr[node_id * 3] > roi_min[0] && node_min_ptr[node_id * 3] < roi_max[0] )||
            (node_max_ptr[node_id * 3] > roi_min[0] && node_max_ptr[node_id * 3] < roi_max[0] )||
            (node_mid_ptr[node_id * 3] > roi_min[0] && node_mid_ptr[node_id * 3] < roi_max[0] )) &&
            ((node_min_ptr[node_id * 3 + 1] > roi_min[1] && node_min_ptr[node_id * 3 + 1] < roi_max[1]) ||
            (node_max_ptr[node_id * 3 + 1] > roi_min[1] && node_max_ptr[node_id * 3 + 1] < roi_max[1]) ||
            (node_mid_ptr[node_id * 3 + 1] > roi_min[1] && node_mid_ptr[node_id * 3 + 1] < roi_max[1])) &&
            ((node_min_ptr[node_id * 3 + 2] > roi_min[2] && node_min_ptr[node_id * 3 + 2] < roi_max[2]) ||
            (node_max_ptr[node_id * 3 + 2] > roi_min[2] && node_max_ptr[node_id * 3 + 2] < roi_max[2]) ||
            (node_mid_ptr[node_id * 3 + 2] > roi_min[2] && node_mid_ptr[node_id * 3 + 2] < roi_max[2]))  )
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    torch::Tensor VolumeRegularizer();


    virtual void InterpolateInactiveNodes(HyperTreeBase old_tree);



   protected:
    void AddParametersToOptimizer() override;

    torch::Tensor SampleVolume(torch::Tensor global_coordinate, torch::Tensor node_id) override;


    virtual torch::Tensor SampleVolumeIndirect(torch::Tensor global_coordinate, torch::Tensor node_id) override;


    ExplicitFeatureGrid explicit_grid_generator = nullptr;
    NeuralGridSampler grid_sampler              = nullptr;

    torch::Tensor nl_grid_tensor;
    private:
        torch::Tensor global_coord_roi;
        Eigen::Vector<int, -1> shape_v;
        torch::Tensor grid_tensor;


};
