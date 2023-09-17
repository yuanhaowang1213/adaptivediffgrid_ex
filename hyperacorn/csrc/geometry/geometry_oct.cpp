#include "geometry_oct.h"

#include "modules/IndirectGridSample3D.h"
#define epsilon 0.0001
GeometryOCT::GeometryOCT(int num_channels, int D, HyperTreeBase tree, std::shared_ptr<CombinedParams> params)
    : HierarchicalNeuralGeometry(num_channels, D, params, tree)
{
    std::cout << "Type: ExEx" << std::endl;
    std::vector<long> features_grid_shape;
    for (int i = 0; i < D; ++i)
    {
        features_grid_shape.push_back(params->net_params.grid_size);
    }



    if(params->train_params.use_NLM  && params->train_params.loss_nlm)
    {
        printf("use nlm loss here\n");
        auto x_coord = torch::linspace(-1,1,params->net_params.NLM_grid_size, torch::TensorOptions().device(torch::kCUDA));
        auto y_coord = x_coord;
        auto z_coord = z_coord;
        auto coord = torch::meshgrid({x_coord, y_coord, z_coord});
        auto coord_onecell = torch::cat({coord[0].unsqueeze(-1),coord[1].unsqueeze(-1),coord[2].unsqueeze(-1)},-1);
        auto nl_grid_tensor = coord_onecell.unsqueeze(0).repeat({params->octree_params.tree_optimizer_params.max_active_nodes_initial,1,1,1,1});
        nl_grid_tensor = nl_grid_tensor.reshape({params->octree_params.tree_optimizer_params.max_active_nodes_initial,params->net_params.NLM_grid_size * params->net_params.NLM_grid_size*params->net_params.NLM_grid_size , 3});
        PrintTensorInfo(nl_grid_tensor);
        register_parameter("nl_grid_tensor", nl_grid_tensor);
    }


    if(params->train_params.loss_tv > 0)
    {
        std::vector<long> sizes;
        // sizes.push_back(params->net_params.grid_features);
        sizes.push_back(params->octree_params.tree_optimizer_params.max_active_nodes_initial);
        for(int i = 0; i < D; ++i)
        sizes.push_back(params->net_params.grid_size);
        sizes.push_back(D);
        grid_tensor  = torch::empty(sizes);
        grid_tensor.uniform_(-1,1);
        // grid_tensor = grid_tensor.reshape({params->octree_params.tree_optimizer_params.max_active_nodes_initial,params->net_params.grid_size * params->net_params.grid_size*params->net_params.grid_size , 3});
        // PrintTensorInfo(grid_tensor);
        register_parameter("grid_tensor", grid_tensor);
    }

    // register_module("decoding", decoding);

    grid_sampler = NeuralGridSampler(false, params->net_params.sampler_align_corners);
    register_module("grid_sampler", grid_sampler);

    {
        int features_after_pe;
        if (params->net_params.decoder_skip)
        {
            features_after_pe = num_channels;
        }
        // only keep active nodes in memory
        explicit_grid_generator =
            ExplicitFeatureGrid(params->octree_params.tree_optimizer_params.max_active_nodes_initial,
                                features_after_pe, features_grid_shape, params->train_params.grid_init);

    }

    register_module("explicit_grid_generator", explicit_grid_generator);

    std::cout << "Feature Grid: " << TensorInfo(explicit_grid_generator->grid_data) << std::endl;
    std::cout << "Numel: " << explicit_grid_generator->grid_data.numel()
              << " Memory: " << explicit_grid_generator->grid_data.numel() * sizeof(float) / 1000000.0 << " MB"
              << std::endl;

    std::cout << "=== ============= ===" << ConsoleColor::RESET << std::endl;

}

void GeometryOCT::AddParametersToOptimizer()
{
    HierarchicalNeuralGeometry::AddParametersToOptimizer();

    if (params->train_params.optimize_volume)
    {
        if (params->train_params.exex_op == "adam")
        {
            std::cout << "Optimizing Explicit Grid with (ADAM) LR " << params->train_params.lr_exex_grid_adam
                      << std::endl;
            optimizer_adam->add_param_group(
                {explicit_grid_generator->parameters(),
                 std::make_unique<torch::optim::AdamOptions>(params->train_params.lr_exex_grid_adam)});
        }
        else if (params->train_params.exex_op == "rms")
        {
            std::cout << "Optimizing Explicit Grid with (RMS) LR " << params->train_params.lr_exex_grid_rms
                      << std::endl;
            optimizer_rms->add_param_group(
                {explicit_grid_generator->parameters(),
                 std::make_unique<torch::optim::RMSpropOptions>(params->train_params.lr_exex_grid_rms)});
        }
        else
        {
            CHECK(false);
        }

    }
}


torch::Tensor GeometryOCT::SampleVolume(torch::Tensor global_coordinate, torch::Tensor node_id)
{
    if (global_coordinate.numel() == 0)
    {
        // No samples -> just return and empty tensor
        return global_coordinate;
    }
                // [98,1024,3]

    torch::Tensor grid, local_samples, neural_features;

    {
        SAIGA_OPTIONAL_TIME_MEASURE("explicit_grid_generator", timer);
        // [num_nodes, num_features, 11, 11, 11]
        auto local_node_id = tree->GlobalNodeIdToLocalActiveId(node_id);
        grid               = explicit_grid_generator->forward(local_node_id);            
    }

    {
        SAIGA_OPTIONAL_TIME_MEASURE("ComputeLocalSamples", timer);
        local_samples = tree->ComputeLocalSamples(global_coordinate, node_id);
        // [98,1024,3]

    }
    CHECK_EQ(local_samples.requires_grad(), global_coordinate.requires_grad());
    {
        SAIGA_OPTIONAL_TIME_MEASURE("grid_sampler->forward", timer);
        // [num_groups, group_size, num_features]
        neural_features = grid_sampler->forward(grid, local_samples);
    }
    return neural_features;
}


torch::Tensor GeometryOCT::VolumeRegularizer()
{

    // [num_nodes, num_channels, 11, 11, 11]
    auto active_node_id = tree->ActiveNodeTensor().to(torch::kCPU);
    std::vector<long> active_node_inroi;
    {
        auto node_min = tree->node_position_min.to(torch::kCPU);
        auto node_max = tree->node_position_max.to(torch::kCPU);
        auto node_mid = node_min + node_max;
        float* node_min_ptr = node_min.data_ptr<float>();
        float* node_max_ptr = node_max.data_ptr<float>();
        float* node_mid_ptr = node_mid.data_ptr<float>();
        auto node_active_bool = tree->node_active.to(torch::kCPU);


        for(int node_id = 0; node_id < tree->NumNodes();++node_id)
        {
            if(in_roi(node_min_ptr, node_max_ptr, node_mid_ptr, params->octree_params.tree_optimizer_params.tree_roi_min, params->octree_params.tree_optimizer_params.tree_roi_max, node_id))

            {
                if(node_active_bool.data_ptr<int>()[node_id] == 1)
                active_node_inroi.push_back(node_id);
            }
        }
    }
    torch::Tensor active_node = torch::from_blob(&active_node_inroi[0], {(long)active_node_inroi.size()},torch::TensorOptions().dtype(torch::kLong))
                                       .clone()
                                       .to(torch::kCUDA);


    std::vector<long> in_roi_node_vec;
    std::vector<long> in_roi_index_vec;
    {
        auto node_min = torch::index_select(tree->node_position_min, 0, tree->active_node_ids).to(torch::kCPU);
        auto node_max = torch::index_select(tree->node_position_max, 0, tree->active_node_ids).to(torch::kCPU);
        auto node_mid = node_min + node_max;

        auto active_node_id = tree->active_node_ids.to(torch::kCPU);

        float* node_min_ptr = node_min.data_ptr<float>();
        float* node_max_ptr = node_max.data_ptr<float>();
        float* node_mid_ptr = node_mid.data_ptr<float>();
        torch::Tensor in_roi_index = torch::zeros({tree->NumActiveNodes(),1},  torch::TensorOptions(torch::kFloat32));
        
        int in_roi_node_num = 0;


        for(int i = 0; i < tree->NumActiveNodes();++i)
        {
            if(in_roi(node_min_ptr, node_max_ptr, node_mid_ptr, params->octree_params.tree_optimizer_params.tree_roi_min,
                     params->octree_params.tree_optimizer_params.tree_roi_max, i))
            {

                in_roi_index.data_ptr<float>()[i] = 10;
                in_roi_node_num += 1;
                in_roi_node_vec.push_back(i);
            }
        }

        for(int i = 0; i < in_roi_node_num; ++i)
        {
            in_roi_index_vec.push_back(i);
        }
    }
    auto in_roi_node = torch::from_blob(&in_roi_node_vec[0], {(long)in_roi_node_vec.size()}, torch::TensorOptions().dtype(torch::kLong)).clone().to(torch::kCUDA);


    torch::Tensor grid;
    {

        {
            grid = explicit_grid_generator(tree->GlobalNodeIdToLocalActiveId(tree->ActiveNodeTensor()));
        }
    }



    torch::Tensor tv_loss, edge_loss, zero_loss;


    if (params->train_params.loss_tv > 0)
    {
        torch::Tensor tv_grid;
        if (params->train_params.tv_loss_in_feature_space)
        {
            if (params->train_params.use_tv_roi )
            {
                tv_grid = torch::index_select(grid, 0, in_roi_node);
            }
            else
            {
                tv_grid = grid;
            }
            // 
        }
        else
        {
            // [512, 11, 11,11,3]
                
            // [512, 11, 11, 11, num_channels]
            auto grid_channel_last = grid.permute({0, 2, 3, 4, 1});

            // [512, 11, 11, 11, 1]
            auto decoded_grid = DecodeFeatures(grid_channel_last);



            // [512, 1, 11, 11, 11]
            decoded_grid = decoded_grid.permute({0, 4, 1, 2, 3});
            tv_grid      = decoded_grid;
        }

        torch::Tensor factor;
        TVLoss tv;
        tv_loss = tv.forward(tv_grid, factor) * params->train_params.loss_tv;
        // tv_loss = tv.forward(tv_grid, local_tv_loss);
    }
    // printf("test zero loss \n");


    // printf("test edge loss \n");
    NonLocallLoss nl_loss;
    torch::Tensor nlm_loss;
    if(params->train_params.loss_nlm && params->train_params.use_NLM)
    {

        std::random_shuffle(in_roi_index_vec.begin(), in_roi_index_vec.end());
        torch::Tensor grid_nl;
        if(params->train_params.update_grid_each_iter)
        {
            std::vector<long> sizes;
            sizes.push_back(tree->active_node_ids.size(0));
            int D = 3;
            for(int i = 0; i < D; ++i)
            sizes.push_back(params->net_params.NLM_grid_size);
            sizes.push_back(D);
            grid_nl = torch::empty(sizes);
            grid_nl.uniform_(-1 + epsilon,1 - epsilon);
            grid_nl = grid_nl.reshape({tree->active_node_ids.size(0),
                    params->net_params.NLM_grid_size * params->net_params.NLM_grid_size*params->net_params.NLM_grid_size , 3}).to(torch::kCUDA);
            
        }
        else
        {
            // grid_nl = nl_grid_tensor.slice(0,0, tree->active_node_ids.size(0)).to(torch::kCUDA);
            grid_nl = nl_grid_tensor.slice(0, 0, in_roi_node.size(0)).to(torch::kCUDA);
        }
        // [num_nodes, 11,11,11,features]
        // torch::Tensor grid_nlm = grid_tensor.slice(0,0, active_node_id.size(0)).to(torch::kCUDA);
        auto grid_nlm = torch::index_select(grid, 0, in_roi_node);
        auto neural_features = grid_sampler->forward( grid_nlm, grid_nl);

        NonLocallLoss nl_loss;
        if(params->train_params.use_cross_nlm)
        {
            if(params->train_params.use_weighted_nlm)
            {
                nlm_loss = nl_loss.forward_cross(neural_features, in_roi_index_vec, local_nlm_loss) * params->train_params.loss_nlm;
            }
            else
            {
                nlm_loss = nl_loss.forward_cross(neural_features, in_roi_index_vec) * params->train_params.loss_nlm;
            }
            
        }
        // PrintTensorInfo(in_roi_index);
        else
        {
            // [512, grid*grid*grid, feature] neural features
            if(params->train_params.use_weighted_nlm)
            {
                nlm_loss = nl_loss.forward(neural_features, local_nlm_loss) *  params->train_params.loss_nlm;
            }
            else
            {
                nlm_loss = nl_loss.forward(neural_features ) * params->train_params.loss_nlm;
            }
        }
        

    }

    if (params->train_params.loss_edge > 0)
    {
        Eigen::Vector<int, -1> shape_v;
        shape_v.resize(D);
        for (int i = 0; i < D; ++i)
        {
            shape_v(i) = params->net_params.grid_size;
        }
        SampleList neighbor_samples = tree->NodeNeighborSamples(shape_v, 0.001, params->octree_params.tree_optimizer_params.tree_edge_roi_min, params->octree_params.tree_optimizer_params.tree_edge_roi_max,params->octree_params.tree_optimizer_params.use_tree_roi);

        int num_rays                = neighbor_samples.size();

        torch::Tensor neural_features;


        auto local_samples = tree->ComputeLocalSamples(neighbor_samples.global_coordinate, neighbor_samples.node_id);
        {
            auto local_node_id = tree->GlobalNodeIdToLocalActiveId(neighbor_samples.node_id);
            //        grid               = explicit_grid_generator->forward(local_node_id);

            neural_features = IndirectGridSample3D(explicit_grid_generator->grid_data, local_node_id, local_samples);
        }
        auto ray_index = neighbor_samples.ray_index;



        // [num_rays, channels]
        auto per_ray_sum = torch::zeros({num_rays, neural_features.size(1)}, neural_features.options());
        //        PrintTensorInfo(per_ray_sum);

        per_ray_sum.index_add_(0, ray_index, neural_features);

        auto t1 = per_ray_sum.slice(0, 0, per_ray_sum.size(0), 2);
        auto t2 = per_ray_sum.slice(0, 1, per_ray_sum.size(0), 2);

        // old code
        // // [num_rays]
        auto edge_error = (t1 - t2).abs().mean(1);



        edge_loss = edge_error.mean() * params->train_params.loss_edge * neural_features.size(1);

    }



    torch::Tensor loss;
    if (edge_loss.defined())
    {
        if (loss.defined())
            loss += edge_loss;
        else
            loss = edge_loss;
    }
    if (tv_loss.defined())
    {
        if (loss.defined())
            loss += tv_loss;
        else
            loss = tv_loss;
    }
    if (zero_loss.defined())
    {
        if (loss.defined())
            loss += zero_loss;
        else
            loss = zero_loss;
    }
    if(nlm_loss.defined())
    {
        if(loss.defined())
        {
            loss += nlm_loss;
        }
        else
        {
            loss = nlm_loss;
        }
    }
    // exit(0);
    return loss;
}
torch::Tensor GeometryOCT::SampleVolumeIndirect(torch::Tensor global_coordinate, torch::Tensor node_id)
{
    torch::Tensor local_samples, neural_features, density;
    // printf("test here\n");
    {
        SAIGA_OPTIONAL_TIME_MEASURE("ComputeLocalSamples", timer);
        local_samples = tree->ComputeLocalSamples(global_coordinate, node_id);
        // [6381180, 3]
    }
    {
        SAIGA_OPTIONAL_TIME_MEASURE("IndirectGridSample3D", timer);
        auto local_node_id = tree->GlobalNodeIdToLocalActiveId(node_id);
        neural_features    = IndirectGridSample3D(explicit_grid_generator->grid_data, local_node_id, local_samples);
    }

    // [1191918,6]
    {
        SAIGA_OPTIONAL_TIME_MEASURE("DecodeFeatures", timer);
        density = DecodeFeatures(neural_features);
        // PrintTensorInfo(density);
    }

    return density;
}

void GeometryOCT::InterpolateInactiveNodes(HyperTreeBase old_tree)
{
    //     PrintTensorInfo(old_tree->active_node_ids);
    //     PrintTensorInfo(tree->active_node_ids);
    // if (params->net_params.save_memory) return;

    torch::NoGradGuard ngg;
    // std::cout << " ==== interpolating inactive nodes..." << std::endl;
    ScopedTimerPrintLine tim("InterpolateInactiveNodes");
    // [num_inactive_nodes]
    auto new_active_nodes = tree->active_node_ids;

    CUDA_SYNC_CHECK_ERROR();
    //    PrintTensorInfo(inactive_nodes);

    // [num_inactive_nodes, 11, 11, 11, channels]
    auto global_coordinate = tree->UniformGlobalSamples(new_active_nodes, params->net_params.grid_size);
    //    PrintTensorInfo(global_coordinate);

    // [num_inactive_nodes, block_size, channels]
    auto global_coordinate_block =
        global_coordinate.reshape({global_coordinate.size(0), -1, global_coordinate.size(-1)});
    global_coordinate_block = global_coordinate_block.reshape({-1, 3});
    //    PrintTensorInfo(global_coordinate_block);
    CUDA_SYNC_CHECK_ERROR();
    // auto node_id = inactive_nodes.reshape({-1});

    auto [node_id, mask] = old_tree->NodeIdForPositionGPU(global_coordinate_block);
    //    PrintTensorInfo(node_id);
    //    PrintTensorInfo(mask);


    auto local_samples = old_tree->ComputeLocalSamples(global_coordinate_block, node_id);
    //    PrintTensorInfo(local_samples);

    CUDA_SYNC_CHECK_ERROR();

    //    torch::Tensor grid;
    torch::Tensor neural_features;
    // if (params->net_params.save_memory)
    {
        auto local_node_id = old_tree->GlobalNodeIdToLocalActiveId(node_id);
        //        grid               = explicit_grid_generator->forward(local_node_id);

        neural_features = IndirectGridSample3D(explicit_grid_generator->grid_data, local_node_id, local_samples);
    }

    CUDA_SYNC_CHECK_ERROR();
    //    PrintTensorInfo(neural_features);

    // [num_inactive_nodes, 11, 11, 11, channels]
    neural_features = neural_features.reshape({global_coordinate.size(0), global_coordinate.size(1),
                                               global_coordinate.size(2), global_coordinate.size(3), -1});
    //    PrintTensorInfo(neural_features);

    // [num_inactive_nodes, channels, 11, 11, 11]
    neural_features = neural_features.permute({0, 4, 1, 2, 3});
    //    PrintTensorInfo(neural_features);

    //    PrintTensorInfo( explicit_grid_generator->grid_data);

    auto new_local_node_id = tree->GlobalNodeIdToLocalActiveId(new_active_nodes);
    explicit_grid_generator->grid_data.index_copy_(0, new_local_node_id, neural_features);
    //    PrintTensorInfo( explicit_grid_generator->grid_data);

    CUDA_SYNC_CHECK_ERROR();
}
