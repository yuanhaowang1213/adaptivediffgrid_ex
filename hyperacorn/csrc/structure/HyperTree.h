#pragma once

#include "HelperStructs.h"

#include <saiga/core/geometry/aabb.h>
using namespace torch::indexing;

#if 0

// Dimension Quadtree==2, Octree==3,...
template <int D, typename Payload = void>
struct HyperNode
{
    static constexpr int NS = (1 << D);

    using Node = HyperNode<D, Payload>;
    using Vec  = Eigen::Vector<float, D>;
    using iVec = Eigen::Vector<int, D>;

    HyperNode()
    {
        for (auto& c : children)
        {
            c = -1;
        }
    }

    HyperNode(int parent_id, int depth, Vec position, bool test, Vec position_max)
        : parent_id(parent_id), depth(depth), position(position), position_max(position_max)
    {
        for (auto& c : children)
        {
            c = -1;
        }
    }

    std::array<Node, NS> Split()
    {
        std::array<Node, NS> result;

        Vec center = Center();

        for (int i = 0; i < NS; ++i)
        {
            Vec new_min = position;
            Vec new_max = position_max;

            for (int d = 0; d < D; ++d)
            {
                if ((i >> d & 1) == 0)
                {
                    new_min[d] = position[d];
                    new_max[d] = center[d];
                }
                else
                {
                    new_min[d] = center[d];
                    new_max[d] = position_max[d];
                }
            }

            result[i] = Node(id, depth + 1, new_min, false, new_max);
        }

        return result;
    }



    // True if the global sample is inside this node
    bool Contains(Vec p, bool verbose = false)
    {
        Vec min_p = position;
        Vec max_p = position_max;

        for (int d = 0; d < D; ++d)
        {
            if (min_p(d) > p(d) || max_p(d) < p(d))
            {
                if (verbose)
                {
                    std::cout << "found axis " << d << std::endl;
                    std::cout << min_p(d) << " > " << p(d) << " = " << (min_p(d) > p(d)) << std::endl;
                    std::cout << max_p(d) << " < " << p(d) << " = " << (max_p(d) < p(d)) << std::endl;
                }
                return false;
            }
        }
        return true;

        bool b1 = !((min_p.array() > p.array()).any() || (max_p.array() < p.array()).any());
        bool b2 = !((min_p.array() > p.array()) || (max_p.array() < p.array())).any();

        CHECK_EQ(b1, b2);
        return b2;
    }
    std::tuple<bool, float, float> Intersect2(Vec origin, Vec direction)
    {
        double t_near = -236436436;  // maximums defined in float.h
        double t_far  = 43637575;

        for (int i = 0; i < D; i++)
        {  // we test slabs in every direction
            if (direction[i] == 0)
            {  // ray parallel to planes in this direction
                if ((origin[i] < position[i]) || (origin[i] >= position_max[i]))
                {
                    return {false, 0, 0};  // parallel AND outside box : no intersection possible
                }
            }
            else
            {  // ray not parallel to planes in this direction
                float T_1 = (position[i] - origin[i]) / direction[i];
                float T_2 = (position_max[i] - origin[i]) / direction[i];

                if (T_1 > T_2)
                {  // we want T_1 to hold values for intersection with near plane
                    std::swap(T_1, T_2);
                }
                if (T_1 > t_near)
                {
                    t_near = T_1;
                }
                if (T_2 < t_far)
                {
                    t_far = T_2;
                }
                if ((t_near > t_far) || (t_far < 0))
                {
                    return {false, 0, 0};
                }
            }
        }
        return {true, t_near, t_far};
    }

    std::tuple<bool, float, float> Intersect(Vec origin, Vec direction)
    {
        // convert -0 to +0
        direction = (direction + Vec::Ones()) - Vec::Ones();

        Vec inv_dir = 1.0 / direction.array();

        // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
        // r.org is origin of ray
        Vec t_min = (position - origin).array() * inv_dir.array();
        Vec t_max = (position_max - origin).array() * inv_dir.array();

        float tmin = t_min.array().min(t_max.array()).maxCoeff();
        float tmax = t_min.array().max(t_max.array()).minCoeff();

        // if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behing us
        if (tmax < 0)
        {
            return {false, 0, 0};
        }

        // if tmin > tmax, ray doesn't intersect AABB
        if (tmin > tmax)
        {
            return {false, 0, 0};
        }

        // Intersects-> return both hitpoints
        return {true, tmin, tmax};
    }


    static iVec Delinearize(int i, iVec size)
    {
        iVec res;

        //        for (int d = 0; d < D; ++d)
        //        {
        //            res(d) = i % size(d);
        //            i /= size(d);
        //        }
        for (int d = D - 1; d >= 0; --d)
        {
            res(d) = i % size(d);
            i /= size(d);
        }
        return res;
    }

    // Volume relative to the unit cube [-1,1]
    float NormalizedVolume()
    {
        float root_volume = (Vec::Ones() * 2).array().prod();
        return Size().array().prod() / root_volume;
    }

    // Side length of this cube
    float SideLength() { return position_max(0) - position(0); }
    float DiagonalLength() { return sqrt(D) * SideLength(); }


    // converts a local coordinate to the global global coordinate system
    // [-1, 1] -> [position, position+size]
    Vec Local2Global(Vec c)
    {
        // [0, 1]
        c = (c + Vec::Ones()) * 0.5;
        // [0, size]
        c = c.array() * Size().array();
        // [position, position +size]
        c = (c + position);
        return c;
    }

    // converts a local coordinate to the global global coordinate system
    // [position, position+size] -> [-1, 1]
    Vec Global2Local(Vec c)
    {
        // [0, size]
        c = c - position;
        // [0, 1]
        c = c.array() / Size().array();
        // [-1, 1]
        c = (c * 2) - Vec::Ones();
        return c;
    }

    void AddSampleLoss(double loss_sum, double weight)
    {
        err_sum += loss_sum;
        err_count += weight;
    }

    void ResetLoss()
    {
        err_sum   = 0;
        err_count = 0;
    }

    bool HasLoss() { return std::isfinite(error); }

    void FinalizeLoss(bool divide_by_count)
    {
        if (divide_by_count)
        {
            if (err_count > 0)
            {
                error = err_sum / err_count;
            }
            else
            {
                error = 0;
            }
        }
        else
        {
            error = err_sum;
        }
    }

    void FinalizeExp(double alpha)
    {
        if (HasLoss())
        {
            error = (1 - alpha) * error + err_sum * alpha;
        }
        else
        {
            error = err_sum;
        }
    }

    // Basic Structural State
    bool active = false;
    int id      = 0;

    // if this node is active it will have a 'local' id in the range [0,#active_nodes-1]
    int active_id = -1;

    int parent_id = -1;
    int depth     = 0;
    std::array<int, NS> children;

    // Relative to the unit cube [-1, 1]
    // Position is min-corner of the node
    Vec position;
    Vec position_max;

    Vec Size() { return position_max - position; }
    Vec Center() { return (position + position_max) * 0.5; }


    float err_sum    = 0;
    double err_count = 0;

    float error = std::numeric_limits<float>::infinity();

    float last_updated = 0;
};

template <int D, typename Payload>
inline std::ostream& operator<<(std::ostream& strm, const HyperNode<D, Payload>& node)
{
    strm << "[Node] " << node.position.transpose() << " | " << node.position_max.transpose();
    return strm;
}
#endif

inline Eigen::Vector<int, -1> Delinearize(int i, Eigen::Vector<int, -1> size, bool swap_xy)
{
    int D = size.rows();
    Eigen::Vector<int, -1> res;
    res.resize(D);

    //        for (int d = 0; d < D; ++d)
    //        {
    //            res(d) = i % size(d);
    //            i /= size(d);
    //        }

    int tmp = i;
    if (swap_xy)
    {
        for (int d = D - 1; d >= 0; --d)
        {
            res(d) = tmp % size(d);
            tmp /= size(d);
        }
    }
    else
    {
        for (int d = 0; d < D; ++d)
        {
            res(d) = tmp % size(d);
            tmp /= size(d);
        }
    }

    return res;
}

HD inline std::tuple<bool, float, float> IntersectBoxRayPrecise(float* pos_min, float* pos_max, float* origin,
                                                                float* direction, int D)
{
    double t_near = -236436436;  // maximums defined in float.h
    double t_far  = 43637575;

    for (int i = 0; i < D; i++)
    {  // we test slabs in every direction
        if (direction[i] == 0)
        {  // ray parallel to planes in this direction
            if ((origin[i] < pos_min[i]) || (origin[i] >= pos_max[i]))
            {
                return {false, 0, 0};  // parallel AND outside box : no intersection possible
            }
        }
        else
        {  // ray not parallel to planes in this direction
            float T_1 = (pos_min[i] - origin[i]) / direction[i];
            float T_2 = (pos_max[i] - origin[i]) / direction[i];

            if (T_1 > T_2)
            {  // we want T_1 to hold values for intersection with near plane
                // std::swap(T_1, T_2);
                auto tmp = T_1;
                T_1      = T_2;
                T_2      = tmp;
            }
            if (T_1 > t_near)
            {
                t_near = T_1;
            }
            if (T_2 < t_far)
            {
                t_far = T_2;
            }
            if ((t_near > t_far) || (t_far < 0))
            {
                return {false, 0, 0};
            }
        }
    }
    return {true, t_near, t_far};
}

HD inline bool BoxContainsPoint(float* pos_min, float* pos_max, float* p, int D)
{
    for (int d = 0; d < D; ++d)
    {
        if (pos_min[d] > p[d] || pos_max[d] < p[d])
        {
            return false;
        }
    }
    return true;
}


class HyperTreeBaseImpl : public virtual torch::nn::Module, public torch::nn::Cloneable<HyperTreeBaseImpl>
{
   public:
    HyperTreeBaseImpl(int d, int max_depth);
    virtual ~HyperTreeBaseImpl() {}

    NodeBatchedSamples GroupSamplesPerNodeGPU(const SampleList& samples, int group_size);

    // For each inactive node we compute the output features as if they had been sampled normally.
    // Used to optimize network after a structure update to a reasonable initial solution.
    //
    // Input:
    //    active_grid: float [num_nodes, num_features, x, y, z]
    // Output:
    //    interpolated_grid: float [num_nodes, num_features, x, y, z]
    torch::Tensor InterpolateGridForInactiveNodes(torch::Tensor active_grid);

    torch::Tensor GetNodePositionScaleForId(torch::Tensor node_ids)
    {
        CHECK(node_position_min.defined());
        CHECK(node_scale.defined());
        auto pos = torch::index_select(node_position_min, 0, node_ids);
        auto sca = torch::index_select(node_scale, 0, node_ids).unsqueeze(1);
        return torch::cat({pos, sca}, 1);
    }

    torch::Tensor ActiveNodeTensor() { return active_node_ids; }

    // The input are the samples batched by node id!
    // global_samples
    //      float [num_groups, group_size, D]
    // node_indices
    //      long [num_groups]
    torch::Tensor ComputeLocalSamples(torch::Tensor global_samples, torch::Tensor node_indices);

    // Traverse the tree and finds the active node id containing the sample position.
    // The returned masked indicates if this sample is valid (i.e. the sample is inside an active node).
    // The mask is should be only 0 if the containing node has been culled or the sample is outside of [-1,1]
    //
    // Input
    //      global_samples float [..., D]
    // Return
    //      node_id, long [N]
    //      mask,    float [N]
    //
    std::tuple<torch::Tensor, torch::Tensor> NodeIdForPositionGPU(torch::Tensor global_samples);

    // For each grid element x we check if x+epsilon is in a different node.
    // If yes, this is a edge-element and we generate 2 output samples using the same coordinates.
    //
    // This should be used to greate a edge-regularizer that makes neighboring nodes have the same value at the edge.
    //
    SampleList NodeNeighborSamples(Eigen::Vector<int, -1> size, double epsilon, vec3 roi_min, vec3 roi_max, bool use_roi);

    // bool in_roi(vec3 node_min, vec3 node_max, vec3 roi_min, vec3 roi_max);

    SampleList UniformPhantomSamplesGPU(Eigen::Vector<int, -1> size, bool swap_xy);

    SampleList UniformPhantomSamplesGPUbySlice(Eigen::Vector<int, -1> size, bool swap_xy, Eigen::Vector<float, 3> roi_min, Eigen::Vector<float, 3> step_size);

    torch::Tensor UniformPhantomSamplesGPUSlice_global(Eigen::Vector<int, -1> size, bool swap_xy, Eigen::Vector<float, 3> roi_min, Eigen::Vector<float, 3> step_size);

    SampleList CreateSamplesForRays(const RayList& rays, int max_samples_per_node, bool interval_jitter);

    torch::Tensor CreateSamplesRandomly(torch::Tensor node_id, int num_of_samples_per_edge);

    // If all children of a node are culled this node will be also culled
    void UpdateCulling();

    // Given a node id, this function creates uniform sample locations inside this node (in global space)
    //
    // Input
    //      node_id int [N]
    // Return
    //      position, float [N, grid_size, grid_size, grid_size, D]
    torch::Tensor UniformGlobalSamples(torch::Tensor node_id, int grid_size);
    torch::Tensor RandomGlobalSamples(torch::Tensor node_id, int grid_size);

    void SetErrorForActiveNodes(torch::Tensor error, std::string strategy = "override");

    void SplitNode(int to_split) {}
    void ResetLoss() {}

    void FinalizeExp(double alpha) {}

    // Sets all nodes of layer i to active.
    // All others to inactive
    void SetActive(int depth);
    void UpdateActive();

    torch::Tensor GlobalNodeIdToLocalActiveId(torch::Tensor node_id)
    {
        auto res = torch::index_select(node_active_prefix_sum, 0, node_id.reshape({-1}));
        return res.reshape(node_id.sizes());
    }

    // Return
    //      long [num_inactive_nodes]
    torch::Tensor InactiveNodeIds();

    std::vector<AABB> ActiveNodeBoxes();

    virtual void reset();

    torch::Device device() { return node_parent.device(); }
    int NumNodes() { return node_parent.size(0); }
    int NumActiveNodes() { return active_node_ids.size(0); }
    int D() { return node_position_min.size(1); }
    int NS() { return node_children.size(1); }

    template <int U>
    friend struct DeviceHyperTree;

    // int [num_nodes]
    // -1 for the root node
    torch::Tensor node_parent;

    // int [num_nodes, NS]
    // -1 for leaf nodes
    torch::Tensor node_children;

    // float [num_nodes, D]
    torch::Tensor node_position_min, node_position_max;

    // float [num_nodes]
    torch::Tensor node_scale;

    // float [num_nodes] (invalid error == -1)
    torch::Tensor node_error;

    // float [num_nodes]
    torch::Tensor node_max_density;

    // int [num_nodes]
    torch::Tensor node_depth;

    // int [num_nodes] boolean 0/1
    torch::Tensor node_active;

    // int [num_nodes] boolean 0/1 default all 0
    torch::Tensor node_culled;

    // Exclusive prefix sum of 'node_active'
    // long [num_nodes]
    torch::Tensor node_active_prefix_sum;

    // long [num_active_nodes]
    torch::Tensor active_node_ids;
};

TORCH_MODULE(HyperTreeBase);