
#pragma once

#include "saiga/core/image/freeimage.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/time.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/torch/ImageTensor.h"

#include "../Settings.h"
#include "../modules/CameraPose.h"
#include "../modules/ToneMapper.h"
#include "../structure/HyperTree.h"
#include "../utils/utils.h"

#include "../geometry/laplacian.h"

constexpr bool uv_align_corners = true;

enum class CameraModel
{
    ORTHOGRAPHIC = 1,
};

struct UnifiedImage
{
    virtual ~UnifiedImage() {}

    // in coordinates: float [num_coords, 2]
    //
    // returns
    //  intensity float [num_channels, num_coords]
    //  mask      float [1, num_coords]
    std::pair<torch::Tensor, torch::Tensor> SampleProjection(torch::Tensor uv);


    // Return the UV coordinates for a pixel row
    // return float [num_column, 2]
    torch::Tensor CoordinatesRow(int row_start, int row_end, int w, int h);

    // Return random UV coordinates
    // return float [count, 2]
    torch::Tensor CoordinatesRandom(int count) { return torch::rand({count, 2}); }


    torch::Tensor CoordinatesRandomNoInterpolate(int count, int w, int h);

    int NumChannels() const { return projection.size(0); }


    // float [num_channels, shape]
    torch::Tensor projection;

    std::vector<torch::Tensor> projections;
    std::vector<torch::Tensor> masks;



    // 0-1 Mask signaling if a pixel is valid
    // float [1, shape]
    torch::Tensor mask;

    int camera_id = 0;
    // world->camera transform in OpenCV convention
    // z points forward
    // y points down
    SE3 pose;

    std::string image_file, mask_file;
};

struct CameraBase : public ParamsBase
{
    SAIGA_PARAM_STRUCT_FUNCTIONS(CameraBase);
    virtual ~CameraBase() {}

    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    {
        SAIGA_PARAM(w);
        SAIGA_PARAM(h);

        auto vector2string = [](auto vector)
        {
            std::stringstream sstrm;
            sstrm << std::setprecision(15) << std::scientific;
            for (unsigned int i = 0; i < vector.size(); ++i)
            {
                sstrm << vector[i];
                if (i < vector.size() - 1) sstrm << " ";
            }
            return sstrm.str();
        };



        {
            std::vector<std::string> K = split(vector2string(this->K.cast<double>().coeffs()), ' ');
            SAIGA_PARAM_LIST_COMMENT(K, ' ', "# fx fy cx cy s");
            SAIGA_ASSERT(K.size() == 5);

            Vector<double, 5> K_coeffs;
            for (int i = 0; i < 5; ++i)
            {
                double d    = to_double(K[i]);
                K_coeffs(i) = d;
            }
            this->K = IntrinsicsPinholed(K_coeffs);
        }
    }

    int w = 0;
    int h = 0;

    // only if type == pinhole or type == orthographic
    // Note, for parallel beam we still use the K matrix but don't do the z division.
    // The values in K therefore mean different things
    IntrinsicsPinholed K;
};

class SceneBase
{
   public:
    SceneBase(std::string _scene_dir);

    void Draw(TensorBoardLogger* logger);

    void save(std::string dir);

    void Finalize();

    void Finalize2();

    void OptimizerStep(int epoch_id, bool only_image_params);

    void PrintGradInfo(int epoch_id, TensorBoardLogger* logger);
    void PrintInfo(int epoch_id, TensorBoardLogger* logger);

    void train(bool on)
    {
        pose->train(on);
        camera_model->train(on);
        tone_mapper->train(on);
        if (structure_optimizer) structure_optimizer->zero_grad();
        if (tm_optimizer_adam) tm_optimizer_adam->zero_grad();
        if (tm_optimizer_sgd) tm_optimizer_sgd->zero_grad();
    }

    void UpdateLearningRate(double factor)
    {
        if (structure_optimizer) UpdateLR(structure_optimizer.get(), factor);
        if (tm_optimizer_adam) UpdateLR(tm_optimizer_adam.get(), factor);
        if (tm_optimizer_sgd) UpdateLR(tm_optimizer_sgd.get(), factor);
    }

    void setcurrentproj(int scale,std::vector<int> indices); //new add

    void setmoment(std::vector<torch::Tensor> projection_images,std::vector<int> indices, float factor, std::string save_file_name);

    void InitializeBiasWithBackground(TensorBoardLogger* logger);

    void NormalizeIndividualImages(TensorBoardLogger* logger);

    // Computes the ray parameters (origin+direction) from the uv coordinates
    // and the camera parameters.
    // The camera parameters are stored in 'pose' and 'model' and are selected by the image_id.
    RayList GetRays(torch::Tensor uv, torch::Tensor image_id, torch::Tensor camera_id);

    // Projects each input points to all(!) images and sets the output to 1 if at least one image sees this point.
    // Today sample image? + threshold?
    // Input:
    //      float [..., 3]
    // Output:
    //      float [..., 3]
    torch::Tensor PointInAnyImage(torch::Tensor points);


    std::vector<CameraBase> cameras;
    std::vector<std::shared_ptr<UnifiedImage>> frames;
    std::vector<int> train_indices, test_indices;

    // int [num_train_images]
    torch::Tensor active_train_images, active_test_images;

    CameraPoseModule pose          = nullptr;
    CameraModelModule camera_model = nullptr;
    std::shared_ptr<torch::optim::Optimizer> structure_optimizer;
    std::shared_ptr<torch::optim::Optimizer> tm_optimizer_adam;
    std::shared_ptr<torch::optim::Optimizer> tm_optimizer_sgd;

    PhotometricCalibration tone_mapper = nullptr;
    std::shared_ptr<CombinedParams> params;

    DatasetParams dataset_params;
    std::string scene_path;
    std::string scene_name;

    torch::Tensor ground_truth_volume;

    torch::Tensor SampleGroundTruth(torch::Tensor global_coordinates);


    void LoadImagesCT(std::vector<int> indices);

    void SaveCheckpoint(const std::string& dir);
    void LoadCheckpoint(const std::string& dir)
    {
        auto prefix = dir + "/" + scene_name + "_";
        if (pose && std::filesystem::exists(prefix + "pose.pth"))
        {
            std::cout << "Load Checkpoint pose" << std::endl;
            torch::load(pose, prefix + "pose.pth");
        }
        if (camera_model && std::filesystem::exists(prefix + "camera_model.pth"))
        {
            std::cout << "Load Checkpoint camera_model" << std::endl;
            torch::load(camera_model, prefix + "camera_model.pth");
        }
        if (tone_mapper && std::filesystem::exists(prefix + "tone_mapper.pth"))
        {
            std::cout << "Load Checkpoint tone_mapper" << std::endl;
            torch::load(tone_mapper, prefix + "tone_mapper.pth");
        }
    }
    // create laplacian 2d filter
    Laplacian2D laplacian2d     =   Laplacian2D( dataset_params.laplacian_size,1);
    // laplacian2d = Laplacian2D( dataset_params.laplacian_size,1);
    int num_channels = 1;
    int D            = 3;

    int ori_w = 0;
    int ori_h = 0;
    int rays_per_image_level0 = 0;

    torch::Tensor vol_translate_para = torch::zeros({3}, torch::TensorOptions(torch::kFloat32));
   private:
    // Loads an image and converts it to a float tensor.
    // No normalization is done.
    // If the input image is 16-bit uint, then the float output will also be i nthe range [0,65556]
    torch::Tensor LoadImageRaw(std::string file);
};

// RayList GenerateRays(torch::Tensor uv, torch::Tensor image_id, )
