/* Yuanhao 's code */


#include "saiga/core/util/directory.h"
#include "saiga/vision/torch/ColorizeTensor.h"

#include "Settings.h"
#include "utils/utils.h"
#include "data/SceneBase.h"
#include "build_config.h"
#include "tensorboard_logger.h"
#include "utils/cimg_wrapper.h"

using namespace Saiga;
using namespace torch::indexing;

int main(int argc, const char* argv[])
{
    torch::NoGradGuard ngg;
    
    
    for (int i = 0; i < 1 ;++i)
    {
        std::string volume_file, angle_file, out_dir, mask_file, file_dir;
        // float volume_shift = 0.2;
        Vec3 volume_shift;
        // float volume_shift;
        int volume_size;


        if(i == 0)
        {
            // volume_file = "/home/wangy0k/Desktop/owntree/hyperacorncontinue/scenes/tutorial1k/tutoral_1k_org.hdr";
            // angle_file  = "/home/wangy0k/Desktop/owntree/hyperacorncontinue/scenes/tutorial1k/tutorial_proj_1k_org.txt";
            // out_dir     = "/home/wangy0k/Desktop/owntree/hyperacorncontinue/scenes/tutorial1k";
            // volume_size = 1024;
            file_dir = "/home/wangy0k/Desktop/owntree/hyperacorncontinue/scenes/10643_51_bin8";
            volume_file = file_dir+"/10643_51_bin8.hdr";
            angle_file  = file_dir+"/b3tilt51_bin4.tlt";
            mask_file   = file_dir+"/mask.hdr";
            out_dir     = file_dir;
            volume_size = 1024;
            volume_shift= Vec3(0 ,0, 0.2);
            // volume_shift= 0.2;

        }


        if(i == 1)
        {
            file_dir = "/home/wangy0k/Desktop/owntree/hyperacorncontinue/scenes/10820_emd25049_bin8_crop2";
            volume_file = file_dir+"/Zad44_No4_20201013_s103_ali_bin8_fid.hdr";
            angle_file  = file_dir+"/Zad44_No4_20201013_s103.tlt";
            mask_file   = file_dir+"/masks.hdr";
            out_dir     = file_dir;
            volume_size = 1044;
            volume_shift= Vec3(0 ,0, 0.2);
            // volume_shift= 0.2;
        }

        std::filesystem::create_directories(out_dir);
        at::Tensor volume = LoadHDRImageTensor(volume_file);

        at::Tensor mask;
        if(std::filesystem::exists(mask_file))
        {
            mask   = LoadHDRImageTensor(mask_file); 
        }
        std::cout << "volume " << TensorInfo(volume) << std::endl;
        std::cout << "mask " << TensorInfo(mask) << std::endl;


        bool in_degree = false; // whether change the degree or not

        // out_params.image_formation = "cryo";
        // out_params.image_formation = "xray";
        // out_params.num_channels    = 1;
        // out_params.image_dir = out_dir + "/images";

        DatasetParams out_params(out_dir + "/dataset.ini");
        out_params.camera_model    = "orthographic";        

        out_params.image_dir       = out_dir + "/images";
        out_params.mask_dir        = out_dir + "/masks";
        // out_params.scene_scale      = 1;
        out_params.log_space_input      = false;
        out_params.use_log10_conversion = false;
        out_params.laplacian_size       = 5;
        out_params.projection_scale     = 3;
        out_params.camera_proj_scale    = (float)volume_size/volume.size(3);


        // out_params.mask_dir  = out_dir + "/masks";
        std::cout << out_params.image_dir << std::endl;
        std::filesystem::create_directories(out_params.image_dir);
        std::filesystem::create_directories(out_params.mask_dir);

        std::string exp_dir = out_dir + "/exp_all";
        std::filesystem::create_directories(exp_dir);

        // std::filesystem::create_directories(out_params.mask_dir);

        std::vector<double> angles;
        std::cout << "angle file " << angle_file << std::endl;
        if(std::filesystem::exists(angle_file))
        {
            printf("angle file exists\n");
            std::ifstream istream1(angle_file);
            std::string line;
            while(std::getline(istream1, line))
            {
                std::stringstream sstream(line);
                double angle;
                sstream >> angle;
                angles.push_back(angle);
            }
        }
        std::cout << "input angle size " << angles.size() << std::endl;
        


        bool normalize = false;



        // volume = volume.slice(1, 0, volume.size(1));

        // auto center_slice = volume.slice(1, volume.size(1) / 2, volume.size(2) / 2 + 1 + 50).squeeze(1);
        // auto slice_color  = ColorizeTensor(center_slice.squeeze(0), colorizeTurbo);
        // TensorToImage<ucvec3>(slice_color).save(out_dir + "/center_slice.png");

        double shift_value     = 0; 
        double src_to_object   = 2. - shift_value;
        double src_to_detector = 14.4;

        double detector_size = 2;
        double detector_to_object = 2. + shift_value;

        double volume_factor = (double)volume_size/volume.size(2);

        CameraBase cam;
        cam.w    = volume.size(2);
        cam.h    = volume.size(3);
        cam.K.cx = cam.w  / 2.;
        cam.K.cy = cam.h / 2.;
        cam.K.fx = cam.w * volume_factor * 0.5;
        cam.K.fy = cam.h * volume_factor * 0.5;
        std::filesystem::remove(out_dir + "/camera.ini");
        cam.Save(out_dir + "/camera.ini");

        {
            std::ofstream ostream1(out_dir + "/info.txt");
            ostream1 << "src_to_object " << src_to_object << std::endl;
            ostream1 << "src_to_detector " << src_to_detector << std::endl;
            ostream1 << "detector_size " << detector_size << std::endl;
            ostream1 << "volume_size 2 x 2 x 2" << std::endl;
            ostream1 << "voxel_size " << 2. / volume_size << " x " << 2. / volume_size << " x "
                        << 2. / volume_size << std::endl;
            ostream1 << "output_volume " << volume_size << " x " << volume_size << " x " << volume_size
                        << std::endl;
        }

        std::ofstream ostream1(out_dir + "/images.txt");
        std::ofstream ostream2(out_dir + "/masks.txt");
        std::vector<std::string> new_image_names;
        std::vector<std::string> new_mask_names;
        for(int i = 0; i < volume.size(1); ++i)
        {
            std::string new_image_name = leadingZeroString(i, 4) + ".tiff";
            new_image_names.push_back(new_image_name);
            std::string new_image_name2 = leadingZeroString(i, 4) + ".png";
            new_mask_names.push_back(new_image_name2);
            ostream1 << new_image_name << std::endl;
            ostream2 << new_image_name2 << std::endl;
        }
        // #pragma omp parallel for num_threads(8)
        // for(int i = 0; i < image_names.size(); i++)
        // {
        //     TemplateImage<unsigned short> raw()        
        // }
        double max_intensity = volume.max().item().toFloat();

        float total_min = 23462364;
        float total_max = -1;

        if(normalize)
        {
            volume *= 1.0/max_intensity;
        }
        std::cout << "projection max value " << std::numeric_limits<unsigned short>::max() << std::endl;
        std::ofstream strm1(out_dir + "/exp_all/train.txt");
        std::ofstream strm2(out_dir + "/exp_all/eval.txt");
        for(int i = 0; i < volume.size(1); ++i)
        {
            auto im = volume.index({0,i}).unsqueeze(0).unsqueeze(0);
            // std::cout << "im " << TensorInfo(im) << std::endl;
            PrintTensorInfo(im);
            auto im1 = TensorToImage<float> (im);

            if(normalize)
            {
                auto colorized = ImageTransformation::ColorizeTurbo(im1);
                TemplatedImage<unsigned short> im1_new(im1.dimensions());

                for(int i : im1.rowRange())
                {
                    for(int j : im1.colRange())
                    {
                        im1_new(i,j) = im1(i,j) * std::numeric_limits<unsigned short>::max();
                    }
                }
                im1_new.save(out_params.image_dir + "/" + new_image_names[i]);
            }
            else
            {
                im1.save(out_params.image_dir + "/" + new_image_names[i]);
            }


            float mi, ma;
            im1.getImageView().findMinMaxOutlier(mi, ma, 0);
            std::cout << "min max " << mi << " " << ma << std::endl;

            total_min = std::min(mi, total_min);
            total_max = std::max(ma, total_max);

            if(std::filesystem::exists(mask_file))
            {
                auto im2 = mask.index({0,i}).unsqueeze(0);
                auto im_tmp = TensorToImage<float> (im2);
                TemplatedImage<unsigned char> im3(im_tmp.dimensions());
                for(int i : im_tmp.rowRange())
                {
                    for(int j : im_tmp.colRange())
                    {
                        if(im_tmp(i,j) > 0.5)
                        {
                            im3(i,j) =  std::numeric_limits<unsigned char>::max();
                        }
                        else
                        {
                            im3(i,j) = 0;
                        }
                        
                    }
                }
                im3.save(out_params.mask_dir + "/" + new_mask_names[i]);
            }

            strm1 << i << "\n";
            strm2 << i << "\n";
        }


        std::filesystem::remove(out_dir + "/dataset.ini");
        out_params.xray_min        = total_min;
        out_params.xray_max        = total_max;
        out_params.Save(out_dir + "/dataset.ini");
        std::vector<SE3> poses;
        {
            std::ofstream ostream2(out_dir + "/camera_indices.txt");
            std::ofstream strm2(out_dir + "/poses.txt");
            std::ofstream strm3(out_dir + "/angles.txt");
            for(int i = 0; i < volume.size(1); ++i)
            {
                double ang = angles[i]/180. * pi<double>();
                // double ang = 0;
                if(in_degree)
                {
                    ang = ang;
                }
                else
                {
                    ang = -ang;
                }
                strm3 << ang << "\n";
                Vec3 dir             = Vec3(sin(-ang), -cos(-ang), 0);

                // printf("direction is \n");
                // std::cout  << dir << std::endl;
                Vec3 source          = (src_to_object )* dir;
                // printf("souce is \n");
                // std::cout << source << std::endl;
                Vec3 detector_center = -dir * (detector_to_object);

                // printf("detector center \n");
                // std::cout << detector_center << std::endl;
                Vec3 up      = Vec3(0, 0, -1);

                Vec3 forward = (detector_center - source).normalized();

                // printf("forward is \n");
                // std::cout << forward << std::endl;
                Vec3 right   = up.cross(forward).normalized();
                // printf("right is \n");
                // std::cout << right << std::endl;

                Mat3 R;
                R.col(0) = right;
                R.col(1) = up;
                R.col(2) = forward;

                ostream2 << 0 << std::endl;

                // std::cout << "R is " << R << std::endl;
                // char c = getchar();
                Quat q = Sophus::SO3d(R).unit_quaternion();
                // printf("q is \n");
                // std::cout << q << std::endl;
                Vec3 t = source;
                // Vec3 t = Vec3(0,0.2,0) ;

                // for(int ii = 0; ii < 3; ++ii)
                // {
                //     t(ii) = volume_shift(ii) * dir(ii);
                // }

                // std::cout << "translation value " << t << std::endl;
                // char c = getchar();
                poses.push_back(SE3(q, t));

                strm2 << std::scientific << std::setprecision(15);
                strm2 << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " " << t.x() << " " << t.y() << " "
                        << t.z() << "\n";
            }
        }
        printf("test here \n");

    }

    return 0;
}