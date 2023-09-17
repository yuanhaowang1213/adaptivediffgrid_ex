/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/freeimage.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/time.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/torch/ImageTensor.h"

#include "../Settings.h"
#include "cnpy.h"

#include "../build_config.h"
#include "tensorboard_logger.h"

using namespace Saiga;

inline std::string EncodeImageToString(const Image& img, std::string type = "png")
{
    auto data = img.saveToMemory(type);

    std::string result;
    result.resize(data.size());

    memcpy(result.data(), data.data(), data.size());
    return result;
}

template <typename T>
inline void LogImage(TensorBoardLogger* tblogger, const TemplatedImage<T>& img, std::string name, int step)
{
    auto img_str = EncodeImageToString(img, "png");
    tblogger->add_image(name, step, img_str, img.h, img.w, channels(img.type));
}

inline void LogImage(TensorBoardLogger* tblogger, const Image& img, std::string name, int step)
{
    auto img_str = EncodeImageToString(img, "png");
    tblogger->add_image(name, step, img_str, img.h, img.w, channels(img.type));
}

inline torch::Tensor loadNumpyIntoTensor(std::string dir)
{
    auto arr = cnpy::npy_load(dir);

    std::vector<long> shape;
    for (auto s : arr.shape)
    {
        shape.push_back(s);
    }

    if (arr.word_size == 8)
    {
        return torch::from_blob(arr.data<double>(), shape, torch::kDouble).clone();
    }
    else
    {
        return torch::from_blob(arr.data<float>(), shape, torch::kFloat).clone();
    }
}
