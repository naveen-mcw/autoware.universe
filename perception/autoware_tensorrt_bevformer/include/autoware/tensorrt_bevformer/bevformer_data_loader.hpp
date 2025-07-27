// Copyright 2025 AutoCore, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AUTOWARE__TENSORRT_BEVFORMER__BEVFORMER_DATA_LOADER_HPP_
#define AUTOWARE__TENSORRT_BEVFORMER__BEVFORMER_DATA_LOADER_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <array>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace autoware
{
namespace tensorrt_bevformer
{

// Structure to store image normalization configuration
struct ImgNormConfig
{
  std::array<float, 3> mean;
  std::array<float, 3> std;
  bool to_rgb;
};

// Class to hold all metadata for BEVFormer input
class ImageMetaData
{
public:
  std::vector<std::string> filenames;
  std::vector<std::array<int, 3>> ori_shapes;
  std::vector<std::array<int, 3>> img_shapes;
  std::vector<std::array<int, 3>> pad_shapes;
  std::vector<cv::Mat> lidar2img_matrices;
  float scale_factor{1.0};
  bool flip{false};
  bool pcd_horizontal_flip{false};
  bool pcd_vertical_flip{false};
  int box_mode_3d{0};  // LIDAR mode
  ImgNormConfig img_norm_cfg;
  std::string sample_token;
  std::string prev_token;
  std::string next_token;
  float pcd_scale_factor{1.0};
  std::string pts_filename;
  std::string scene_token;
  std::vector<float> can_bus;
};

class BEVFormerDataLoader
{
public:
  BEVFormerDataLoader();
  ~BEVFormerDataLoader() = default;

  /**
   * @brief Creates the nested data structure expected by BEVFormer
   */
  ImageMetaData createImageMetadata(
    const std::vector<cv::Mat> & preprocessed_images,
    const std::vector<cv::Mat> & lidar2img_matrices, const std::vector<float> & can_bus,
    const std::string & sample_token, const std::string & prev_token,
    const std::string & next_token, const std::string & scene_token,
    const std::pair<int, int> & target_shape = {736, 1280});

  /**
   * @brief Creates a 5D tensor from preprocessed images [1, 6, 3, H, W]
   *
   * @note Assumes images are already in CV_32FC3 and properly normalized.
   */
  cv::Mat createImageTensor(
    const std::vector<cv::Mat> & images,
    const ImgNormConfig & norm_cfg = {{0, 0, 0}, {1, 1, 1}, false}  // default no-op
  );

  /**
   * @brief Provides default normalization parameters (BEVFormer standard)
   */
  ImgNormConfig getDefaultNormConfig()
  {
    return {{103.53f, 116.28f, 123.675f}, {1.0f, 1.0f, 1.0f}, false};
  }

private:
  std::pair<int, int> default_shape_{736, 1280};  // Height, Width

  /**
   * @brief Applies normalization to an image (used only if needed explicitly)
   */
  cv::Mat normalizeImage(const cv::Mat & image, const ImgNormConfig & norm_cfg);
};

// Generic dictionary for structured data (if needed in future)
using DataDict = std::map<
  std::string, std::variant<
                 bool, int, float, std::string, std::vector<float>, std::vector<int>,
                 std::vector<std::string>, std::vector<cv::Mat>, std::vector<cv::Size_<int>>>>;

}  // namespace tensorrt_bevformer
}  // namespace autoware

#endif  // AUTOWARE__TENSORRT_BEVFORMER__BEVFORMER_DATA_LOADER_HPP_

// #ifndef AUTOWARE__TENSORRT_BEVFORMER__BEVFORMER_DATA_LOADER_HPP_
// #define AUTOWARE__TENSORRT_BEVFORMER__BEVFORMER_DATA_LOADER_HPP_

// #include <opencv2/opencv.hpp>
// #include <Eigen/Core>
// #include <Eigen/Geometry>
// #include <vector>
// #include <array>
// #include <map>
// #include <string>
// #include <memory>
// #include <unordered_map>
// #include <variant>
// #include <opencv2/core.hpp>

// namespace autoware {
// namespace tensorrt_bevformer {

// // Structure to store image normalization configuration
// struct ImgNormConfig {
//     std::array<float, 3> mean;
//     std::array<float, 3> std;
//     bool to_rgb;
// };

// // Structure to store metadata for each camera
// struct CameraMetadata {
//     std::string filename;  // For compatibility, may not be used in ROS
//     std::array<int, 3> ori_shape;
//     std::array<int, 3> img_shape;
//     std::array<int, 3> pad_shape;
//     cv::Mat lidar2img;  // 4x4 transformation matrix
// };

// // Class to hold all metadata for BEVFormer input
// class ImageMetaData {
// public:
//     std::vector<std::string> filenames;
//     std::vector<std::array<int, 3>> ori_shapes;
//     std::vector<std::array<int, 3>> img_shapes;
//     std::vector<std::array<int, 3>> pad_shapes;
//     std::vector<cv::Mat> lidar2img_matrices;
//     float scale_factor{1.0};
//     bool flip{false};
//     bool pcd_horizontal_flip{false};
//     bool pcd_vertical_flip{false};
//     int box_mode_3d{0};  // LIDAR mode
//     ImgNormConfig img_norm_cfg;
//     std::string sample_token;
//     std::string prev_token;
//     std::string next_token;
//     float pcd_scale_factor{1.0};
//     std::string pts_filename;  // Optional, for compatibility
//     std::string scene_token;
//     std::vector<float> can_bus;  // Vehicle state data
// };

// class BEVFormerDataLoader {
// public:
//     BEVFormerDataLoader();
//     ~BEVFormerDataLoader() = default;

//     /**
//      * @brief Creates the nested data structure expected by BEVFormer
//      *
//      * @param preprocessed_images Vector of preprocessed camera images
//      * @param lidar2img_matrices Lidar to image transformation matrices
//      * @param can_bus CAN bus data with vehicle state
//      * @param sample_token Current sample token
//      * @param prev_token Previous sample token
//      * @param next_token Next sample token
//      * @param scene_token Scene token
//      * @param target_shape Target shape for images [height, width]
//      * @return ImageMetaData Complete metadata structure
//      */
//     ImageMetaData createImageMetadata(
//         const std::vector<cv::Mat>& preprocessed_images,
//         const std::vector<cv::Mat>& lidar2img_matrices,
//         const std::vector<float>& can_bus,
//         const std::string& sample_token,
//         const std::string& prev_token,
//         const std::string& next_token,
//         const std::string& scene_token,
//         const std::pair<int, int>& target_shape = {736, 1280}
//     );

//     /**
//      * @brief Create a 5D tensor from 6 camera images in format [1, 6, 3, H, W]
//      *
//      * @param images Vector of 6 camera images
//      * @param norm_cfg Normalization configuration
//      * @return cv::Mat 5D tensor with shape [1, 6, 3, H, W]
//      */
//     cv::Mat createImageTensor(
//         const std::vector<cv::Mat>& images,
//         const ImgNormConfig& norm_cfg
//     );

//     /**
//      * @brief Get the default normalization configuration
//      *
//      * @return ImgNormConfig with BEVFormer default values
//      */
//     ImgNormConfig getDefaultNormConfig() {
//         return {{103.53f, 116.28f, 123.675f}, {1.0f, 1.0f, 1.0f}, false};
//     }

// private:
//     std::pair<int, int> default_shape_{736, 1280};  // Height, Width

//     /**
//      * @brief Normalize an image according to the given configuration
//      *
//      * @param image Input image
//      * @param norm_cfg Normalization configuration
//      * @return cv::Mat Normalized image
//      */
//     cv::Mat normalizeImage(const cv::Mat& image, const ImgNormConfig& norm_cfg);
// };

// // Remove duplicate or unnecessary types and ensure only one of each type
// using DataDict = std::map<std::string, std::variant<
//     bool,
//     int,
//     float,
//     std::string,
//     std::vector<float>,
//     std::vector<int>,
//     std::vector<std::string>,
//     std::vector<cv::Mat>,
//     std::vector<cv::Size_<int>>
// >>;

// }  // namespace tensorrt_bevformer
// }  // namespace autoware

// #endif  // AUTOWARE__TENSORRT_BEVFORMER__BEVFORMER_DATA_LOADER_HPP_
