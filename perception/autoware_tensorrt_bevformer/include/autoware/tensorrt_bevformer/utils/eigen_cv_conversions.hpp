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

#ifndef AUTOWARE__TENSORRT_BEVFORMER__UTILS__EIGEN_CV_CONVERSIONS_HPP_
#define AUTOWARE__TENSORRT_BEVFORMER__UTILS__EIGEN_CV_CONVERSIONS_HPP_

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace autoware
{
namespace tensorrt_bevformer
{

// Convert Eigen matrix to OpenCV matrix
template <typename EigenType>
inline void eigen2cv(const EigenType & eigen_mat, cv::Mat & cv_mat)
{
  if (
    cv_mat.type() != CV_32F || cv_mat.rows != eigen_mat.rows() || cv_mat.cols != eigen_mat.cols()) {
    cv_mat = cv::Mat::zeros(eigen_mat.rows(), eigen_mat.cols(), CV_32F);
  }

  for (int i = 0; i < eigen_mat.rows(); ++i) {
    for (int j = 0; j < eigen_mat.cols(); ++j) {
      cv_mat.at<float>(i, j) = static_cast<float>(eigen_mat(i, j));
    }
  }
}

// Convert OpenCV matrix to Eigen matrix
template <typename EigenType>
inline void cv2eigen(const cv::Mat & cv_mat, EigenType & eigen_mat)
{
  eigen_mat.resize(cv_mat.rows, cv_mat.cols);
  for (int i = 0; i < cv_mat.rows; ++i) {
    for (int j = 0; j < cv_mat.cols; ++j) {
      eigen_mat(i, j) = cv_mat.at<float>(i, j);
    }
  }
}

}  // namespace tensorrt_bevformer
}  // namespace autoware

#endif  // AUTOWARE__TENSORRT_BEVFORMER__UTILS__EIGEN_CV_CONVERSIONS_HPP_
