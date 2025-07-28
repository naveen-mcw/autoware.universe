#ifndef AUTOWARE__TENSORRT_BEVFORMER__BEVFORMER_DATA_LOADER_HPP_
#define AUTOWARE__TENSORRT_BEVFORMER__BEVFORMER_DATA_LOADER_HPP_

#include <opencv2/core.hpp>
#include <vector>

namespace autoware
{
namespace tensorrt_bevformer
{

class BEVFormerDataLoader
{
public:
  BEVFormerDataLoader();
  ~BEVFormerDataLoader() = default;

  // Creates a flat tensor from 6 images in CHW format, shape [1, 6*3*H*W]
  cv::Mat createImageTensor(const std::vector<cv::Mat> & images);
};

}  // namespace tensorrt_bevformer
}  // namespace autoware

#endif