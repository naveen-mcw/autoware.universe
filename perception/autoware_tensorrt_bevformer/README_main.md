# tensorrt_bevformer <!-- cspell: ignore bevformer -->

## Purpose

The core algorithm, named `BEVFormer`, unifies multi-view images into the BEV perspective for 3D object detection tasks with temporal fusion.

## Inner-workings / Algorithms

### Cite

- Zhicheng Wang, et al., "BEVFormer: Incorporating Transformers for Multi-Camera 3D Detection" [[ref](https://arxiv.org/abs/2203.17270)]
- The Node is ported and adapted from the [original Python implementation by DerryHub](https://github.com/DerryHub/BEVFormer_tensorrt.git), re-written for Autoware with C++/TensorRT support.

## Inputs / Outputs

### Inputs

| Name                                         | Type                           | Description                         |
| -------------------------------------------- | ------------------------------ | ----------------------------------- |
| `~/input/topic_img_front_left`               | `sensor_msgs::msg::Image`      | input front_left camera image       |
| `~/input/topic_img_front`                    | `sensor_msgs::msg::Image`      | input front camera image            |
| `~/input/topic_img_front_right`              | `sensor_msgs::msg::Image`      | input front_right camera image      |
| `~/input/topic_img_back_left`                | `sensor_msgs::msg::Image`      | input back_left camera image        |
| `~/input/topic_img_back`                     | `sensor_msgs::msg::Image`      | input back camera image             |
| `~/input/topic_img_back_right`               | `sensor_msgs::msg::Image`      | input back_right camera image       |
| `~/input/topic_img_front_left/camera_info`   | `sensor_msgs::msg::CameraInfo` | input front_left camera parameters  |
| `~/input/topic_img_front/camera_info`        | `sensor_msgs::msg::CameraInfo` | input front camera parameters       |
| `~/input/topic_img_front_right/camera_info`  | `sensor_msgs::msg::CameraInfo` | input front_right camera parameters |
| `~/input/topic_img_back_left/camera_info`    | `sensor_msgs::msg::CameraInfo` | input back_left camera parameters   |
| `~/input/topic_img_back/camera_info`         | `sensor_msgs::msg::CameraInfo` | input back camera parameters        |
| `~/input/topic_img_back_right/camera_info`   | `sensor_msgs::msg::CameraInfo` | input back_right camera parameters  |
| `~/input/scene_token`                        | `autoware_custom_msgs::msg::SceneInfo`        | NuScenes scene token                |
| `~/input/can_bus`                            | `autoware_custom_msgs::msg::CanBusData` | CAN bus data for ego-motion        |

### Outputs

| Name              | Type                                             | Description                                 |
| ----------------- | ------------------------------------------------ | ------------------------------------------- |
| `~/output/boxes`  | `autoware_perception_msgs::msg::DetectedObjects` | detected objects                            |
| `~/output_bboxes` | `visualization_msgs::msg::MarkerArray`           | detected objects for nuScenes visualization |

## How to Use Tensorrt BEVFormer Node

### Prerequisites

- **TensorRT** 10.8.0.43
- **CUDA** 12.4
- **cuDNN** 8.9.2

### Trained Models

Download the [`bevformer_small.onnx`](https://drive.google.com/file/d/1qHyfHnP3sveT3cJ8XHjfVL0UQHcG5zqg/view?usp=sharing) trained model to:

```bash
$HOME/autoware_data/tensorrt_bevformer
```

- The **BEVFormer** model was trained on the **NuScenes** dataset for 24 epochs with temporal fusion enabled.  
  - **Results:**  
    - NDS: 0.478  
    - mAP: 0.370  

### Test TensorRT BEVFormer Node with NuScenes

1. Integrate this package into your **autoware_universe/perception** directory.

2. To play ROS 2 bag of NuScenes data:

   ```bash
   cd autoware/src
   git clone -b feature/bevformer-integration https://github.com/naveen-mcw/ros2_dataset_bridge.git
   cd ..
   ```

   > **Note:**  
   > The `feature/bevformer-integration` branch provides required data for the BEVFormer.  

   Open and edit the launch file to set dataset paths/configs:

   ```bash
   nano src/ros2_dataset_bridge/launch/nuscenes_launch.xml
   ```

   Update as needed:

   ```xml
   <arg name="NUSCENES_DIR" default="<nuscenes_dataset_path>"/>
   <arg name="NUSCENES_CAN_BUS_DIR" default="<can_bus_path>"/>
   <arg name="NUSCENES_VER" default="v1.0-trainval"/>
   <arg name="UPDATE_FREQUENCY" default="10.0"/>
   ```

3. Build Autoware:

   ```bash
   colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
   ```

   Source environments:

   ```bash
   source install/setup.bash
   source /opt/ros/humble/setup.bash
   ```

4. Launch dataset publisher and GUI tools:

   ```bash
   ros2 launch ros2_dataset_bridge nuscenes_launch.xml
   ```

   > 🧠 **Tip:** If NuScenes boxes aren't visible in RViz, uncheck **Stop** in the GUI controller, then click **OK**.
   > ⚠️ **Note:** ROS bag playback is limited to **10 Hz**, constraining BEVFormer node to the same. BEVFormer achieves up to **5 FPS (FP16)** on RTX 2080.

5. Launch TensorRT BEVFormer Node

```bash
# 1. Default mode (FP16)
ros2 launch autoware_tensorrt_bevformer tensorrt_bevformer.launch.xml

# 2. FP32 precision mode
ros2 launch autoware_tensorrt_bevformer tensorrt_bevformer.launch.xml precision:=fp32

# 3. Default mode (FP16) with visualization support
ros2 launch autoware_tensorrt_bevformer tensorrt_bevformer.launch.xml debug_mode:=true

# 4. FP32 precision mode with visualization support
ros2 launch autoware_tensorrt_bevformer tensorrt_bevformer.launch.xml precision:=fp32 debug_mode:=true
```

### Configuration

The configuration file in `config/bevformer.param.yaml` can be modified to suit your needs:

- Modify `precision` to `fp16` or `fp32`
- Set `debug_mode` to `true` to enable publishing bounding box markers.

## Limitation

The model is trained on the open-source dataset **NuScenes** and may have poor generalization on your own dataset. If you want to use this model for your data, you need to retrain it.

## Training BEVFormer Model

If you want to train a model using the [TIER IV's internal database (~2600 key frames)](https://drive.google.com/file/d/1UaarK88HZu09sf7Ix-bEVl9zGNGFwTVL/view?usp=sharing), please refer to: [BEVFormer adapted to TIER IV dataset](https://github.com/cyn-liu/BEVDet/tree/train_export).

## References/External links

[1] [BEVFormer (arXiv)](https://arxiv.org/abs/2203.17270)  
[2] [Original Python BEVFormer TensorRT](https://github.com/DerryHub/BEVFormer_tensorrt.git)  
[3] [NuScenes Dataset](https://www.nuscenes.org/)  

---
