# TensorRT BEVFormer for Autoware

## Prerequisites

- **TensorRT** 10.8.0.43  
- **CUDA** 12.4  
- **cuDNN** 8.9.2  

---

## Trained Models

Download the [`bevformer_small.onnx`](https://drive.google.com/file/d/1qHyfHnP3sveT3cJ8XHjfVL0UQHcG5zqg/view?usp=sharing) trained model to:

```bash
$HOME/autoware_data/tensorrt_bevformer
```

- The **BEVFormer** model was trained on the **NuScenes** dataset for 24 epochs with temporal fusion enabled.  
  - **Results:**  
    - NDS: 0.478  
    - mAP: 0.370  

---

## Test TensorRT BEVFormer Node with NuScenes

### 1. Integrate BEVFormer Node into Autoware

Add this branch to your `autoware_universe/perception` directory.

---

### 2. Play ROS 2 Bag for NuScenes

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

Build Autoware:

```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Source environments:

```bash
source install/setup.bash
source /opt/ros/humble/setup.bash
```

Launch tools:

```bash
ros2 launch ros2_dataset_bridge nuscenes_launch.xml
```

> 🧠 **Tip:** If NuScenes boxes aren't visible in RViz, uncheck **Stop** in the GUI controller, then click **OK**.
> ⚠️ **Note:** ROS bag playback is limited to **10 Hz**, constraining BEVFormer node to the same. BEVFormer achieves up to **5 FPS (FP16)** in RTX 2080.

---

### 4️⃣ Launch TensorRT BEVFormer Node

```bash
# Default (FP16) with visuliazation support
ros2 launch autoware_tensorrt_bevformer tensorrt_bevformer.launch.xml debug_mode:=true

# FP32 precision mode with visualization support
ros2 launch autoware_tensorrt_bevformer tensorrt_bevformer.launch.xml precision:=fp32 debug_mode:=true
```

---

## ✨ Acknowledgements

- [BEVFormer: Incorporating Transformers for Multi-Camera 3D Detection](https://arxiv.org/abs/2203.17270)
- TensorRT acceleration by [NVIDIA](https://developer.nvidia.com/tensorrt)
- [NuScenes Dataset](https://www.nuscenes.org/)
- **Ported from Python implementation:** [DerryHub/BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt.git)

---
