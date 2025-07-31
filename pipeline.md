官方推荐使用docker，这里我给出不使用docker直接部署的方案。
git clone
conda env create -f environment.yml
conda run -n foundation_stereo pip install flash-attn
conda activate foundation_stereo

python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/

#tensorrt实时推理
pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt --extra-index-url https://pypi.nvidia.com
pip install pycuda

#
# 1) 添加 CUDA 仓库 keyring（有助于依赖解析）
wget -qO /tmp/cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/cuda-keyring.deb

# 2) 添加 NVIDIA Machine Learning 仓库 keyring（TensorRT 所在仓库）
wget -qO /tmp/nvidia-ml-keyring.deb \
  https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2204/x86_64/nvidia-machine-learning-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/nvidia-ml-keyring.deb

# 3) 更新索引并检查是否能搜到 nvinfer 包
sudo apt-get update
apt-cache search -n 'nvinfer|tensorrt' | sort

export PATH=/usr/src/tensorrt/bin:$PATH
# Make ONNX:
XFORMERS_DISABLED=1 python scripts/make_onnx.py --save_path ./pretrained_models/foundation_stereo.onnx --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --height 448 --width 672 --valid_iters 20
#  - --height 448：输入图像高度
#  - --width 672：输入图像宽度
#  - --valid_iters 20：推理时的迭代次数
#    - 控制精度与速度的平衡
#    - 迭代次数越多，视差估计越精确
#    - 默认值是 16，这里设为 20 以提高精度


# Convert to TRT
trtexec --onnx=pretrained_models/foundation_stereo.onnx --verbose --saveEngine=pretrained_models/foundation_stereo.plan --fp16

# Run TRT:
python scripts/run_demo_tensorrt.py \
        --left_img ${PWD}/assets/left.png \
        --right_img ${PWD}/assets/right.png \
        --save_path ${PWD}/output \
        --pretrained pretrained_models/foundation_stereo.plan \
        --height 448 \ #默认 需要能整除以32 (2的5次方 用于ffn上采样)
        --width 672 \  # 默认
        --pc \
        --z_far 100.0
        #点云深度裁剪阈值（默认：100米），只保留深度小于此值的点

realsense实时推理：
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')

import argparse
import cv2
import numpy as np
import os
import sys
import torch
import time
import pyrealsense2 as rs
from onnx_tensorrt import tensorrt_engine
import tensorrt as trt
import onnxruntime as ort

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from Utils import *


def init_realsense(width=640, height=480, fps=30):
    """Initialize RealSense camera and configure IR streams"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable left and right IR streams
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)  # Left IR
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)  # Right IR
    
    # Start streaming
    pipeline_profile = pipeline.start(config)
    
    # Get camera intrinsics
    profile = pipeline_profile.get_stream(rs.stream.infrared, 1)
    intrinsics = profile.as_video_stream_profile().get_intrinsics()
    
    # Get baseline (distance between cameras)
    device = pipeline_profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    # For D435/D415, baseline is approximately 50mm
    # You can also get it from extrinsics if needed
    baseline = 0.050  # 50mm in meters
    
    return pipeline, intrinsics, baseline


def get_onnx_model(args):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = ort.InferenceSession(args.pretrained, sess_options=session_options, providers=['CUDAExecutionProvider'])
    return model


def get_engine_model(args):
    with open(args.pretrained, 'rb') as file:
        engine_data = file.read()
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_data)
    engine = tensorrt_engine.Engine(engine)
    return engine


def preprocess_ir_frame(frame, target_height, target_width):
    """Preprocess IR frame for model input"""
    # Convert single channel IR to 3-channel for model compatibility
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    # Resize if needed
    if frame_rgb.shape[:2] != (target_height, target_width):
        frame_rgb = cv2.resize(frame_rgb, (target_width, target_height))
    
    # Convert to tensor format
    frame_tensor = torch.as_tensor(frame_rgb.copy()).float()[None].permute(0,3,1,2).contiguous()
    return frame_tensor, frame_rgb


def run_realtime_inference(model, pipeline, intrinsics, baseline, args):
    """Main loop for real-time inference"""
    cv2.namedWindow('RealSense IR Left', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('RealSense IR Right', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Disparity Map', cv2.WINDOW_AUTOSIZE)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, 'realsense')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if camera resolution matches model input
    camera_width = intrinsics.width
    camera_height = intrinsics.height
    model_width = args.width
    model_height = args.height
    
    needs_resize = (camera_width != model_width) or (camera_height != model_height)
    
    if needs_resize:
        print(f"Camera resolution ({camera_width}x{camera_height}) differs from model input ({model_width}x{model_height})")
        print("Will resize images and adjust intrinsics accordingly.")
        
        # Calculate scale factors
        scale_x = model_width / camera_width
        scale_y = model_height / camera_height
        
        # Create adjusted intrinsics for model resolution
        model_intrinsics = {
            'fx': intrinsics.fx * scale_x,
            'fy': intrinsics.fy * scale_y,
            'cx': intrinsics.ppx * scale_x,
            'cy': intrinsics.ppy * scale_y,
            'width': model_width,
            'height': model_height
        }
        
        print(f"Adjusted intrinsics for model:")
        print(f"  fx={model_intrinsics['fx']:.2f}, fy={model_intrinsics['fy']:.2f}")
        print(f"  cx={model_intrinsics['cx']:.2f}, cy={model_intrinsics['cy']:.2f}")
    else:
        print(f"Camera resolution matches model input: {camera_width}x{camera_height}")
        model_intrinsics = {
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'cx': intrinsics.ppx,
            'cy': intrinsics.ppy,
            'width': camera_width,
            'height': camera_height
        }
    
    fps_counter = 0
    fps_start_time = time.time()
    fps_display = 0
    
    try:
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            
            # Get left and right IR frames
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            
            if not left_frame or not right_frame:
                continue
            
            # Convert to numpy arrays
            left_image = np.asanyarray(left_frame.get_data())
            right_image = np.asanyarray(right_frame.get_data())
            
            # Preprocess for model
            left_tensor, left_rgb = preprocess_ir_frame(left_image, args.height, args.width)
            right_tensor, right_rgb = preprocess_ir_frame(right_image, args.height, args.width)
            
            # Run inference
            inference_start = time.time()
            
            if args.pretrained.endswith('.onnx'):
                left_disp = model.run(None, {'left': left_tensor.numpy(), 'right': right_tensor.numpy()})[0]
            else:
                left_disp = model.run([left_tensor.numpy(), right_tensor.numpy()])[0]
            
            inference_time = time.time() - inference_start
            
            # Process disparity
            left_disp = left_disp.squeeze()
            
            # Calculate depth from disparity
            # depth = focal_length * baseline / disparity
            # Use the model-resolution intrinsics for depth calculation
            focal_length = model_intrinsics['fx']  # Use fx from model resolution
            depth = focal_length * baseline / (left_disp + 1e-6)  # Add small epsilon to avoid division by zero
            
            # Visualize disparity
            disp_vis = vis_disparity(left_disp)
            
            # Optional: Create depth visualization
            depth_vis = vis_disparity(depth, min_val=0.3, max_val=5.0)  # Visualize 0.3m to 5m range
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 10 == 0:
                fps_end_time = time.time()
                fps_display = 10 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
            
            # Add FPS and inference time text
            cv2.putText(disp_vis, f'FPS: {fps_display:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(disp_vis, f'Inference: {inference_time*1000:.1f}ms', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display images
            cv2.imshow('RealSense IR Left', left_image)
            cv2.imshow('RealSense IR Right', right_image)
            cv2.imshow('Disparity Map', disp_vis)
            
            # Check for exit
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break
            elif key == ord('s'):  # 's' to save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(output_dir, f'left_ir_{timestamp}.png'), left_image)
                cv2.imwrite(os.path.join(output_dir, f'right_ir_{timestamp}.png'), right_image)
                cv2.imwrite(os.path.join(output_dir, f'disparity_{timestamp}.png'), disp_vis)
                
                # Save depth as numpy array for later use
                np.save(os.path.join(output_dir, f'depth_{timestamp}.npy'), depth)
                
                # Save both original and model camera parameters
                cam_params = {
                    'original_fx': intrinsics.fx, 'original_fy': intrinsics.fy,
                    'original_cx': intrinsics.ppx, 'original_cy': intrinsics.ppy,
                    'original_width': intrinsics.width, 'original_height': intrinsics.height,
                    'model_fx': model_intrinsics['fx'], 'model_fy': model_intrinsics['fy'],
                    'model_cx': model_intrinsics['cx'], 'model_cy': model_intrinsics['cy'],
                    'model_width': model_intrinsics['width'], 'model_height': model_intrinsics['height'],
                    'baseline': baseline
                }
                with open(os.path.join(output_dir, f'camera_params_{timestamp}.txt'), 'w') as f:
                    f.write("# Original camera parameters\n")
                    f.write(f"original_resolution: {intrinsics.width}x{intrinsics.height}\n")
                    f.write(f"original_fx: {intrinsics.fx}\n")
                    f.write(f"original_fy: {intrinsics.fy}\n")
                    f.write(f"original_cx: {intrinsics.ppx}\n")
                    f.write(f"original_cy: {intrinsics.ppy}\n")
                    f.write("\n# Model input parameters\n")
                    f.write(f"model_resolution: {model_intrinsics['width']}x{model_intrinsics['height']}\n")
                    f.write(f"model_fx: {model_intrinsics['fx']}\n")
                    f.write(f"model_fy: {model_intrinsics['fy']}\n")
                    f.write(f"model_cx: {model_intrinsics['cx']}\n")
                    f.write(f"model_cy: {model_intrinsics['cy']}\n")
                    f.write(f"\n# Stereo baseline\n")
                    f.write(f"baseline_meters: {baseline}\n")
                
                print(f"Saved frames and depth to {output_dir} at {timestamp}")
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='RealSense Stereo TensorRT Demo')
    code_dir = os.path.dirname(os.path.realpath(__file__))
    
    parser.add_argument('--pretrained', default='pretrained_models/foundation_stereo.plan', 
                       help='Path to pretrained model (.onnx, .engine, or .plan)')
    parser.add_argument('--height', type=int, default=480, help='Model input height')
    parser.add_argument('--width', type=int, default=640, help='Model input width')
    parser.add_argument('--camera_width', type=int, default=640, help='Camera capture width')
    parser.add_argument('--camera_height', type=int, default=480, help='Camera capture height')
    parser.add_argument('--fps', type=int, default=30, help='Camera FPS')
    parser.add_argument('--output_dir', default=f'{code_dir}/../output', help='Output directory for saved frames')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Check model exists
    assert os.path.isfile(args.pretrained), f'Pretrained model {args.pretrained} not found'
    print(f'Loading model from {args.pretrained}')
    
    # Load model
    set_seed(0)
    if args.pretrained.endswith('.onnx'):
        model = get_onnx_model(args)
    elif args.pretrained.endswith('.engine') or args.pretrained.endswith('.plan'):
        model = get_engine_model(args)
    else:
        assert False, f'Unknown model format {args.pretrained}'
    
    # Initialize RealSense
    print("Initializing RealSense camera...")
    pipeline, intrinsics, baseline = init_realsense(args.camera_width, args.camera_height, args.fps)
    
    # Print camera parameters
    print(f"Camera intrinsics:")
    print(f"  Resolution: {intrinsics.width}x{intrinsics.height}")
    print(f"  Focal length: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
    print(f"  Principal point: cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
    print(f"  Baseline: {baseline*1000:.1f}mm")
    
    # Warm up
    print("Warming up...")
    for _ in range(5):
        frames = pipeline.wait_for_frames()
    
    print("Starting real-time inference. Press 'q' or ESC to quit, 's' to save frames.")
    
    # Run real-time inference
    run_realtime_inference(model, pipeline, intrinsics, baseline, args)


if __name__ == '__main__':
    main()
python scripts/run_demo_tensorrt_rs.py \
  --pretrained pretrained_models/foundation_stereo.plan \
  --height 448 \
  --width 672 \
  --fps 30
      
      
      
  python scripts/run_demo_tensorrt_rs.py \
      --pretrained output/foundation_stereo_640x480.plan \
      --height 480 \
      --width 640 \
      --fps 30

# torch 版本（没加速）
python scripts/run_demo_rs.py --ckpt_dir \
  pretrained_models/23-51-11/model_best_bp2.pth
# 2-3hz

修改默认分辨率
默认的是448*672 和realsense常用的分辨率不匹配.   简单方法是裁减，裁减后再修改内参(速度快)。
或者：
1. 生成新尺寸的 ONNX 模型
    为什么必须是 32 的倍数？
  - 模型使用了多层下采样（通常是 5 层，2^5=32）
  - 特征图需要能够精确还原到原始尺寸
  - 非 32 倍数会导致上采样时尺寸不匹配
  所以需要新写一个make_onnx_no_xformers.py （旧的编译不过）
XFORMERS_DISABLED=1

python scripts/make_onnx_no_xformers.py \
--height 480 \
--width 640 \
--save_path output/foundation_stereo_640x480.onnx \
--ckpt_dir pretrained_models/23-51-11/model_best_bp2.pth \
--valid_iters 20

   
  
  
2. 转换为 TensorRT 引擎
  
  使用 trtexec 工具（根据 readme.md 第116行）：
  编写能编译通过的版本。
import warnings, argparse, logging, os, sys
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')

# Force disable xformers before any imports
os.environ['XFORMERS_DISABLED'] = '1'
os.environ['XFORMERS_AVAILABLE'] = '0'

# Mock xformers to prevent its usage
import unittest.mock as mock
sys.modules['xformers'] = mock.MagicMock()
sys.modules['xformers.ops'] = mock.MagicMock()
sys.modules['xformers.ops.fmha'] = mock.MagicMock()

import omegaconf, yaml, torch, pdb
from omegaconf import OmegaConf
from core.foundation_stereo import FoundationStereo


class FoundationStereoOnnx(FoundationStereo):
    def __init__(self, args):
        super().__init__(args)

    @torch.no_grad()
    def forward(self, left, right):
        """ Removes extra outputs and hyper-parameters """
        with torch.amp.autocast('cuda', enabled=True):
            disp = FoundationStereo.forward(self, left, right, iters=self.args.valid_iters, test_mode=True)
        return disp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=f'{code_dir}/../output/foundation_stereo.onnx', help='Path to save results.')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--height', type=int, default=448)
    parser.add_argument('--width', type=int, default=672)
    parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    torch.autograd.set_grad_enabled(False)

    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    for k in args.__dict__:
      cfg[k] = args.__dict__[k]
    if 'vit_size' not in cfg:
      cfg['vit_size'] = 'vitl'
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")
    model = FoundationStereoOnnx(cfg)
    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()


    left_img = torch.randn(1, 3, args.height, args.width).cuda().float()
    right_img = torch.randn(1, 3, args.height, args.width).cuda().float()

    torch.onnx.export(
        model,
        (left_img, right_img),
        args.save_path,
        opset_version=16,
        input_names = ['left', 'right'],
        output_names = ['disp'],
        dynamic_axes={
            'left': {0 : 'batch_size'},
            'right': {0 : 'batch_size'},
            'disp': {0 : 'batch_size'}
        },
    )
export PATH=/usr/src/tensorrt/bin:$PATH  
trtexec --onnx=output/foundation_stereo_640x480.onnx \
   --verbose \
   --saveEngine=output/foundation_stereo_640x480.plan \
   --fp16
  


