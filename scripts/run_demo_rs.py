# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

import argparse
import cv2
import numpy as np
import torch
import time
import logging
import pyrealsense2 as rs
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *


def init_realsense(width=640, height=480, fps=30):
    """Initialize RealSense camera and configure IR streams
    
    Args:
        width: Camera capture width
        height: Camera capture height  
        fps: Camera FPS
        
    Returns:
        pipeline: RealSense pipeline object
        intrinsics: Camera intrinsics
        baseline: Stereo baseline in meters
    """
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
    # For D435/D415, baseline is approximately 50mm
    baseline = 0.050  # 50mm in meters
    
    return pipeline, intrinsics, baseline


def load_model(args):
    """Load FoundationStereo model from checkpoint
    
    Args:
        args: Arguments containing ckpt_dir path
        
    Returns:
        model: Loaded FoundationStereo model on GPU
    """
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    
    # Merge args into config
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    
    args_merged = OmegaConf.create(cfg)
    logging.info(f"Loading model from {ckpt_dir}")
    
    # Create model
    model = FoundationStereo(args_merged)
    
    # Load checkpoint
    ckpt = torch.load(ckpt_dir)
    logging.info(f"Checkpoint - global_step: {ckpt['global_step']}, epoch: {ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])
    
    model.cuda()
    model.eval()
    
    return model, args_merged


def preprocess_frames(left_frame, right_frame, padder=None):
    """Preprocess IR frames for model input
    
    Args:
        left_frame: Left IR frame (H,W) numpy array
        right_frame: Right IR frame (H,W) numpy array
        padder: InputPadder instance for padding
        
    Returns:
        left_tensor: Preprocessed left tensor
        right_tensor: Preprocessed right tensor
        left_rgb: Left frame as RGB for visualization
        right_rgb: Right frame as RGB for visualization
    """
    # Convert single channel IR to 3-channel for model compatibility
    left_rgb = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2RGB)
    right_rgb = cv2.cvtColor(right_frame, cv2.COLOR_GRAY2RGB)
    
    # Convert to tensor format
    left_tensor = torch.as_tensor(left_rgb).cuda().float()[None].permute(0,3,1,2)
    right_tensor = torch.as_tensor(right_rgb).cuda().float()[None].permute(0,3,1,2)
    
    # Pad if padder provided
    if padder is not None:
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
    
    return left_tensor, right_tensor, left_rgb, right_rgb


def run_realtime_inference(model, pipeline, intrinsics, baseline, args):
    """Main loop for real-time inference
    
    Args:
        model: FoundationStereo model
        pipeline: RealSense pipeline
        intrinsics: Camera intrinsics
        baseline: Stereo baseline
        args: Command line arguments
    """
    # Create windows
    cv2.namedWindow('RealSense IR Stereo', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Disparity Map', cv2.WINDOW_AUTOSIZE)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    output_dir = os.path.join(args.out_dir, 'realsense')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup padding
    H, W = intrinsics.height, intrinsics.width
    padder = InputPadder((1, 3, H, W), divis_by=32, force_square=False)
    
    # Build camera matrix
    K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                  [0, intrinsics.fy, intrinsics.ppy],
                  [0, 0, 1]], dtype=np.float32)
    
    # FPS tracking
    fps_counter = 0
    fps_start_time = time.time()
    fps_display = 0
    inference_times = []
    inference_hz = 0
    
    logging.info("Starting real-time inference. Press 'q' or ESC to quit, 's' to save frames.")
    logging.info(f"Camera resolution: {W}x{H}")
    logging.info(f"Camera intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
    logging.info(f"Baseline: {baseline*1000:.1f}mm")
    
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
            left_tensor, right_tensor, left_rgb, right_rgb = preprocess_frames(
                left_image, right_image, padder
            )
            
            # Run inference
            inference_start = time.time()
            
            with torch.cuda.amp.autocast(True):
                with torch.no_grad():
                    if not args.hiera:
                        disp = model.forward(left_tensor, right_tensor, 
                                           iters=args.valid_iters, test_mode=True)
                    else:
                        disp = model.run_hierachical(left_tensor, right_tensor, 
                                                    iters=args.valid_iters, test_mode=True, 
                                                    small_ratio=0.5)
            
            disp = padder.unpad(disp.float())
            disp = disp.data.cpu().numpy().reshape(H, W)
            
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            # Remove invisible pixels if requested
            if args.remove_invisible:
                yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
                us_right = xx - disp
                invalid = us_right < 0
                disp[invalid] = np.inf
            
            # Calculate depth from disparity
            depth = K[0,0] * baseline / (disp + 1e-6)
            
            # Visualize disparity
            disp_vis = vis_disparity(disp)
            
            # Create stereo pair visualization
            stereo_vis = np.hstack([left_image, right_image])
            stereo_vis = cv2.cvtColor(stereo_vis, cv2.COLOR_GRAY2RGB)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 10 == 0:
                fps_end_time = time.time()
                fps_display = 10 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                
                # Calculate average inference time
                avg_inference = np.mean(inference_times[-10:]) if inference_times else 0
                inference_hz = 1.0 / avg_inference if avg_inference > 0 else 0
            
            # Add FPS and inference time text
            cv2.putText(disp_vis, f'FPS: {fps_display:.1f} Hz', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(disp_vis, f'Inference: {inference_time*1000:.1f}ms ({inference_hz:.1f} Hz)', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display images
            cv2.imshow('RealSense IR Stereo', stereo_vis)
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
                np.save(os.path.join(output_dir, f'disparity_{timestamp}.npy'), disp)
                np.save(os.path.join(output_dir, f'depth_meter_{timestamp}.npy'), depth)
                
                # Save camera parameters
                with open(os.path.join(output_dir, f'K_{timestamp}.txt'), 'w') as f:
                    for row in K:
                        f.write(' '.join(map(str, row)) + '\n')
                    f.write(str(baseline))
                
                # Save point cloud if requested
                if args.get_pc:
                    xyz_map = depth2xyzmap(depth, K)
                    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), left_rgb.reshape(-1,3))
                    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
                    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
                    pcd = pcd.select_by_index(keep_ids)
                    o3d.io.write_point_cloud(os.path.join(output_dir, f'cloud_{timestamp}.ply'), pcd)
                
                logging.info(f"Saved frames to {output_dir} at {timestamp}")
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        
        # Print statistics
        if inference_times:
            avg_inference = np.mean(inference_times)
            logging.info(f"\nInference statistics:")
            logging.info(f"  Average inference time: {avg_inference*1000:.1f}ms")
            logging.info(f"  Average inference Hz: {1.0/avg_inference:.1f} Hz")
            logging.info(f"  Min inference time: {np.min(inference_times)*1000:.1f}ms")
            logging.info(f"  Max inference time: {np.max(inference_times)*1000:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description='RealSense Stereo Real-time Demo')
    
    # Model parameters
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', 
                       type=str, help='pretrained model path')
    
    # Camera parameters
    parser.add_argument('--width', type=int, default=640, help='Camera capture width')
    parser.add_argument('--height', type=int, default=480, help='Camera capture height')
    parser.add_argument('--fps', type=int, default=30, help='Camera FPS')
    
    # Inference parameters
    parser.add_argument('--valid_iters', type=int, default=32, 
                       help='number of flow-field updates during forward pass')
    parser.add_argument('--hiera', default=0, type=int, 
                       help='hierarchical inference (only needed for high-resolution images (>1K))')
    
    # Output parameters
    parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, 
                       help='the directory to save results')
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--remove_invisible', default=1, type=int, 
                       help='remove non-overlapping observations between left and right images')
    
    args = parser.parse_args()
    
    # Setup logging
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    
    # Check model exists
    assert os.path.isfile(args.ckpt_dir), f'Checkpoint {args.ckpt_dir} not found'
    
    # Load model
    model, full_args = load_model(args)
    
    # Initialize RealSense
    logging.info("Initializing RealSense camera...")
    try:
        pipeline, intrinsics, baseline = init_realsense(args.width, args.height, args.fps)
    except Exception as e:
        logging.error(f"Failed to initialize RealSense camera: {e}")
        logging.error("Please make sure a RealSense camera is connected.")
        return
    
    # Warm up
    logging.info("Warming up...")
    for _ in range(5):
        frames = pipeline.wait_for_frames()
    
    # Run real-time inference
    run_realtime_inference(model, pipeline, intrinsics, baseline, full_args)


if __name__ == '__main__':
    main()