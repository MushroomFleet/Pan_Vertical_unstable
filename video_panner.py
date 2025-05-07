#!/usr/bin/env python3
"""
Video Panning Tool - Convert landscape videos to portrait with panning effects.

This script transforms 1280x720 landscape videos into 720x1280 portrait videos
with a configurable panning effect that moves from left to right through
user-defined markpoints.

This version supports CUDA-acceleration for improved processing speed.

Usage:
    python video_panner.py --config path/to/config.json
"""

import os
import sys
import json
import argparse
import time
import subprocess
import threading
import queue

# Pipeline processing constants
INPUT_QUEUE_SIZE = 60  # Frames waiting to be processed
OUTPUT_QUEUE_SIZE = 30  # Frames waiting to be written

class FrameReaderThread(threading.Thread):
    """Thread that reads frames from a video file and places them in a queue."""
    def __init__(self, video_path, frame_queue, total_frames=None, x_offsets=None, verbose=False):
        threading.Thread.__init__(self)
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.total_frames = total_frames
        self.x_offsets = x_offsets
        self.verbose = verbose
        self.stopped = False
        # Will hold video properties like fps, dimensions
        self.video_properties = {}
        
    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file at {self.video_path}")
                return
                
            # Extract video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.video_properties = {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames
            }
            
            if self.verbose:
                print(f"FrameReaderThread: Video properties - {width}x{height}, {fps} fps, {total_frames} frames")
            
            frame_count = 0
            start_time = time.time()
            
            while not self.stopped:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Put frame and its metadata in queue
                item = {
                    'frame': frame,
                    'frame_idx': frame_count,
                    'x_offset': self.x_offsets[frame_count] if self.x_offsets is not None else 0
                }
                self.frame_queue.put(item)
                
                frame_count += 1
                
                # Show progress occasionally
                if self.verbose and frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_read = frame_count / max(0.1, elapsed)
                    print(f"Reading: {frame_count}/{total_frames} frames - {fps_read:.2f} fps")
                
            # Signal end of frames
            self.frame_queue.put(None)
            
            if self.verbose:
                elapsed = time.time() - start_time
                print(f"FrameReaderThread: Completed reading {frame_count} frames in {elapsed:.2f}s")
                
            cap.release()
            
        except Exception as e:
            print(f"Error in FrameReaderThread: {str(e)}")
            # Signal error to other threads
            self.frame_queue.put(None)
            
    def stop(self):
        self.stopped = True


class FrameProcessorThread(threading.Thread):
    """Thread that processes frames from the input queue and places them in the output queue."""
    def __init__(self, frame_queue, output_queue, y_offset, overlay_data=None, verbose=False):
        threading.Thread.__init__(self)
        self.frame_queue = frame_queue
        self.output_queue = output_queue
        self.y_offset = y_offset
        self.overlay_data = overlay_data
        self.verbose = verbose
        self.stopped = False
        self.frames_processed = 0
        
    def run(self):
        try:
            start_time = time.time()
            
            # Create portrait frame template
            portrait_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
            
            while not self.stopped:
                # Get item from input queue
                item = self.frame_queue.get()
                
                # Check for end signal
                if item is None:
                    # Pass the end signal to the output queue
                    self.output_queue.put(None)
                    break
                    
                frame = item['frame']
                frame_idx = item['frame_idx']
                x_offset = item['x_offset']
                
                # Process the frame
                processed_frame = self._process_frame(frame, x_offset, portrait_frame.copy())
                
                # Put processed frame in output queue
                self.output_queue.put({
                    'frame': processed_frame,
                    'frame_idx': frame_idx
                })
                
                # Signal queue item is processed
                self.frame_queue.task_done()
                
                self.frames_processed += 1
                
                # Show occasional progress if verbose
                if self.verbose and self.frames_processed % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_proc = self.frames_processed / max(0.1, elapsed)
                    print(f"Processing: {self.frames_processed} frames at {fps_proc:.2f} fps")
            
            if self.verbose:
                elapsed = time.time() - start_time
                print(f"FrameProcessorThread: Processed {self.frames_processed} frames in {elapsed:.2f}s")
                
        except Exception as e:
            print(f"Error in FrameProcessorThread: {str(e)}")
            # Signal error to other threads
            self.output_queue.put(None)
            
    def _process_frame(self, frame, x_offset, portrait_frame):
        """Process a single frame to apply cropping, portrait conversion, and overlay."""
        try:
            # Crop the frame to get the 720 wide section - use GPU if available
            if CUDA_AVAILABLE:
                # Transfer frame to GPU
                gpu_frame = to_gpu(frame)
                
                # Crop on GPU
                gpu_cropped = gpu_crop(gpu_frame, x_offset, 0, OUTPUT_WIDTH, frame.shape[0])
                
                # Transfer back to CPU
                cropped_frame = to_cpu(gpu_cropped)
            else:
                # CPU fallback
                cropped_frame = frame[:, x_offset:x_offset+OUTPUT_WIDTH, :]
            
            # Place cropped frame in the middle of the portrait frame vertically
            portrait_frame[self.y_offset:self.y_offset+frame.shape[0], :, :] = cropped_frame
            
            # Add overlay if available
            if self.overlay_data is not None:
                resized_overlay, placements = self.overlay_data
                y_pos, x_pos, h, w = placements
                
                # Verify overlay bounds are within the portrait frame
                if y_pos < 0 or x_pos < 0 or y_pos + h > OUTPUT_HEIGHT or x_pos + w > OUTPUT_WIDTH:
                    # Adjust placement to ensure it's within bounds
                    y_pos = max(0, min(y_pos, OUTPUT_HEIGHT - h))
                    x_pos = max(0, min(x_pos, OUTPUT_WIDTH - w))
                
                # Apply overlay based on whether it has an alpha channel
                if resized_overlay.shape[2] == 4:  # With alpha channel
                    # Get the overlay area in the portrait frame
                    overlay_area = portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w]
                    
                    if CUDA_AVAILABLE:
                        try:
                            # GPU-accelerated alpha blending
                            gpu_overlay = cp.asarray(resized_overlay)
                            gpu_area = cp.asarray(overlay_area)
                            
                            # Alpha blending formula on GPU
                            alpha_factor = gpu_overlay[:,:,3].astype(cp.float32) / 255.0
                            
                            # Create the result array on GPU
                            blended_gpu = cp.zeros_like(gpu_area)
                            
                            # Apply blending for each channel
                            for c in range(3):
                                blended_gpu[:,:,c] = (gpu_area[:,:,c] * (1 - alpha_factor) + 
                                                  gpu_overlay[:,:,c] * alpha_factor).astype(cp.uint8)
                            
                            # Transfer back to CPU
                            blended_area = cp.asnumpy(blended_gpu)
                            
                            # Copy the blended result back to the portrait frame
                            portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w] = blended_area
                        except Exception:
                            # CPU fallback implementation
                            blended_area = overlay_area.copy()
                            alpha_factor = resized_overlay[:,:,3].astype(float) / 255.0
                            for c in range(3):
                                blended_area[:,:,c] = (overlay_area[:,:,c] * (1 - alpha_factor) + 
                                                   resized_overlay[:,:,c] * alpha_factor).astype(np.uint8)
                            portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w] = blended_area
                    else:
                        # CPU implementation
                        blended_area = overlay_area.copy()
                        alpha_factor = resized_overlay[:,:,3].astype(float) / 255.0
                        for c in range(3):
                            blended_area[:,:,c] = (overlay_area[:,:,c] * (1 - alpha_factor) + 
                                               resized_overlay[:,:,c] * alpha_factor).astype(np.uint8)
                        portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w] = blended_area
                else:
                    # For non-transparent overlays
                    if CUDA_AVAILABLE:
                        try:
                            # Use CUDA-accelerated addWeighted
                            gpu_overlay_area = to_gpu(portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w])
                            gpu_overlay_img = to_gpu(resized_overlay)
                            
                            # Perform blending on GPU
                            gpu_result = cuda.addWeighted(gpu_overlay_area, 0.7, gpu_overlay_img, 0.3, 0.0)
                            
                            # Transfer result back to CPU and place in the portrait frame
                            portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w] = to_cpu(gpu_result)
                        except Exception:
                            # Fallback to CPU
                            cv2.addWeighted(
                                portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w], 0.7,
                                resized_overlay, 0.3, 0,
                                dst=portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w]
                            )
                    else:
                        # CPU version
                        cv2.addWeighted(
                            portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w], 0.7,
                            resized_overlay, 0.3, 0,
                            dst=portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w]
                        )
            
            return portrait_frame
            
        except Exception as e:
            print(f"Error in _process_frame: {str(e)}")
            raise
            
    def stop(self):
        self.stopped = True


class FrameWriterThread(threading.Thread):
    """Thread that writes processed frames to an output video file."""
    def __init__(self, output_queue, output_file, fps, verbose=False):
        threading.Thread.__init__(self)
        self.output_queue = output_queue
        self.output_file = output_file
        self.fps = fps
        self.verbose = verbose
        self.stopped = False
        self.frames_written = 0
        
    def run(self):
        try:
            start_time = time.time()
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                self.output_file, 
                fourcc, 
                self.fps, 
                (OUTPUT_WIDTH, OUTPUT_HEIGHT)
            )
            
            if not out.isOpened():
                print(f"Error: Could not create output video writer for {self.output_file}")
                return
            
            while not self.stopped:
                # Get processed frame from queue
                item = self.output_queue.get()
                
                # Check for end signal
                if item is None:
                    break
                    
                # Write the frame
                out.write(item['frame'])
                self.frames_written += 1
                
                # Signal queue item is processed
                self.output_queue.task_done()
                
                # Show occasional progress if verbose
                if self.verbose and self.frames_written % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_write = self.frames_written / max(0.1, elapsed)
                    print(f"Writing: {self.frames_written} frames at {fps_write:.2f} fps")
            
            # Release resources
            out.release()
            
            if self.verbose:
                elapsed = time.time() - start_time
                print(f"FrameWriterThread: Wrote {self.frames_written} frames in {elapsed:.2f}s")
                
        except Exception as e:
            print(f"Error in FrameWriterThread: {str(e)}")
        
    def stop(self):
        self.stopped = True
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Set CUDA environment variables before importing OpenCV
cuda_path = os.environ.get('CUDA_PATH')
if cuda_path:
    # Set OpenCV-specific CUDA environment variables
    os.environ['OPENCV_CUDA_PATH'] = cuda_path
    os.environ['CUDA_HOME'] = cuda_path
    os.environ['OPENCV_CUDA_TOOLKIT_ROOT_DIR'] = cuda_path
    
    # Set paths for various CUDA libraries
    os.environ['OPENCV_CUDA_LIB_PATH'] = os.path.join(cuda_path, 'lib/x64')
    os.environ['OPENCV_CUDA_BIN_PATH'] = os.path.join(cuda_path, 'bin')
    os.environ['OPENCV_CUDA_INCLUDE_PATH'] = os.path.join(cuda_path, 'include')
    
    # Add CUDA bin directory to PATH
    if sys.platform == 'win32':
        cuda_bin_path = os.path.join(cuda_path, 'bin')
        os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ.get('PATH', '')

import cv2
import numpy as np
import ffmpeg

# Import CUDA-specific modules
try:
    import cupy as cp
    import cupyx.scipy.ndimage
    from cv2 import cuda  # Still try to import for compatibility
    
    # Debug CUDA environment
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH')}")
    print(f"OPENCV_CUDA_PATH: {os.environ.get('OPENCV_CUDA_PATH')}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
    
    # Check CuPy CUDA support (this is what we'll primarily use)
    cupy_cuda_available = cp.cuda.is_available()
    print(f"CuPy CUDA available: {cupy_cuda_available}")
    
    # Initialize advanced features flags
    USE_ADVANCED_FEATURES = False
    MAX_RECOMMENDED_BATCH = 1
    TOTAL_VRAM_GB = 0
    
    if cupy_cuda_available:
        print(f"CuPy CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"CuPy CUDA device count: {cp.cuda.runtime.getDeviceCount()}")
        device_props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"CuPy CUDA device name: {device_props['name'].decode('utf-8')}")
        
        # Get VRAM size and set feature flags based on available memory
        TOTAL_VRAM_GB = device_props['totalGlobalMem'] / (1024**3)
        print(f"CuPy CUDA device memory: {TOTAL_VRAM_GB:.2f} GB")
        
        if TOTAL_VRAM_GB >= 12.0:
            USE_ADVANCED_FEATURES = True
            print(f"High-VRAM GPU detected ({TOTAL_VRAM_GB:.2f} GB). Enabling advanced features!")
            # Heuristic: Each 1080p frame is about 8MB, allow overhead
            MAX_RECOMMENDED_BATCH = min(64, int(TOTAL_VRAM_GB * 1.5))
            print(f"Maximum recommended batch size: {MAX_RECOMMENDED_BATCH}")
        else:
            print(f"Standard GPU detected ({TOTAL_VRAM_GB:.2f} GB). Using basic acceleration.")
            MAX_RECOMMENDED_BATCH = min(8, max(1, int(TOTAL_VRAM_GB / 0.5)))
            print(f"Maximum recommended batch size: {MAX_RECOMMENDED_BATCH}")
    
    # Also check OpenCV CUDA support as a fallback
    have_opencv_cuda = hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'getCudaEnabledDeviceCount')
    print(f"OpenCV has CUDA module: {have_opencv_cuda}")
    
    opencv_cuda_available = False
    if have_opencv_cuda:
        device_count = cuda.getCudaEnabledDeviceCount()
        opencv_cuda_available = device_count > 0
        print(f"OpenCV CUDA device count: {device_count}")
        
        if opencv_cuda_available:
            print(f"OpenCV CUDA is available. Found {device_count} CUDA device(s).")
            device_info = cuda.getDevice()
            print(f"Using OpenCV CUDA device: {device_info}")
            
            # Try to get more device info
            try:
                device_name = cuda.getDeviceName(device_info)
                print(f"OpenCV CUDA device name: {device_name}")
            except Exception as e:
                print(f"Could not get OpenCV CUDA device name: {str(e)}")
    
    # Use CUDA if either CuPy or OpenCV can use it, but prefer CuPy
    CUDA_AVAILABLE = cupy_cuda_available
    if CUDA_AVAILABLE:
        print("CUDA acceleration enabled via CuPy!")
        
        # Set up GPU memory pool based on available memory
        if USE_ADVANCED_FEATURES:
            # Configure memory pool for large GPU
            pool = cp.get_default_memory_pool()
            with cp.cuda.Device(0):
                total_memory = cp.cuda.Device().mem_info[1]
                # Default to 70% of available memory, can be overridden in config
                DEFAULT_MEMORY_ALLOCATION = 0.7
                pool.set_limit(DEFAULT_MEMORY_ALLOCATION * total_memory)
                print(f"GPU memory pool limit set to {DEFAULT_MEMORY_ALLOCATION * 100}% of available memory")
    elif opencv_cuda_available:
        print("CUDA acceleration enabled via OpenCV!")
        CUDA_AVAILABLE = True
    else:
        print("No CUDA-enabled devices found. Falling back to CPU processing.")
except ImportError as e:
    CUDA_AVAILABLE = False
    USE_ADVANCED_FEATURES = False
    MAX_RECOMMENDED_BATCH = 1
    TOTAL_VRAM_GB = 0
    print(f"CUDA modules not available: {str(e)}. Falling back to CPU processing.")

# Constants
DEFAULT_MARKPOINT1 = 5.0   # 5% into video duration, left align
DEFAULT_MARKPOINT2 = 50.0  # 50% into video duration, center align
DEFAULT_MARKPOINT3 = 90.0  # 90% into video duration, right align
EXPECTED_WIDTH = 1280
EXPECTED_HEIGHT = 720
OUTPUT_WIDTH = 720
OUTPUT_HEIGHT = 1280

# GPU helper functions using CuPy and OpenCV for fallback
def to_gpu(img):
    """Transfer an image to the GPU using CuPy or OpenCV"""
    if not CUDA_AVAILABLE:
        return img
    
    try:
        # Try CuPy first
        if cupy_cuda_available:
            return cp.asarray(img)
        # Fallback to OpenCV CUDA if available
        elif opencv_cuda_available:
            return cuda.GpuMat(img)
    except Exception as e:
        print(f"GPU transfer failed: {str(e)}. Using CPU fallback.")
    
    return img

def to_cpu(gpu_mat):
    """Transfer an image from the GPU to the CPU"""
    if not CUDA_AVAILABLE:
        return gpu_mat
    
    try:
        # Check for CuPy array
        if cupy_cuda_available and isinstance(gpu_mat, cp.ndarray):
            return cp.asnumpy(gpu_mat)
        # Check for OpenCV CUDA GpuMat
        elif opencv_cuda_available and isinstance(gpu_mat, cuda.GpuMat):
            return gpu_mat.download()
    except Exception as e:
        print(f"CPU transfer failed: {str(e)}. Using as-is.")
    
    return gpu_mat

def gpu_resize(img, size):
    """Resize an image using GPU acceleration if available"""
    if not CUDA_AVAILABLE:
        return cv2.resize(img, size)
    
    try:
        # CuPy resize using scipy.ndimage
        if cupy_cuda_available:
            # Convert OpenCV size format (width, height) to numpy/cupy format (height, width)
            target_height, target_width = size[1], size[0]
            
            # Transfer to GPU if not already there
            if not isinstance(img, cp.ndarray):
                gpu_img = cp.asarray(img)
            else:
                gpu_img = img
            
            # Calculate scale factors
            h_scale = target_height / gpu_img.shape[0]
            w_scale = target_width / gpu_img.shape[1]
            
            # Perform the resize for each channel separately
            channels = []
            for c in range(gpu_img.shape[2]):
                channel = cupyx.scipy.ndimage.zoom(gpu_img[:,:,c], (h_scale, w_scale), order=1)
                channels.append(channel)
            
            # Stack channels back together
            resized = cp.stack(channels, axis=2)
            return resized
        
        # OpenCV CUDA resize fallback
        elif opencv_cuda_available:
            if not isinstance(img, cuda.GpuMat):
                gpu_img = cuda.GpuMat(img)
            else:
                gpu_img = img
            result = cuda.resize(gpu_img, size)
            return result
    except Exception as e:
        print(f"GPU resize failed: {str(e)}. Using CPU fallback.")
    
    # CPU fallback
    return cv2.resize(img, size)
        
def gpu_crop(img, x, y, width, height):
    """Crop an image using GPU acceleration if available"""
    if not CUDA_AVAILABLE:
        return img[y:y+height, x:x+width, :]
    
    try:
        # CuPy crop
        if cupy_cuda_available:
            if not isinstance(img, cp.ndarray):
                gpu_img = cp.asarray(img)
            else:
                gpu_img = img
            return gpu_img[y:y+height, x:x+width, :]
        
        # OpenCV CUDA crop fallback
        elif opencv_cuda_available:
            if not isinstance(img, cuda.GpuMat):
                gpu_img = cuda.GpuMat(img)
            else:
                gpu_img = img
            return gpu_img[y:y+height, x:x+width]
    except Exception as e:
        print(f"GPU crop failed: {str(e)}. Using CPU fallback.")
    
    # CPU fallback
    return img[y:y+height, x:x+width, :]

def resize_and_center_overlay(overlay, target_width, target_height):
    """
    Resize an overlay image to fit within target dimensions while preserving aspect ratio,
    then center it on a canvas of the target dimensions.
    
    Args:
        overlay (numpy.ndarray): The overlay image to resize
        target_width (int): The width of the target canvas
        target_height (int): The height of the target canvas
        
    Returns:
        tuple: The resized overlay and placement coordinates
    """
    # Get the original dimensions
    h, w = overlay.shape[:2]
    
    # Calculate the aspect ratios
    aspect_src = w / h
    aspect_target = target_width / target_height
    
    # Determine the resize dimensions to maintain aspect ratio
    if aspect_src > aspect_target:
        # Image is wider than target, scale by width
        new_width = target_width
        new_height = int(new_width / aspect_src)
    else:
        # Image is taller than target, scale by height
        new_height = target_height
        new_width = int(new_height * aspect_src)
    
    # Resize the image while preserving aspect ratio using GPU if available
    if CUDA_AVAILABLE:
        gpu_overlay = to_gpu(overlay)
        gpu_resized = gpu_resize(gpu_overlay, (new_width, new_height))
        resized_overlay = to_cpu(gpu_resized)
    else:
        resized_overlay = cv2.resize(overlay, (new_width, new_height))
    
    # Calculate position to center the resized image on the canvas
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    return resized_overlay, (y_offset, x_offset, new_height, new_width)

def load_config(config_path):
    """Load and validate the configuration file."""
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        
        # Validate required fields
        required_fields = ['source', 'output']
        for field in required_fields:
            if field not in config:
                print(f"Error: Missing required field '{field}' in config file")
                sys.exit(1)
        
        # Set default values for optional fields
        if 'markpoint1' not in config:
            config['markpoint1'] = DEFAULT_MARKPOINT1
        if 'markpoint2' not in config:
            config['markpoint2'] = DEFAULT_MARKPOINT2
        if 'markpoint3' not in config:
            config['markpoint3'] = DEFAULT_MARKPOINT3
        if 'overlay' not in config:
            config['overlay'] = 'none'
        
        # Set default values for new optimization parameters
        if 'batch_size' not in config:
            config['batch_size'] = 8  # Default batch size
        else:
            # Ensure batch_size is a positive integer
            try:
                batch_size = int(config['batch_size'])
                if batch_size <= 0:
                    print(f"Warning: Invalid batch_size {batch_size}. Using default value of 8.")
                    config['batch_size'] = 8
                else:
                    config['batch_size'] = batch_size
            except ValueError:
                print(f"Warning: Invalid batch_size value. Using default value of 8.")
                config['batch_size'] = 8
                
        if 'gpu_memory_allocation' not in config:
            config['gpu_memory_allocation'] = 0.7  # Default to 70% of GPU memory
        else:
            # Ensure gpu_memory_allocation is a float between 0.1 and 0.95
            try:
                gpu_mem = float(config['gpu_memory_allocation'])
                if gpu_mem < 0.1 or gpu_mem > 0.95:
                    print(f"Warning: gpu_memory_allocation {gpu_mem} out of range (0.1-0.95). Using default value of 0.7.")
                    config['gpu_memory_allocation'] = 0.7
            except ValueError:
                print(f"Warning: Invalid gpu_memory_allocation value. Using default value of 0.7.")
                config['gpu_memory_allocation'] = 0.7
        
        # Validate markpoint values
        try:
            mp1 = float(config['markpoint1'])
            mp2 = float(config['markpoint2'])
            mp3 = float(config['markpoint3'])
            
            if not (0 <= mp1 < mp2 < mp3 <= 100):
                print("Error: Markpoints must be in ascending order and between 0 and 100")
                sys.exit(1)
                
        except ValueError:
            print("Error: Markpoints must be numeric values")
            sys.exit(1)
            
        return config
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in config file")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        sys.exit(1)

def ensure_templates_directory():
    """Create templates directory if it doesn't exist."""
    if not os.path.exists('templates'):
        os.makedirs('templates')
        with open(os.path.join('templates', 'templates-here.txt'), 'w') as f:
            f.write("Place your template overlay files in this directory.")
        print("Created 'templates' directory")

def process_video_pipeline(config, verbose=False):
    """Process a single video using a parallel pipeline architecture for improved speed."""
    source_path = os.path.normpath(config['source'])
    output_path = os.path.normpath(config['output'])
    markpoint1 = float(config['markpoint1']) / 100.0  # Convert percentage to decimal
    markpoint2 = float(config['markpoint2']) / 100.0
    markpoint3 = float(config['markpoint3']) / 100.0
    
    if verbose:
        print(f"Pipeline processing - source={source_path}, output={output_path}")
        print(f"Processing with markpoints: {markpoint1*100}%, {markpoint2*100}%, {markpoint3*100}%")
    
    try:
        # Check if source file exists
        if not os.path.isfile(source_path):
            print(f"Error: Source file not found at {source_path}")
            return None
        
        # Configure memory pool based on user settings
        if CUDA_AVAILABLE and cupy_cuda_available:
            memory_allocation = config.get('gpu_memory_allocation', 0.7)
            try:
                pool = cp.get_default_memory_pool()
                with cp.cuda.Device(0):
                    total_memory = cp.cuda.Device().mem_info[1]
                    pool.set_limit(memory_allocation * total_memory)
                    if verbose:
                        print(f"GPU memory pool limit set to {memory_allocation*100:.1f}% "
                              f"({memory_allocation*total_memory/(1024**3):.2f} GB)")
            except Exception as e:
                if verbose:
                    print(f"Could not configure GPU memory pool: {str(e)}")
        
        # Open the video to get properties before starting threads
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {source_path}")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure the input video has expected properties
        if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
            print(f"Warning: Input video dimensions ({width}x{height}) differ from expected ({EXPECTED_WIDTH}x{EXPECTED_HEIGHT})")
        
        # Close the video for now, it will be reopened in the reader thread
        cap.release()
        
        # Calculate frame indices for markpoints
        markpoint1_frame = int(total_frames * markpoint1)
        markpoint2_frame = int(total_frames * markpoint2)
        markpoint3_frame = int(total_frames * markpoint3)
        
        # Pre-calculate all x_offsets to avoid redundant calculations
        x_offsets = np.zeros(total_frames, dtype=np.int32)
        for frame_idx in range(total_frames):
            if frame_idx < markpoint1_frame:
                # Before markpoint1: left-aligned
                x_offsets[frame_idx] = 0
            elif frame_idx < markpoint2_frame:
                # Between markpoint1 and markpoint2: animate from left to center
                progress = (frame_idx - markpoint1_frame) / max(1, (markpoint2_frame - markpoint1_frame))
                max_offset = (width - OUTPUT_WIDTH) / 2  # Center position
                x_offsets[frame_idx] = int(progress * max_offset)
            elif frame_idx < markpoint3_frame:
                # Between markpoint2 and markpoint3: animate from center to right
                progress = (frame_idx - markpoint2_frame) / max(1, (markpoint3_frame - markpoint2_frame))
                start_offset = (width - OUTPUT_WIDTH) / 2  # Center position
                end_offset = width - OUTPUT_WIDTH  # Right-aligned position
                x_offsets[frame_idx] = int(start_offset + (progress * (end_offset - start_offset)))
            else:
                # After markpoint3: right-aligned
                x_offsets[frame_idx] = width - OUTPUT_WIDTH
            
            # Ensure x_offset is within bounds
            x_offsets[frame_idx] = max(0, min(width - OUTPUT_WIDTH, x_offsets[frame_idx]))
        
        # Create output path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(source_path)
        name, ext = os.path.splitext(filename)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        output_file = os.path.join(output_path, f"{name}_{timestamp}.mp4")
        
        # Load and pre-process overlay if specified
        overlay_data = None
        if config['overlay'] != 'none':
            template_path = os.path.join('templates', config['overlay'])
            if os.path.exists(template_path):
                overlay = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
                if overlay is None:
                    print(f"Warning: Could not load overlay image at {template_path}")
                else:
                    if verbose:
                        print(f"Loaded overlay: {template_path}, shape: {overlay.shape}")
                    # Pre-process overlay once instead of for every frame
                    resized_overlay, placements = resize_and_center_overlay(overlay, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                    overlay_data = (resized_overlay, placements)
            else:
                print(f"Warning: Overlay image not found at {template_path}")
        
        # Calculate the vertical offset for placing the frame in portrait mode
        y_offset = (OUTPUT_HEIGHT - height) // 2
        
        # Create queues for thread communication
        frame_queue = queue.Queue(maxsize=INPUT_QUEUE_SIZE)
        output_queue = queue.Queue(maxsize=OUTPUT_QUEUE_SIZE)
        
        # Create and start the reader thread
        reader = FrameReaderThread(source_path, frame_queue, total_frames, x_offsets, verbose)
        
        # Determine number of processor threads based on CPU cores and CUDA availability
        if CUDA_AVAILABLE:
            # With CUDA, we're mostly GPU-bound, so fewer processor threads
            num_processors = 1
            if USE_ADVANCED_FEATURES:
                # For high-VRAM GPUs, we can process more in parallel
                num_processors = min(2, os.cpu_count() or 1)
        else:
            # CPU-bound, so use more threads
            num_processors = max(1, min(4, os.cpu_count() or 1))
        
        if verbose:
            print(f"Starting pipeline with {num_processors} processor thread(s)")
        
        # Create the writer thread
        writer = FrameWriterThread(output_queue, output_file, fps, verbose)
        
        # Start all threads
        start_time = time.time()
        reader.start()
        
        # Create and start processor threads
        processors = []
        for i in range(num_processors):
            processor = FrameProcessorThread(frame_queue, output_queue, y_offset, overlay_data, verbose)
            processor.start()
            processors.append(processor)
        
        writer.start()
        
        # Wait for all threads to complete
        reader.join()
        for processor in processors:
            processor.join()
        writer.join()
        
        elapsed_time = time.time() - start_time
        frames_processed = writer.frames_written
        fps_processed = frames_processed / elapsed_time
        
        print(f"Pipeline processing complete: {frames_processed} frames in {elapsed_time:.2f}s ({fps_processed:.2f} fps)")
        print(f"Silent video saved to {output_file}")
        
        # Create a path for the final video with audio
        name_parts = os.path.splitext(output_file)
        final_output = f"{name_parts[0]}_with_audio{name_parts[1]}"
        
        # Add audio from source video to the processed video
        print(f"Adding audio from source to processed video...")
        final_output_file = add_audio_from_source(source_path, output_file, final_output, verbose)
        
        if final_output_file:
            print(f"Final video with audio saved to {final_output_file}")
            # Remove the silent video to save space if audio was added successfully
            try:
                os.remove(output_file)
                if verbose:
                    print(f"Removed silent video: {output_file}")
            except Exception as e:
                if verbose:
                    print(f"Could not remove silent video: {str(e)}")
            return final_output_file
        else:
            print(f"Failed to add audio. Silent video is still available at {output_file}")
            return output_file
            
    except Exception as e:
        print(f"Error in process_video_pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_video(config, verbose=False):
    """Process a single video to apply panning effect.
    
    This function will use the pipeline architecture when possible for better performance,
    or fall back to the original sequential processing method.
    """
    # Use pipeline architecture for improved speed when appropriate conditions are met
    if USE_ADVANCED_FEATURES and CUDA_AVAILABLE:
        if verbose:
            print("Using advanced pipeline architecture for processing")
        return process_video_pipeline(config, verbose)
    else:
        if verbose:
            print("Using standard sequential processing")
    source_path = os.path.normpath(config['source'])
    output_path = os.path.normpath(config['output'])
    markpoint1 = float(config['markpoint1']) / 100.0  # Convert percentage to decimal
    markpoint2 = float(config['markpoint2']) / 100.0
    markpoint3 = float(config['markpoint3']) / 100.0
    
    if verbose:
        print(f"Normalized paths: source={source_path}, output={output_path}")
        print(f"Processing with markpoints: {markpoint1*100}%, {markpoint2*100}%, {markpoint3*100}%")
    
    try:
        # Check if source file exists
        if not os.path.isfile(source_path):
            print(f"Error: Source file not found at {source_path}")
            return None
            
        # Configure memory pool based on user settings
        if CUDA_AVAILABLE and cupy_cuda_available:
            # Get user-defined memory allocation or use default
            memory_allocation = config.get('gpu_memory_allocation', 0.7)
            
            # Set the memory pool limit
            try:
                pool = cp.get_default_memory_pool()
                with cp.cuda.Device(0):
                    total_memory = cp.cuda.Device().mem_info[1]
                    pool.set_limit(memory_allocation * total_memory)
                    if verbose:
                        print(f"GPU memory pool limit set to {memory_allocation*100:.1f}% "
                              f"({memory_allocation*total_memory/(1024**3):.2f} GB)")
            except Exception as e:
                if verbose:
                    print(f"Could not configure GPU memory pool: {str(e)}")
        
        # Open the source video
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {source_path}")
            print(f"Ensure the path is correct and the video file is valid.")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get batch size from config and adapt to GPU capabilities
        batch_size = config.get('batch_size', 1)
        if CUDA_AVAILABLE:
            # Ensure batch size is reasonable for GPU
            batch_size = min(batch_size, MAX_RECOMMENDED_BATCH)
            if verbose:
                print(f"Using batch size: {batch_size} (requested: {config.get('batch_size', 1)}, "
                      f"max recommended: {MAX_RECOMMENDED_BATCH})")
        else:
            # Fallback to single frame processing on CPU
            batch_size = 1
            if verbose and config.get('batch_size', 1) > 1:
                print(f"CUDA not available, using batch size: {batch_size} "
                      f"(requested: {config.get('batch_size', 1)})")
        
        if verbose:
            print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Ensure the input video is the expected dimensions
        if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
            print(f"Warning: Input video dimensions ({width}x{height}) differ from expected ({EXPECTED_WIDTH}x{EXPECTED_HEIGHT})")
        
        # Calculate frame indices for markpoints
        markpoint1_frame = int(total_frames * markpoint1)
        markpoint2_frame = int(total_frames * markpoint2)
        markpoint3_frame = int(total_frames * markpoint3)
        
        if verbose:
            print(f"Markpoint frames: {markpoint1_frame}, {markpoint2_frame}, {markpoint3_frame}")
        
        # Create output path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(source_path)
        name, ext = os.path.splitext(filename)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        output_file = os.path.join(output_path, f"{name}_{timestamp}.mp4")
        
        # Set up the output video writer - 720x1280 (portrait)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        
        if not out.isOpened():
            print(f"Error: Could not create output video writer")
            cap.release()
            return None
        
        # Load and pre-process overlay if specified
        overlay_data = None
        if config['overlay'] != 'none':
            template_path = os.path.join('templates', config['overlay'])
            if os.path.exists(template_path):
                overlay = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
                if overlay is None:
                    print(f"Warning: Could not load overlay image at {template_path}")
                else:
                    if verbose:
                        print(f"Loaded overlay: {template_path}, shape: {overlay.shape}")
                    # Pre-process overlay once instead of for every frame
                    resized_overlay, placements = resize_and_center_overlay(overlay, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                    overlay_data = (resized_overlay, placements)
            else:
                print(f"Warning: Overlay image not found at {template_path}")
        
        # Pre-allocate portrait frame to reuse
        portrait_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
        
        # Calculate the vertical offset for cropping once
        y_offset = (OUTPUT_HEIGHT - height) // 2
        
        # Process each frame
        current_frame = 0
        start_time = time.time()
        
        # Prepare array of x_offsets for the entire video
        x_offsets = np.zeros(total_frames, dtype=np.int32)
        
        # Pre-calculate all x_offsets to avoid redundant calculations
        for frame_idx in range(total_frames):
            if frame_idx < markpoint1_frame:
                # Before markpoint1: left-aligned
                x_offsets[frame_idx] = 0
            elif frame_idx < markpoint2_frame:
                # Between markpoint1 and markpoint2: animate from left to center
                progress = (frame_idx - markpoint1_frame) / max(1, (markpoint2_frame - markpoint1_frame))
                max_offset = (width - OUTPUT_WIDTH) / 2  # Center position
                x_offsets[frame_idx] = int(progress * max_offset)
            elif frame_idx < markpoint3_frame:
                # Between markpoint2 and markpoint3: animate from center to right
                progress = (frame_idx - markpoint2_frame) / max(1, (markpoint3_frame - markpoint2_frame))
                start_offset = (width - OUTPUT_WIDTH) / 2  # Center position
                end_offset = width - OUTPUT_WIDTH  # Right-aligned position
                x_offsets[frame_idx] = int(start_offset + (progress * (end_offset - start_offset)))
            else:
                # After markpoint3: right-aligned
                x_offsets[frame_idx] = width - OUTPUT_WIDTH
            
            # Ensure x_offset is within bounds
            x_offsets[frame_idx] = max(0, min(width - OUTPUT_WIDTH, x_offsets[frame_idx]))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get pre-calculated x_offset for current frame
            x_offset = x_offsets[current_frame]
            
            # Add debugging to check source frame shape before cropping
            if verbose and current_frame == 0:
                print(f"Source frame shape: {frame.shape}")
                print(f"x_offset: {x_offset}, OUTPUT_WIDTH: {OUTPUT_WIDTH}")
                print(f"Calculated crop region: x_offset={x_offset} to {x_offset+OUTPUT_WIDTH}")
            
            # Ensure x_offset + OUTPUT_WIDTH doesn't exceed the frame width
            if x_offset + OUTPUT_WIDTH > width:
                if verbose:
                    print(f"Warning: Crop region exceeds frame width. Adjusting x_offset from {x_offset} to {width - OUTPUT_WIDTH}")
                x_offset = width - OUTPUT_WIDTH
            
            # Save the first frame for debugging if requested
            if current_frame == 0 and '--check-first-frame' in sys.argv:
                debug_dir = os.path.join(output_path, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                
                # Save the source frame
                source_debug_path = os.path.join(debug_dir, f"{name}_source_frame.jpg")
                cv2.imwrite(source_debug_path, frame)
                print(f"Saved source frame to {source_debug_path}")
            
            # Crop the frame to get the 720 wide section - use GPU if available
            try:
                if CUDA_AVAILABLE:
                    # Transfer frame to GPU
                    gpu_frame = to_gpu(frame)
                    
                    # Crop on GPU
                    gpu_cropped = gpu_crop(gpu_frame, x_offset, 0, OUTPUT_WIDTH, height)
                    
                    # Transfer back to CPU
                    cropped_frame = to_cpu(gpu_cropped)
                else:
                    # CPU fallback
                    cropped_frame = frame[:, x_offset:x_offset+OUTPUT_WIDTH, :]
                
                if verbose and current_frame == 0:
                    print(f"Cropped frame shape: {cropped_frame.shape}")
                
                # Save the cropped frame for debugging if requested
                if current_frame == 0 and '--check-first-frame' in sys.argv:
                    cropped_debug_path = os.path.join(debug_dir, f"{name}_cropped_frame.jpg")
                    cv2.imwrite(cropped_debug_path, cropped_frame)
                    print(f"Saved cropped frame to {cropped_debug_path}")
            except Exception as e:
                print(f"Error during frame cropping: {str(e)}")
                print(f"Frame shape: {frame.shape}, x_offset: {x_offset}, OUTPUT_WIDTH: {OUTPUT_WIDTH}")
                raise
            
            # Clear the portrait frame for reuse (more efficient than creating a new array)
            portrait_frame.fill(0)
            
            # Place cropped frame in the middle of the portrait frame vertically
            try:
                if CUDA_AVAILABLE:
                    # For complex slicing operations, sometimes CPU operations are still more efficient
                    # due to memory transfer overheads
                    portrait_frame[y_offset:y_offset+height, :, :] = cropped_frame
                else:
                    portrait_frame[y_offset:y_offset+height, :, :] = cropped_frame
                
                if verbose and current_frame == 0:
                    print(f"Portrait frame shape after placement: {portrait_frame.shape}")
                    print(f"y_offset: {y_offset}, height: {height}")
            except Exception as e:
                print(f"Error during frame placement: {str(e)}")
                print(f"Portrait frame shape: {portrait_frame.shape}, y_offset: {y_offset}, height: {height}")
                print(f"Cropped frame shape: {cropped_frame.shape}")
                raise
            
            # Add overlay if available - use pre-processed data
            if overlay_data is not None:
                resized_overlay, placements = overlay_data
                y_pos, x_pos, h, w = placements
                
                if verbose and current_frame == 0:
                    print(f"Overlay shape: {resized_overlay.shape}")
                    print(f"Overlay placement: y_pos={y_pos}, x_pos={x_pos}, h={h}, w={w}")
                
                # Verify overlay bounds are within the portrait frame
                if y_pos < 0 or x_pos < 0 or y_pos + h > OUTPUT_HEIGHT or x_pos + w > OUTPUT_WIDTH:
                    if verbose and current_frame == 0:
                        print("Warning: Overlay placement exceeds portrait frame bounds, adjusting...")
                    
                    # Adjust placement to ensure it's within bounds
                    y_pos = max(0, min(y_pos, OUTPUT_HEIGHT - h))
                    x_pos = max(0, min(x_pos, OUTPUT_WIDTH - w))
                
                try:
                    # Simple direct overlay with alpha if present
                    if resized_overlay.shape[2] == 4:  # With alpha channel
                        # Get the overlay area in the portrait frame
                        overlay_area = portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w]
                        
                        # Debug the overlay_area
                        if verbose and current_frame == 0:
                            print(f"Overlay area shape: {overlay_area.shape}")
                            print(f"Overlay area max value before blending: {np.max(overlay_area)}")
                        
                        if CUDA_AVAILABLE:
                            try:
                                # GPU-accelerated alpha blending
                                # Create arrays on GPU
                                gpu_overlay = cp.asarray(resized_overlay)
                                gpu_area = cp.asarray(overlay_area)
                                
                                # Alpha blending formula on GPU
                                alpha_factor = gpu_overlay[:,:,3].astype(cp.float32) / 255.0
                                
                                # Create the result array on GPU
                                blended_gpu = cp.zeros_like(gpu_area)
                                
                                # Apply blending for each channel
                                for c in range(3):
                                    blended_gpu[:,:,c] = (gpu_area[:,:,c] * (1 - alpha_factor) + 
                                                      gpu_overlay[:,:,c] * alpha_factor).astype(cp.uint8)
                                
                                # Transfer back to CPU
                                blended_area = cp.asnumpy(blended_gpu)
                                
                                # Copy the blended result back to the portrait frame
                                portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w] = blended_area
                            except Exception as cuda_e:
                                # Fallback to CPU if CUDA blending fails
                                if verbose and current_frame == 0:
                                    print(f"CUDA blending failed, falling back to CPU: {str(cuda_e)}")
                                # CPU fallback implementation
                                blended_area = overlay_area.copy()
                                alpha_factor = resized_overlay[:,:,3].astype(float) / 255.0
                                for c in range(3):
                                    blended_area[:,:,c] = (overlay_area[:,:,c] * (1 - alpha_factor) + 
                                                       resized_overlay[:,:,c] * alpha_factor).astype(np.uint8)
                                portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w] = blended_area
                        else:
                            # CPU implementation
                            blended_area = overlay_area.copy()
                            alpha_factor = resized_overlay[:,:,3].astype(float) / 255.0
                            for c in range(3):
                                blended_area[:,:,c] = (overlay_area[:,:,c] * (1 - alpha_factor) + 
                                                   resized_overlay[:,:,c] * alpha_factor).astype(np.uint8)
                            portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w] = blended_area
                        
                        if verbose and current_frame == 0:
                            print(f"Overlay area max value after blending: {np.max(portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w])}")
                    else:
                        # For non-transparent overlays
                        if CUDA_AVAILABLE:
                            try:
                                # Use CUDA-accelerated addWeighted
                                gpu_overlay_area = to_gpu(portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w])
                                gpu_overlay_img = to_gpu(resized_overlay)
                                
                                # Perform blending on GPU
                                gpu_result = cuda.addWeighted(gpu_overlay_area, 0.7, gpu_overlay_img, 0.3, 0.0)
                                
                                # Transfer result back to CPU and place in the portrait frame
                                portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w] = to_cpu(gpu_result)
                            except Exception as cuda_e:
                                if verbose and current_frame == 0:
                                    print(f"CUDA addWeighted failed, falling back to CPU: {str(cuda_e)}")
                                # Fallback to CPU
                                cv2.addWeighted(
                                    portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w], 0.7,
                                    resized_overlay, 0.3, 0,
                                    dst=portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w]
                                )
                        else:
                            # CPU version
                            cv2.addWeighted(
                                portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w], 0.7,
                                resized_overlay, 0.3, 0,
                                dst=portrait_frame[y_pos:y_pos+h, x_pos:x_pos+w]
                            )
                except Exception as e:
                    print(f"Error applying overlay: {str(e)}")
                    print(f"Portrait frame shape: {portrait_frame.shape}")
                    print(f"Overlay dimensions: {resized_overlay.shape}")
                    print(f"Placement coordinates: y_pos={y_pos}, x_pos={x_pos}, h={h}, w={w}")
            
            # Save the final portrait frame (with overlay) for debugging if requested
            if current_frame == 0 and '--check-first-frame' in sys.argv:
                portrait_debug_path = os.path.join(debug_dir, f"{name}_portrait_frame.jpg")
                cv2.imwrite(portrait_debug_path, portrait_frame)
                print(f"Saved portrait frame to {portrait_debug_path}")
            
            # Write the frame to output video
            out.write(portrait_frame)
            
            current_frame += 1
            
            # Show progress
            if current_frame % 100 == 0 or current_frame == total_frames:
                progress_percent = int(current_frame/total_frames*100)
                elapsed_time = time.time() - start_time
                frames_per_second = current_frame / max(1, elapsed_time)
                remaining_frames = total_frames - current_frame
                estimated_remaining_time = remaining_frames / max(1, frames_per_second)
                
                if verbose:
                    print(f"Processing {filename}: {current_frame}/{total_frames} frames ({progress_percent}%) - "
                          f"{frames_per_second:.2f} fps, ETA: {estimated_remaining_time:.2f}s")
                else:
                    print(f"Processing {filename}: {progress_percent}% complete, ETA: {estimated_remaining_time:.2f}s")
        
        # Release resources
        cap.release()
        out.release()
        
        elapsed_time = time.time() - start_time
        print(f"Processing complete for {filename}. Silent video saved to {output_file} (took {elapsed_time:.2f}s)")
        
        # Create a path for the final video with audio
        name_parts = os.path.splitext(output_file)
        final_output = f"{name_parts[0]}_with_audio{name_parts[1]}"
        
        # Add audio from source video to the processed video
        print(f"Adding audio from source to processed video...")
        final_output_file = add_audio_from_source(source_path, output_file, final_output, verbose)
        
        if final_output_file:
            print(f"Final video with audio saved to {final_output_file}")
            # Remove the silent video to save space if audio was added successfully
            try:
                os.remove(output_file)
                if verbose:
                    print(f"Removed silent video: {output_file}")
            except Exception as e:
                if verbose:
                    print(f"Could not remove silent video: {str(e)}")
            return final_output_file
        else:
            print(f"Failed to add audio. Silent video is still available at {output_file}")
            return output_file
    
    except Exception as e:
        print(f"Error processing video {source_path}: {str(e)}")
        return None

def add_audio_from_source(source_video, silent_video, output_video, verbose=False):
    """
    Add audio from the source video to the processed silent video.
    
    Args:
        source_video (str): Path to the original video with audio
        silent_video (str): Path to the processed video without audio
        output_video (str): Path to save the final video with audio
        verbose (bool): Whether to show detailed progress information
    
    Returns:
        str: Path to the final video with audio, or None if failed
    """
    try:
        if verbose:
            print(f"Adding audio from {source_video} to {silent_video}")
        
        # Get the audio from the original video
        input_audio = ffmpeg.input(source_video).audio
        
        # Get the video from the processed file
        input_video = ffmpeg.input(silent_video).video
        
        # Combine audio and video
        output = ffmpeg.output(input_video, input_audio, output_video, codec='copy')
        
        # Run the FFmpeg command
        ffmpeg.run(output, quiet=not verbose, overwrite_output=True)
        
        if verbose:
            print(f"Successfully added audio to {output_video}")
        
        return output_video
    except Exception as e:
        print(f"Error adding audio: {str(e)}")
        # If using FFmpeg directly is necessary as a fallback
        try:
            if verbose:
                print("Attempting to use direct FFmpeg command...")
            
            cmd = [
                'ffmpeg', '-y',
                '-i', silent_video,
                '-i', source_video,
                '-c:v', 'copy',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                output_video
            ]
            
            result = subprocess.run(cmd, 
                                   stdout=subprocess.PIPE if not verbose else None,
                                   stderr=subprocess.PIPE if not verbose else None)
            
            if result.returncode == 0:
                if verbose:
                    print(f"Successfully added audio using direct FFmpeg command")
                return output_video
            else:
                print(f"Error executing FFmpeg command: {result.stderr.decode() if result.stderr else 'Unknown error'}")
                return None
        except Exception as sub_e:
            print(f"Error using direct FFmpeg command: {str(sub_e)}")
            return None

def process_batch(config, verbose=False):
    """Process a batch of videos based on the config."""
    source_path = config['source']
    
    # Normalize file paths (handle Windows backslashes)
    source_path = os.path.normpath(source_path)
    
    if verbose:
        print(f"Normalized source path: {source_path}")
    
    # Check if source is a directory
    if os.path.isdir(source_path):
        # Process all video files in the directory
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        processed_files = []
        
        # Get list of video files more efficiently
        video_files = []
        try:
            for f in os.listdir(source_path):
                file_path = os.path.join(source_path, f)
                if os.path.isfile(file_path) and os.path.splitext(f)[1].lower() in video_extensions:
                    video_files.append(f)
        except Exception as e:
            print(f"Error listing directory {source_path}: {str(e)}")
            return []
        
        if not video_files:
            print(f"No video files found in {source_path}")
            return []
        
        print(f"Found {len(video_files)} video file(s) to process")
        
        # Pre-copy the config to avoid repeated copies
        base_config = config.copy()
        
        # Determine if we can use parallel processing for multiple videos
        parallel_processing = USE_ADVANCED_FEATURES and CUDA_AVAILABLE and TOTAL_VRAM_GB >= 16.0
        
        if parallel_processing and len(video_files) > 1 and os.cpu_count() > 2:
            # Process videos in parallel using ThreadPoolExecutor
            max_workers = min(4, os.cpu_count() - 1)  # Leave one core free for system
            print(f"Using parallel processing with {max_workers} workers for batch processing")
            
            def process_single_file(file_info):
                idx, filename = file_info
                file_path = os.path.join(source_path, filename)
                print(f"\nProcessing file {idx+1}/{len(video_files)}: {filename}")
                file_config = base_config.copy()
                file_config['source'] = file_path
                return process_video(file_config, verbose)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Process files in parallel and collect results as they complete
                future_to_file = {executor.submit(process_single_file, (i, filename)): filename 
                                 for i, filename in enumerate(video_files)}
                
                for future in future_to_file:
                    output_file = future.result()
                    if output_file:
                        processed_files.append(output_file)
        else:
            # Sequential processing
            for i, filename in enumerate(video_files):
                file_path = os.path.join(source_path, filename)
                
                print(f"\nProcessing file {i+1}/{len(video_files)}: {filename}")
                # Use the base config and just update the source
                file_config = base_config.copy()
                file_config['source'] = file_path
                
                # Process the video
                output_file = process_video(file_config, verbose)
                if output_file:
                    processed_files.append(output_file)
        
        return processed_files
    else:
        # Process a single video file
        output_file = process_video(config, verbose)
        return [output_file] if output_file else []

def main():
    """Main function to process videos based on config."""
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Video panning effect processor')
        parser.add_argument('--config', required=True, help='Path to configuration JSON file')
        parser.add_argument('--verbose', action='store_true', help='Show detailed progress information')
        parser.add_argument('--check-first-frame', action='store_true', help='Save the first frame as an image for debugging')
        args = parser.parse_args()
        
        if not os.path.exists(args.config):
            print(f"Error: Config file not found at {args.config}")
            sys.exit(1)
        
        # Ensure templates directory exists
        ensure_templates_directory()
        
        # Load the configuration
        config = load_config(args.config)
        
        # Normalize paths in config
        if 'source' in config:
            config['source'] = os.path.normpath(config['source'])
        if 'output' in config:
            config['output'] = os.path.normpath(config['output'])
        
        # Debug info for overlay
        if args.verbose and 'overlay' in config and config['overlay'] != 'none':
            template_path = os.path.join('templates', config['overlay'])
            if os.path.exists(template_path):
                # Test read the overlay to check if it's valid
                try:
                    test_overlay = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
                    if test_overlay is not None:
                        print(f"Overlay image found and readable: {template_path}")
                        print(f"Overlay dimensions: {test_overlay.shape}")
                        print(f"Overlay channels: {test_overlay.shape[2] if len(test_overlay.shape) > 2 else 1}")
                        
                        # Save a copy of the overlay to the output directory for verification
                        test_output_dir = os.path.normpath(config['output'])
                        os.makedirs(test_output_dir, exist_ok=True)
                        overlay_output = os.path.join(test_output_dir, f"overlay_debug_{config['overlay']}")
                        cv2.imwrite(overlay_output, test_overlay)
                        print(f"Saved debug copy of overlay to {overlay_output}")
                    else:
                        print(f"WARNING: Overlay image could not be read: {template_path}")
                except Exception as e:
                    print(f"ERROR reading overlay image: {str(e)}")
            else:
                print(f"WARNING: Overlay image not found: {template_path}")
        
        print(f"Starting video processing with configuration from {args.config}")
        print(f"Source: {config['source']}")
        print(f"Output: {config['output']}")
        
        start_time = time.time()
        
        # Process the video(s)
        output_files = process_batch(config, args.verbose)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if output_files:
            print(f"\nProcessing complete. {len(output_files)} file(s) processed in {elapsed_time:.2f} seconds.")
            for file in output_files:
                print(f"- {file}")
        else:
            print(f"\nNo files were successfully processed.")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
