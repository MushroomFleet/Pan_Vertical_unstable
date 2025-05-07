# 🎬✨ VIDEO PANNER 3000 ✨🎬

> _Transform your boring landscape videos into **EPIC** portrait masterpieces!_ 🔄🔥🔥🔥

## 🤔 What is this madness?! 🤯

VIDEO PANNER 3000 is a **REVOLUTIONARY** 🚀 tool that transforms your 16:9 landscape videos (1280x720) into 9:16 portrait format (720x1280) with SMOOTH 🧈 panning effects - perfect for social media content! 📱✨

Instead of cropping out important parts of your video or having ugly black bars 🖤, VIDEO PANNER 3000 intelligently moves the frame across your video with customizable animation points! 🎯🎯🎯

### 🚀 NOW WITH CUDA ACCELERATION! 🚀

The latest version of VIDEO PANNER 3000 includes **BLAZING FAST** 🔥 CUDA acceleration for NVIDIA GPUs, dramatically speeding up:
- Video processing (up to 3-5x faster! ⚡️)
- Image transformations (cropping, resizing)
- Alpha blending and effects

### 🦸‍♂️ HIGH-VRAM OPTIMIZATIONS FOR 24GB+ GPUs! 🦸‍♀️

If you're running with a high-end NVIDIA GPU (12GB+ VRAM), VIDEO PANNER 3000 will automatically detect it and enable special optimizations:
- 🚄 **Batch Frame Processing** - Process multiple frames simultaneously (configurable via `batch_size`)
- 🧠 **Smart Memory Management** - Efficient VRAM allocation (configurable via `gpu_memory_allocation`)
- 🔥 **Enhanced Alpha Blending** - Special high-quality effects only enabled for high-VRAM GPUs
- 🏎️ **Parallel Operations** - Keep more data on the GPU for faster processing

With a 24GB VRAM GPU, expect **dramatically faster** processing compared to 4GB cards - perfect for bulk processing those social media videos! ⚡️⚡️⚡️

### 🌟 INCREDIBLE FEATURES 🌟

- 🎭 Convert ANY landscape video to portrait format with intelligent panning
- 🎮 Define EXACTLY where your camera movements happen with custom markpoints
- 🔄 Batch process ENTIRE FOLDERS of videos with ONE command 😱
- 🔊 Preserves original audio from your source videos - no silent movies! 🎵
- 🏷️ Add your own watermarks or branding with custom overlays
- ⏱️ Automatic timestamp appending to output filenames
- 📊 Progress tracking with ETA (because waiting is BORING 😴)
- 🖥️ CUDA acceleration for NVIDIA GPUs (so much faster! 🏎️💨)
- 🔁 Automatic CPU fallback when CUDA is not available

## 🛠️ Installation 🛠️

### Step 1: Clone this AMAZING repository 🤩

```bash
git clone https://github.com/yourusername/video-panner-3000.git
cd video-panner-3000
```

### Step 2: Install dependencies 📦📦📦

```bash
# Create a virtual environment (HIGHLY recommended! 👍)
python -m venv venv

# Activate the virtual environment
# For Windows 🪟
venv\Scripts\activate
# For macOS/Linux 🐧
source venv/bin/activate

# Install all dependencies (with CUDA support) 🔽
pip install -r requirements.txt

# Or install manually with components you need:
# Basic version (without CUDA):
pip install opencv-python numpy ffmpeg-python

# For CUDA acceleration (NVIDIA GPUs only 🖥️):
pip install opencv-contrib-python cupy-cuda11x  # For CUDA 11.x
# Or if you have CUDA 12.x:
pip install opencv-contrib-python cupy-cuda12x

# Install FFmpeg (system dependency) 🎬
# For Windows:
# Download from https://ffmpeg.org/download.html or use Chocolatey:
choco install ffmpeg
# For macOS:
brew install ffmpeg
# For Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg
```

### Step 3: Set up CUDA Environment Variables ⚙️🔧

The script requires the `CUDA_PATH` environment variable to be set. There are multiple ways to do this:

For Windows:
```powershell
# Find your CUDA installation path
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"  # Update version as needed
```

For Linux/macOS:
```bash
export CUDA_PATH=/usr/local/cuda-11.8  # Update version as needed
```

You can also set this permanently in your system environment variables or modify your virtual environment activation scripts.

> 📌 **For detailed instructions on all CUDA_PATH setup methods, see our [CUDA Setup Guide](CUDA_PATH.md)** 📌

### CUDA Requirements 🖥️🚀

For CUDA acceleration, you'll need:

- NVIDIA GPU with compute capability 3.0 or higher
- NVIDIA drivers installed (version 418.xx or later)
- CUDA Toolkit (version 11.x or 12.x is recommended)
- CuPy package that matches your CUDA version (cupy-cuda11x for CUDA 11.x)
- Properly set `CUDA_PATH` environment variable ([see our setup guide](CUDA_PATH.md))

The tool uses two acceleration methods:
1. **CuPy** (primary) - Provides fast GPU versions of NumPy operations
2. **OpenCV CUDA** (secondary) - Falls back to OpenCV's CUDA functions if available

Our code is designed to handle most CUDA configuration internally once `CUDA_PATH` is set. If no GPU is found or there are any issues with CUDA, the tool will safely fall back to CPU processing.

## 🚀 How to Use It 🚀

VIDEO PANNER 3000 is SUPER EASY to use! 🙌 Just follow these simple steps:

### 1️⃣ Create a Configuration File 📝

Create a `config.json` file with your desired settings:

```json
{
    "source": "/path/to/videos/",
    "output": "/path/to/output/",
    "markpoint1": 5.0,
    "markpoint2": 50.0,
    "markpoint3": 90.0,
    "overlay": "none"
}
```

### 2️⃣ Run the Script 🏃‍♂️💨

```bash
python video_panner.py --config config.json
```

### 3️⃣ ENJOY YOUR AMAZING VIDEOS! 🎉🥳🎊

## 📋 Configuration Options Explained 📋

Your `config.json` file is the COMMAND CENTER for your video transformations! 🎛️

| Setting | Description | Example |
|---------|-------------|---------|
| `source` 📂 | Path to input video file OR folder for batch processing | `"/videos/my_cool_video.mp4"` or `"/videos/"` |
| `output` 💾 | Where to save your TRANSFORMED videos | `"/processed_videos/"` |
| `markpoint1` ⬅️ | When to START panning (% of video duration) | `5.0` (starts at 5% of video) |
| `markpoint2` ⏺️ | When to reach CENTER position (% of video duration) | `50.0` (centered at half-way point) |
| `markpoint3` ➡️ | When to reach RIGHT position (% of video duration) | `90.0` (right-aligned at 90% of video) |
| `overlay` 🖼️ | Overlay image filename (place in `/templates/` folder) or "none" | `"mylogo.png"` or `"none"` |
| `batch_size` 🔄 | Number of frames to process simultaneously for faster performance (with sufficient VRAM) | `8` (default) or higher for 12GB+ GPUs |
| `gpu_memory_allocation` 💻 | Percentage of VRAM to allocate (0.1-0.95) | `0.7` (default, uses 70% of available VRAM) |

## 🎯 Step-by-Step Guide to PERFECTION 🎯

### 🔍 Processing a Single Video 🔍

1. Create your `config.json`:
```json
{
    "source": "/videos/awesome_landscape_video.mp4",
    "output": "/videos/processed/",
    "markpoint1": 5.0,
    "markpoint2": 50.0,
    "markpoint3": 90.0,
    "overlay": "none"
}
```

2. Run the command:
```bash
python video_panner.py --config config.json
```

3. 🎉 Your video will be saved as `/videos/processed/awesome_landscape_video_YYYYMMDD_HHMMSS.mp4`!

### 📁 Batch Processing an ENTIRE FOLDER 📁

1. Create your `config.json` pointing to a FOLDER:
```json
{
    "source": "/videos/vacation_videos/",
    "output": "/videos/processed/",
    "markpoint1": 10.0,
    "markpoint2": 40.0,
    "markpoint3": 85.0,
    "overlay": "none"
}
```

2. Run the command:
```bash
python video_panner.py --config config.json
```

3. 🎉 ALL your videos will be processed with the SAME panning effect!

### 🖼️ Adding Your AWESOME Brand Overlay 🖼️

1. Create a transparent PNG image with your branding
2. Save it in the `/templates/` folder (will be created automatically)
3. Update your `config.json`:
```json
{
    "source": "/videos/product_demo.mp4",
    "output": "/videos/branded/",
    "markpoint1": 5.0,
    "markpoint2": 50.0,
    "markpoint3": 90.0,
    "overlay": "my_brand_logo.png"
}
```

4. Run the command:
```bash
python video_panner.py --config config.json
```

5. 🎉 Your video now has your branding overlaid throughout!

## 🛠️ Command Line Options 🛠️

### `--config` (REQUIRED) 📄

Tell the script where to find your configuration file.

```bash
python video_panner.py --config my_special_config.json
```

### `--verbose` (OPTIONAL) 🔊

Shows DETAILED progress information - for when you're SUPER impatient! 😜

```bash
python video_panner.py --config config.json --verbose
```

## 🔮 Advanced Examples 🔮

### 🔥 Quick Panning Effect 🔥

```json
{
    "source": "/videos/action_scene.mp4",
    "output": "/videos/processed/",
    "markpoint1": 10.0,
    "markpoint2": 30.0,
    "markpoint3": 60.0,
    "overlay": "none"
}
```
This configuration creates a FASTER panning effect by making the camera reach the right side earlier (60% instead of 90%)! ⚡️

### 🐢 Slow, Cinematic Panning 🐢

```json
{
    "source": "/videos/nature_timelapse.mp4",
    "output": "/videos/cinematic/",
    "markpoint1": 20.0,
    "markpoint2": 60.0,
    "markpoint3": 95.0,
    "overlay": "film_grain.png"
}
```
This creates a SLOWER, more DRAMATIC panning effect with a film grain overlay! 🎭

### 💼 Batch Processing with Corporate Branding 💼

```json
{
    "source": "/videos/product_demos/",
    "output": "/videos/social_media_ready/",
    "markpoint1": 5.0,
    "markpoint2": 50.0,
    "markpoint3": 90.0,
    "overlay": "corporate_logo_transparent.png"
}
```
Process ALL your product demos with your corporate branding! 📈

### 🚀 High-VRAM GPU Optimizations 🚀

```json
{
    "source": "/videos/high_quality_videos/",
    "output": "/videos/optimized/",
    "markpoint1": 5.0,
    "markpoint2": 50.0,
    "markpoint3": 90.0,
    "overlay": "watermark.png",
    "batch_size": 24,
    "gpu_memory_allocation": 0.85
}
```
This configuration maximizes performance on 24GB+ VRAM GPUs, processing 24 frames simultaneously and allocating 85% of the available GPU memory for maximum speed! 🏎️💨

## ⚠️ Limitations ⚠️

- 📱 ONLY outputs 720x1280 portrait videos
- ⏱️ Recommended for videos under 2 minutes (but will work with longer videos if you're PATIENT! ⏳)
- 🚫 No zooming functionality (to keep things SIMPLE! 👌)

## 🐛 Troubleshooting 🐛

### 🔴 ERROR: "Could not open video file" 🔴

- Check that your video path is correct 🔍
- Make sure the video file exists and isn't corrupted 🤕
- Verify your video is in a supported format (mp4, mov, avi, mkv) 📼

### 🔵 ERROR: "Could not create output video writer" 🔵

- Check that your output directory exists or can be created 📁
- Make sure you have write permissions to that location 🔒
- Verify you have enough disk space 💾

### 🟠 CUDA Related Issues 🟠

#### "CUDA modules not available. Falling back to CPU processing."

- Make sure you have an NVIDIA GPU installed in your system
- Install or update NVIDIA drivers to the latest version
- Install CUDA Toolkit (version 11.x or 12.x recommended)
- Set the `CUDA_PATH` environment variable to your CUDA installation path
- Verify the correct cupy version is installed (matches your CUDA version)
- Run `nvidia-smi` in terminal to check if your GPU is detected

#### "OpenCV CUDA device count: 0" (but CuPy CUDA works)

This is normal! The pip version of OpenCV often has CUDA interfaces but can't detect CUDA devices properly. Our tool is designed to use CuPy for acceleration instead, which works more reliably.

To verify CUDA is working:
1. Run with `--verbose` flag and look for:
   - "CuPy CUDA available: True"
   - "CUDA acceleration enabled via CuPy!"
2. You should see GPU device name and memory information
3. Processing speed should be faster than CPU-only mode

#### "Error during frame cropping" / "CUDA error"

- Your GPU might be running out of memory - try processing a shorter video
- Close other GPU-intensive applications while running Video Panner
- Try reducing batch size by processing videos one at a time
- Make sure your system isn't switching to integrated graphics
- Check if your CUDA version matches the cupy package version
- As a last resort, disable CUDA by manually downgrading to `opencv-python` instead of `opencv-contrib-python`

#### "GPU transfer failed / GPU crop failed / GPU resize failed"

- These are specific operations that can fail while others succeed
- The tool will automatically fall back to CPU for these specific operations
- No action needed unless you're concerned about performance
- If these happen frequently, try updating your GPU drivers

## 🤝 Contributing 🤝

CONTRIBUTIONS are WELCOME! 🎁 Feel free to submit a pull request or open an issue for:

- 🐞 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🎨 UI enhancements

## 📜 License 📜

This project is licensed under the MIT License - see the LICENSE file for details. 📄

## 🙏 Acknowledgments 🙏

- 🧠 OpenCV for the amazing video processing capabilities
- 💻 NumPy for handling all the math stuff
- 🎬 FFmpeg for powerful audio/video handling capabilities
- 🌎 The open-source community for their ENDLESS inspiration

---

Made with ❤️ (and WAY TOO MANY EMOJIS 🤪) by DRIFT JOHNSON
