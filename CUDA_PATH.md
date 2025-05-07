# Setting CUDA_PATH for Video Panner GPU Acceleration

Setting up the `CUDA_PATH` environment variable is **critical** for enabling GPU acceleration in Video Panner. This variable helps the application find your CUDA installation and properly utilize your NVIDIA GPU.

## How Our Implementation Works

Our video processing script now has two layers of CUDA acceleration handling:

1. **Environment Variable Setup**: You need to set `CUDA_PATH` using one of the methods described below.
2. **Automatic Configuration**: The script automatically configures additional CUDA environment variables internally based on `CUDA_PATH`.

This means you only need to set `CUDA_PATH` by any method you prefer, and the script will handle the rest of the configuration.

> **Important Note**: While we initially modified the virtual environment activation scripts to get CUDA working, this is now just one of several optional ways to set `CUDA_PATH`. Choose the method below that works best for your workflow.

## For Windows:

### Using PowerShell (Recommended)

1. **Install CUDA Toolkit** first if you haven't already. Download from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
   - We've successfully tested with CUDA 11.8, but other versions should work too

2. **Find your CUDA installation path**. It's typically something like:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
   ```
   (update version as needed)

3. **Set the environment variable** in PowerShell:
   ```powershell
   $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
   ```

4. **Verify the setup**:
   ```powershell
   $env:CUDA_PATH
   ```
   
5. **Make it permanent** (optional):
   - Go to Windows Settings > System > About > Advanced system settings
   - Click "Environment Variables"
   - Add a new System variable with name "CUDA_PATH" and the path to your CUDA installation

### Using Virtual Environment (activate.bat)

If you're using a virtual environment with batch file activation:

1. **Edit your virtual environment's activation script**:
   - Navigate to your virtual environment's directory
   - Edit `venv\Scripts\activate.bat` to include:
   ```bat
   @echo off
   set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
   set "PATH=%CUDA_PATH%\bin;%PATH%"
   ```

2. **For deactivation** (optional), edit `venv\Scripts\deactivate.bat`:
   ```bat
   @echo off
   set CUDA_PATH=
   ```

3. **Reactivate your virtual environment**:
   ```
   deactivate
   venv\Scripts\activate
   ```

4. **Verify the setup**:
   ```
   echo %CUDA_PATH%
   ```

## For Linux/macOS:

1. **Install CUDA Toolkit** if needed.

2. **Create/edit activation script** in your venv:
   - Find `bin/activate` in your venv folder
   - Add these lines at the end:
   ```bash
   # Store old path to restore later
   export _OLD_CUDA_PATH="$CUDA_PATH"
   
   # Set CUDA path (adjust version as needed)
   export CUDA_PATH="/usr/local/cuda-12.x"
   export PATH="$CUDA_PATH/bin:$PATH"
   export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
   ```

3. **Update deactivation** in the same file:
   ```bash
   # Add this to the deactivate() function
   if [ -n "${_OLD_CUDA_PATH+_}" ] ; then
       CUDA_PATH="$_OLD_CUDA_PATH"
       export CUDA_PATH
       unset _OLD_CUDA_PATH
   else
       unset CUDA_PATH
   fi
   ```

4. **Reactivate your environment**:
   ```bash
   deactivate
   source venv/bin/activate
   ```

5. **Verify with**:
   ```bash
   echo $CUDA_PATH
   ```

Would you like me to help troubleshoot any specific issues with this setup?
