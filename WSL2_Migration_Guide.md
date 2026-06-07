# WSL2 Migration Guide: TensorFlow GPU Unlock

This guide outlines the rigorous, verified steps required to port our execution environment to WSL2 (Ubuntu) to natively unlock the NVIDIA Quadro P620 GPU using TensorFlow 2.15.0.

## 1. Install WSL2 and Reboot

Open your Windows PowerShell **as Administrator** and execute:
```powershell
wsl --install
```
*(Reboot your PC).*

## 2. The Manual Override (If Ubuntu is missing or still flashes)

If the installation was interrupted before the reboot or the distribution isn't mounting correctly, verify the status:
Open PowerShell as Administrator and run:
```powershell
wsl --list --verbose
```

**Scenario A:** It says "Windows Subsystem for Linux has no installed distributions."
This means the download failed or was interrupted. Simply force the installation by running:
```powershell
wsl --install -d Ubuntu
```

**Scenario B: Google it, Gpt it, or use your own head and debug it.**


## 3. Create Your Linux Environment, Activate, and Install the "Magic" Package

Open your new "Ubuntu" app from the Windows Start Menu. Once it initializes and you set your UNIX username/password, run these commands to update the system and create your isolated workspace. 

*Note: The default Ubuntu 24.04 ships with Python 3.12, which can cause dependency resolution failures for specific ML tools. We manually inject the deadsnakes PPA to install Python 3.10 and lock TensorFlow to `2.15.0` via the NVIDIA PyPI index.*

```bash
# Add deadsnakes PPA and install Python 3.10
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv python3.10-dev -y

# Create a dedicated virtual environment for the project
python3.10 -m venv ~/skin_lesion_env

# Activate the virtual environment
source ~/skin_lesion_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies and the specialized NVIDIA-indexed TensorFlow
pip install "tensorflow[and-cuda]==2.15.0" pandas numpy scikit-learn --extra-index-url https://pypi.nvidia.com
```

## 4. Verify the GPU

Run this to confirm the Quadro P620 is officially mapped and accessible to TensorFlow natively within WSL2:
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
*(You should see an output indicating a physical GPU device is present).*

## 5. Rerun Previous Phases
If any path-based or serialization errors occur due to the cross-OS transition, re-execute the earlier phases:
```bash
cd /mnt/d/Code/Projects/datacrumbs/datacrumbs_capstone_project_6/
# Regenerate path standards and uncompiled model
python3 -m src.data_preprocessing
python3 -m src.model
```

## 6. Execute Phase 4 Training

With the hardware natively bridged, execute the dynamic training loop:
```bash
cd /mnt/d/Code/Projects/datacrumbs/datacrumbs_capstone_project_6/
python3 -m src.train
```
