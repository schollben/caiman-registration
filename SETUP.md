# CaImAn Registration Setup Guide

## Quick Setup

### 1. Install wxPython in your caiman environment

```bash
# Activate the caiman environment
conda activate caiman

# Install wxPython
conda install -c conda-forge wxpython
```

### 2. Open in VS Code

- Open this folder in VS Code
- VS Code will automatically use the caiman environment (configured in `.vscode/settings.json`)
- The environment will be activated automatically when you open a terminal

### 3. Run the Registration Script

**Option A: With GUI (for multiple folders)**
```bash
python image_registration.py
```
This will open a dialog where you can:
1. Select multiple directories containing TIF stacks
2. Choose which registration operations to perform (First Rigid, Additional Rigid, NoRMCorre)
3. Process all selected directories

**Option B: Command Line (for single folder)**
```bash
# Rigid registration
python image_registration.py /path/to/tif/directory

# Non-rigid registration
python image_registration.py /path/to/tif/directory --nonrigid
```

## What the Script Does

1. **Finds TIF stacks**: Locates all `.tif` and `.tiff` files in the directory
2. **Computes template**: Uses the first TIF stack to create a registration template
3. **Registers all stacks**: Applies motion correction to each stack using the template
4. **Saves outputs**:
   - `registered/` - Contains registered TIF stacks
   - `registered/registration_template.tif` - The template used
   - `registered/shifts/` - Motion correction shifts for each stack

## Troubleshooting

### wxPython won't install
If `conda install` doesn't work, try:
```bash
# Option 1: Install system dependencies first
sudo apt-get install -y libgtk-3-dev libwebkit2gtk-4.0-dev

# Option 2: Use pip (after conda install fails)
pip install wxPython

# Option 3: Skip GUI and use command-line mode
python image_registration.py /path/to/directory
```

### VS Code not using correct Python
1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose: `Python 3.x.x ('caiman')`

### Environment not activating
Reload VS Code window:
1. Press `Ctrl+Shift+P`
2. Type "Developer: Reload Window"

## Registration Parameters

Edit these in `image_registration.py` if needed:
- `fr = 30` - Frame rate (fps)
- `max_shift_um = (32, 32)` - Maximum shift in microns
- `patch_motion_um = (64., 64.)` - Patch size for non-rigid
- `overlaps = (32, 32)` - Overlap between patches

## Directory Structure

```
your-data-folder/
├── stack1.tif
├── stack2.tif
└── registered/           (created by script)
    ├── stack1.tif        (registered)
    ├── stack2.tif        (registered)
    ├── registration_template.tif
    └── shifts/
        ├── stack1_shifts_rig.npy
        └── stack2_shifts_rig.npy
```
