# NNAA Shader Studio

A GUI tool for training, converting, and testing Neural Network Anti-Aliasing (NNAA) models for [ReShade](https://reshade.me/).

Train a CNN to remove aliasing from any game, then export the model directly to a ReShade compute shader.

![NNAA Shader Studio](https://img.shields.io/badge/Python-3.10+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

---

## Requirements

```bash
pip install tensorflow numpy Pillow
```

## Quick Start

```bash
cd models/nnaa
python nnaa_studio.py
```

> All settings (paths, hyperparameters) are automatically saved between sessions.

---

## Preparing a Dataset

The model learns by comparing **aliased** screenshots (no AA) with **clean** screenshots (high-quality AA or supersampled).

### Step 1: Capture Screenshots

For each scene, take two perfectly aligned screenshots:

| Folder | What to capture |
|--------|----------------|
| `bad/`  | AA completely disabled (no FXAA, no MSAA, no TXAA) |
| `fixed/` | Highest quality AA (MSAA x8, or render at 4K and downscale to 1080p) |

> **Important:** Both images of the same scene must have the **exact same filename** (e.g., `scene_001.png`).

### Step 2: Organize Files

```
data/
├── train/
│   ├── bad/1280x720/       ← Aliased training images
│   └── fixed/1280x720/     ← Clean training images
└── test/
    ├── bad/2560x1440/      ← Aliased test images
    └── fixed/2560x1440/    ← Clean test images
```

Aim for **200+ training pairs** across different scenes, lighting, and weather conditions.

---

## Using the App

### 🏋 Train Tab

1. Set the **dataset folder paths** for your training and test images
2. Adjust **hyperparameters**:

   | Parameter | Default | Description |
   |-----------|---------|-------------|
   | Learning Rate | `0.00001` | Initial LR. Auto-reduced by 50% when loss plateaus (ReduceLROnPlateau) |
   | Train Batch | `16` | Reduce if you run out of memory |
   | Test Batch | `4` | Batch size for evaluation |
   | Epochs/Run | `5` | Epochs before evaluating on test set |
   | Patch Size | `128` | Random crop size in pixels. Set `0` for full images |
   | Patience | `20` | Stop after N evaluation runs with no improvement. Set `0` to disable |
   | Augment | ✅ On | Random horizontal + vertical flips (4× effective data) |

3. Set the **Model Name** and **Output Directory**
4. Click **▶ Start Training**
5. Monitor the **loss chart** and **training log** — the model auto-saves when it beats its best error
6. Training stops automatically when patience is exceeded, or click **■ Stop** manually

**Training features:**
- 📈 Live loss sparkline chart with best-loss marker
- ⏱ Elapsed time display
- 🔄 Learning rate automatically halves when loss plateaus
- 🎲 Data augmentation (random flips) and random patch cropping
- ⚡ Parallel image loading for fast dataset caching

The trained model is saved as `<output_dir>/<model_name>/<model_name>.keras`.

### 🔄 Convert Tab

1. Select your trained `.keras` model file
2. Choose an output path for the `.fx` shader file
3. Click **⚡ Convert to Shader**
4. Copy the generated `.fx` file to your ReShade `Shaders` folder

The converter validates the model architecture before conversion and provides clear error messages if the model doesn't match the expected NNAA structure.

### 🧪 Test Tab

1. Select a `.keras` model file
2. Select an input image (an aliased screenshot)
3. Click **▶ Run Inference** to see the before/after comparison
4. Use the **synchronized zoom viewer** to inspect details:

   | Control | Action |
   |---------|--------|
   | Scroll wheel | Zoom in/out (0.25× to 32×) — zooms toward cursor |
   | Click + drag | Pan both images simultaneously |
   | Double-click | Reset to fit-to-window view |

   At ≥2× zoom, pixel-perfect (nearest-neighbor) interpolation is used so you can inspect individual pixels.

5. Click **💾 Save Result** to export the anti-aliased image

> The model is cached — re-running inference on a different image with the same model is near-instant.

---

## Command-Line Usage

### Train
```bash
python nnaa_train.py
```
Edit the configuration variables at the top of the `if __name__` block to set dataset paths, learning rate, patch size, and patience.

### Convert
```bash
# Default: nnaa.keras → out_nnaa.fx
python convert.py

# Custom paths
python convert.py my_model.keras my_shader.fx
```

### Test on a Single Image
```bash
python use.py image.png
```
Produces `image_AA.png` (anti-aliased) and `image_AA_black_diff.png` (difference map).

---

## Installing the Shader in ReShade

1. Copy the generated `.fx` file to your game's `reshade-shaders/Shaders/` folder
2. Launch the game and open the ReShade overlay
3. Enable **Sarenya NNAA** in the technique list

> **Note:** This shader requires ReShade with compute shader support (DX11/DX12/Vulkan).

---

## How It Works

The model is a small CNN that operates on the brightness (luma) channel:

```
Input → Luma → Conv2D 8×8 (stride 2) → 3× Conv2D 3×3 → ConvTranspose 2×2 → Residual → Recombine RGB
```

The converter exports all learned weights as hardcoded HLSL constants in a ReShade compute shader, using:
- **Space-to-Depth** optimization for the 8×8 first layer
- **Ping-pong storage textures** for intermediate layers
- **Fused Depth-to-Space** for the final ConvTranspose layer

The shader runs entirely on the GPU in real-time (~5 compute passes).

## Credits

Original NNAA model and shader concept by [Léo Calvis](https://github.com/sareyko/NN-Shaders).
