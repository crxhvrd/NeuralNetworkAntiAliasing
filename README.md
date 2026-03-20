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
python nnaa_studio.py
```

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
2. Adjust **hyperparameters** if needed (defaults work well for most cases):
   - **Learning Rate:** `0.00001` — lower = more stable, higher = faster
   - **Train Batch:** `16` — reduce if you run out of memory
   - **Epochs/Run:** `5` — how many epochs before evaluating
3. Set the **Model Name** and **Output Directory**
4. Click **▶ Start Training**
5. Watch the **Training Log** — the model auto-saves whenever it beats its best error score
6. Click **■ Stop** when satisfied (training runs indefinitely otherwise)

The trained model is saved as `<output_dir>/<model_name>/<model_name>.keras`.

### 🔄 Convert Tab

1. Select your trained `.keras` model file
2. Choose an output path for the `.fx` shader file
3. Click **⚡ Convert to Shader**
4. Copy the generated `.fx` file to your ReShade `Shaders` folder

### 🧪 Test Tab

1. Select a `.keras` model file
2. Select an input image (an aliased screenshot)
3. Click **▶ Run Inference** to see the before/after comparison
4. Click **💾 Save Result** to export the anti-aliased image

---

## Command-Line Usage

You can also use the tools from the command line:

### Train
```bash
python nnaa_train.py
```
Edit the paths at the top of the script to point to your dataset.

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
Input Image → Extract Luma → Conv2D 8×8 → Conv2D 3×3 → Conv2D 3×3 → Conv2D 3×3 → ConvTranspose 2×2 → Add to Luma → Recombine Color
```

The converter exports all learned weights as hardcoded HLSL constants in a ReShade compute shader, so the neural network runs entirely on the GPU in real-time.

## Credits

Original NNAA model and shader concept by [Léo Calvis](https://github.com/sareyko/NN-Shaders).
