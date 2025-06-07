# PBR Texture Generator

A Python tool designed to **accelerate the PBR texture creation pipeline** by automatically generating approximated PBR (Physically Based Rendering) maps from a single albedo image.

## Purpose

This tool is ideal for rapid prototyping, automated workflows, or early-stage asset development. Instead of manually creating each texture map, this script provides a fast and consistent way to generate a full PBR texture set from just one input.

## Features

Generates the following PBR texture maps:
- **Normal Map**
- **Bump (Height) Map**
- **Roughness Map**
- **Specular Map**
- **Ambient Occlusion (AO) Map**
- **Metallic Map**
- **Emission Map**

All output textures are exported in both **PNG** and **JPEG** formats, keeping the original image resolution.

## Requirements

- Python 3.6 or higher
- OpenCV
- NumPy
- Pillow

## Installation

1. Clone this repository or download the source code.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
````

## Usage

Basic usage:

```bash
python pbr_texture_generator.py path/to/your/albedo_texture.png
```

With a custom output directory:

```bash
python pbr_texture_generator.py path/to/your/albedo_texture.png --output_dir path/to/output
```

## Output Structure

The script creates two subfolders in the output directory:

* `png/`: All generated textures in PNG format
* `jpg/`: All generated textures in JPEG format

Example (input: `metal.png`):

```
output/
├── png/
│   ├── metal_albedo.png
│   ├── metal_normal.png
│   ├── metal_bump.png
│   └── ...
└── jpg/
    ├── metal_albedo.jpg
    ├── metal_normal.jpg
    ├── metal_bump.jpg
    └── ...
```

## Input Requirements

* Input must be a PNG or JPEG image.
* The filename will be used as the base name for all output textures (e.g., `wood.jpg` → `wood_normal.png`, etc.).
* All generated maps will match the input image dimensions.

## How It Works

This tool uses heuristic and image processing techniques to simulate key PBR maps from the albedo input:

1. **Normal Map** – Edge detection (Sobel filters) to infer surface direction.
2. **Bump Map** – Contrast enhancement (CLAHE) for height approximation.
3. **Roughness Map** – Detail frequency analysis to estimate roughness.
4. **Specular Map** – Derived from luminance in LAB color space.
5. **AO Map** – Ambient occlusion simulation via multi-pass Gaussian blur.
6. **Metallic Map** – Detects metallic areas based on saturation thresholds.
7. **Emission Map** – Isolates highly bright and saturated regions.

## Limitations

* These maps are **approximations** based solely on 2D albedo input.
* Results will vary depending on the quality and characteristics of the input texture.
* This is **not a replacement for proper baking** using high-poly 3D geometry.
* AO and Normal maps are 2D-derived and may lack depth accuracy.

## License

MIT License
