# PBR Texture Generator

A Python tool that automatically generates PBR (Physically Based Rendering) textures from a single albedo map.

## Features

Generates the following PBR textures:
- Normal Map
- Bump Map (Height)
- Roughness Map
- Specular Map
- Ambient Occlusion (AO) Map
- Metallic Map
- Emission Map

All textures are generated in both PNG and JPEG formats, maintaining the original input dimensions.

## Requirements

- Python 3.6 or higher
- OpenCV
- NumPy
- Pillow

## Installation

1. Clone this repository or download the source code
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python pbr_texture_generator.py path/to/your/albedo_texture.png
```

Specify custom output directory:
```bash
python pbr_texture_generator.py path/to/your/albedo_texture.png --output_dir path/to/output
```

## Output Structure

The script creates two subdirectories in the output location:
- `png/`: Contains all textures in PNG format
- `jpg/`: Contains all textures in JPEG format

For example, if your input file is `metal.png`, the output will be:
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

- Input file must be in PNG or JPEG format
- Input filename will be used as the material name (e.g., `metal.png` → `metal_*`)
- All generated textures will match the input image dimensions

## How It Works

The script uses various image processing techniques to generate PBR textures:

1. **Normal Map**: Uses Sobel filters to detect edges and generate surface normals
2. **Bump Map**: Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance height details
3. **Roughness Map**: Estimates surface roughness based on detail frequency
4. **Specular Map**: Derived from brightness in LAB color space
5. **AO Map**: Simulates ambient occlusion using multiple Gaussian blurs
6. **Metallic Map**: Analyzes color saturation to identify metallic surfaces
7. **Emission Map**: Detects bright and saturated regions that could be emissive

## Limitations

- These are approximate maps generated heuristically
- Results vary based on the input albedo texture quality
- Not a replacement for true PBR baking from 3D geometry
- AO and Normal maps are 2D-based estimations

## License

MIT License 