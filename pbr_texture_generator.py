import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

class PBRTextureGenerator:
    def __init__(self, input_path, output_dir=None):
        """
        Initialize the PBR texture generator.
        
        Args:
            input_path (str): Path to the input albedo texture
            output_dir (str, optional): Base directory to save generated textures. If None, uses input directory
        """
        self.input_path = input_path
        self.base_output_dir = output_dir if output_dir else os.path.dirname(input_path)
        self.material_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Create PNG and JPEG output directories
        self.png_output_dir = os.path.join(self.base_output_dir, 'png')
        self.jpg_output_dir = os.path.join(self.base_output_dir, 'jpg')
        os.makedirs(self.png_output_dir, exist_ok=True)
        os.makedirs(self.jpg_output_dir, exist_ok=True)
        
        self.albedo = None
        self.grayscale = None
        self.input_dimensions = None
        
    def load_image(self):
        """Load and preprocess the input albedo texture."""
        # Read image using OpenCV
        self.albedo = cv2.imread(self.input_path)
        if self.albedo is None:
            raise ValueError(f"Failed to load image: {self.input_path}")
            
        # Store input dimensions
        self.input_dimensions = self.albedo.shape[:2]
            
        # Convert BGR to RGB
        self.albedo = cv2.cvtColor(self.albedo, cv2.COLOR_BGR2RGB)
        
        # Create grayscale version
        self.grayscale = cv2.cvtColor(self.albedo, cv2.COLOR_RGB2GRAY)
        
    def ensure_dimensions(self, image):
        """Ensure the image matches input dimensions."""
        if image.shape[:2] != self.input_dimensions:
            return cv2.resize(image, (self.input_dimensions[1], self.input_dimensions[0]))
        return image
        
    def generate_normal_map(self):
        """Generate a blue-dominant tangent-space normal map from grayscale, then invert the colors."""
        # Compute Sobel gradients (height derivatives)
        sobelx = cv2.Sobel(self.grayscale, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.grayscale, cv2.CV_32F, 0, 1, ksize=3)

        # Scale factor: controls the 'strength' of the normal map
        strength = 2.0  # You can adjust this value

        # Normalize gradients to [-1, 1]
        dx = sobelx / 255.0 * strength
        dy = sobely / 255.0 * strength

        # Calculate Z (blue) channel
        dz = np.ones_like(dx)
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        nx = dx / norm
        ny = dy / norm
        nz = dz / norm

        # Remap from [-1,1] to [0,255]
        normal = np.zeros((*self.grayscale.shape, 3), dtype=np.float32)
        normal[..., 0] = (nx + 1.0) * 0.5 * 255.0  # X -> R
        normal[..., 1] = (ny + 1.0) * 0.5 * 255.0  # Y -> G
        normal[..., 2] = (nz + 1.0) * 0.5 * 255.0  # Z -> B

        normal = np.clip(normal, 0, 255).astype(np.uint8)
        # Invert the normal map colors
        return self.ensure_dimensions(255 - normal)
    
    def generate_bump_map(self):
        """Generate bump map (height map) from grayscale."""
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        bump_map = clahe.apply(self.grayscale)
        
        # Optional: Apply slight Gaussian blur
        bump_map = cv2.GaussianBlur(bump_map, (3,3), 0)
        return self.ensure_dimensions(bump_map)
    
    def generate_roughness_map(self):
        """Generate roughness map based on detail estimation."""
        # Apply high-pass filter to detect detail
        blur = cv2.GaussianBlur(self.grayscale, (0,0), 3)
        detail = cv2.subtract(self.grayscale, blur)
        
        # Normalize and invert (high detail = high roughness)
        roughness = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)
        roughness = 255 - roughness  # Invert to match roughness convention
        return self.ensure_dimensions(roughness)
    
    def generate_specular_map(self):
        """Generate specular map based on brightness."""
        # Convert to LAB color space
        lab = cv2.cvtColor(self.albedo, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        
        # Normalize and enhance contrast
        specular = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)
        specular = cv2.pow(specular/255.0, 0.5) * 255  # Gamma correction
        return self.ensure_dimensions(specular.astype(np.uint8))
    
    def generate_ao_map(self):
        """Generate ambient occlusion map."""
        # Apply multiple Gaussian blurs
        blur1 = cv2.GaussianBlur(self.grayscale, (0,0), 3)
        blur2 = cv2.GaussianBlur(self.grayscale, (0,0), 7)
        
        # Calculate local contrast
        ao_map = cv2.subtract(blur1, blur2)
        
        # Normalize and invert
        ao_map = cv2.normalize(ao_map, None, 0, 255, cv2.NORM_MINMAX)
        ao_map = 255 - ao_map  # Invert to match AO convention
        return self.ensure_dimensions(ao_map)
    
    def generate_metallic_map(self):
        """Generate metallic map based on color analysis."""
        # Convert to LAB color space
        lab = cv2.cvtColor(self.albedo, cv2.COLOR_RGB2LAB)
        
        # Calculate color saturation (distance from gray)
        a = lab[:,:,1].astype(np.float32) - 128
        b = lab[:,:,2].astype(np.float32) - 128
        saturation = np.sqrt(np.clip(a*a + b*b, 0, None))
        
        # Normalize and invert (low saturation = metallic)
        metallic = cv2.normalize(saturation, None, 0, 255, cv2.NORM_MINMAX)
        metallic = 255 - metallic
        return self.ensure_dimensions(metallic.astype(np.uint8))
    
    def generate_emission_map(self):
        """Generate emission map based on brightness and saturation."""
        # Convert to HSV
        hsv = cv2.cvtColor(self.albedo, cv2.COLOR_RGB2HSV)
        
        # Create mask for bright and saturated pixels
        brightness_mask = hsv[:,:,2] > 200
        saturation_mask = hsv[:,:,1] > 100
        
        # Combine masks
        emission = np.zeros_like(self.grayscale)
        emission[brightness_mask & saturation_mask] = 255
        return self.ensure_dimensions(emission)
    
    def save_texture(self, texture, name):
        """Save a texture to both PNG and JPEG directories."""
        # Save PNG version
        png_path = os.path.join(self.png_output_dir, f"{self.material_name}_{name}.png")
        cv2.imwrite(png_path, texture)
        print(f"Saved PNG: {png_path}")
        
        # Save JPEG version
        jpg_path = os.path.join(self.jpg_output_dir, f"{self.material_name}_{name}.jpg")
        cv2.imwrite(jpg_path, texture, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved JPEG: {jpg_path}")
    
    def generate_all_textures(self):
        """Generate all PBR textures from the input albedo."""
        print("Loading input image...")
        self.load_image()
        
        # Re-export albedo in both formats
        print("Re-exporting albedo...")
        self.save_texture(self.albedo, "albedo")
        
        print("Generating textures...")
        textures = {
            'normal': self.generate_normal_map(),
            'bump': self.generate_bump_map(),
            'roughness': self.generate_roughness_map(),
            'specular': self.generate_specular_map(),
            'ao': self.generate_ao_map(),
            'metallic': self.generate_metallic_map(),
            'emission': self.generate_emission_map()
        }
        
        print("Saving textures...")
        for name, texture in textures.items():
            self.save_texture(texture, name)
        
        print("All textures generated successfully!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PBR textures from an albedo map')
    parser.add_argument('input_path', help='Path to the input albedo texture (PNG or JPEG)')
    parser.add_argument('--output_dir', help='Base directory to save generated textures (optional)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    
    input_ext = os.path.splitext(args.input_path)[1].lower()
    if input_ext not in ['.png', '.jpg', '.jpeg']:
        raise ValueError("Input file must be PNG or JPEG format")
    
    generator = PBRTextureGenerator(args.input_path, args.output_dir)
    generator.generate_all_textures()

if __name__ == '__main__':
    main() 