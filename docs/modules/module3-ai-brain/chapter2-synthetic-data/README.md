# Chapter 2: Photorealistic Simulation & Synthetic Data Generation

## Learning Objectives

After completing this chapter, you will be able to:
- Configure photorealistic rendering in Isaac Sim
- Generate synthetic datasets for computer vision training
- Implement domain randomization techniques
- Create diverse environments for perception training
- Validate synthetic data quality and realism

## Introduction to Photorealistic Simulation

Photorealistic simulation is crucial for humanoid robotics, particularly for perception systems that must operate in human environments. Unlike traditional simulation that focuses primarily on physics accuracy, photorealistic simulation emphasizes visual fidelity to create images that closely match real-world conditions. This is essential for training computer vision models that will eventually operate on real robots.

### Why Photorealistic Simulation Matters

- **Perception Training**: Train neural networks with realistic visual data
- **Domain Adaptation**: Bridge the sim-to-real gap for vision systems
- **Safety**: Test perception algorithms without real-world risks
- **Cost-Effectiveness**: Generate large datasets without physical hardware
- **Reproducibility**: Consistent testing conditions for algorithm evaluation

## Isaac Sim Rendering Pipeline

### RTX Rendering Architecture

Isaac Sim leverages NVIDIA's RTX technology for photorealistic rendering:

```
Scene Description (USD)
    ↓
Material Definition (MDL/PBR)
    ↓
Light Transport Simulation (Path Tracing)
    ↓
Denoising (AI-accelerated)
    ↓
Post-Processing (Color Grading, Effects)
    ↓
Final Image Output
```

### Configuring Photorealistic Materials

```python
import omni
from pxr import Usd, UsdShade, Sdf, Gf
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage

def setup_photorealistic_materials():
    """Configure photorealistic materials for humanoid robot"""

    # Example: Set up skin-like material for humanoid head
    head_material_path = "/World/Humanoid/Head/Materials/SkinMaterial"

    # Create material prim
    stage = omni.usd.get_context().get_stage()
    material_prim = stage.DefinePrim(head_material_path, "Material")

    # Create USD preview surface shader
    shader_path = head_material_path + "/PreviewSurface"
    shader_prim = stage.DefinePrim(shader_path, "Shader")
    shader = UsdShade.Shader(shader_prim)
    shader.CreateIdAttr("UsdPreviewSurface")

    # Configure skin-like properties
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.9, 0.8, 0.7))  # Skin color
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.1, 0.1, 0.1))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).Set(
        Gf.Vec3f(0, 0, 1))

    # Apply material to geometry
    head_geom = get_prim_at_path("/World/Humanoid/Head")
    UsdShade.MaterialBindingAPI(head_geom).Bind(shader.GetPrim())

def setup_environment_materials():
    """Configure photorealistic environment materials"""

    # Floor material - realistic wood/stone
    floor_material_path = "/World/Room/Materials/FloorMaterial"
    stage = omni.usd.get_context().get_stage()
    material_prim = stage.DefinePrim(floor_material_path, "Material")

    shader_path = floor_material_path + "/PreviewSurface"
    shader_prim = stage.DefinePrim(shader_path, "Shader")
    shader = UsdShade.Shader(shader_prim)
    shader.CreateIdAttr("UsdPreviewSurface")

    # Wood-like material
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.6, 0.4, 0.2))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.5, 0.5, 0.5))

    # Add normal map for texture variation
    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).Set(
        Gf.Vec3f(0, 0, 1))

def setup_lighting_for_photorealism():
    """Configure realistic lighting setup"""

    # Add dome light for environment illumination
    dome_light_path = "/World/DomeLight"
    stage = omni.usd.get_context().get_stage()

    # Create dome light
    dome_prim = stage.DefinePrim(dome_light_path, "DomeLight")
    dome_light = stage.GetPrimAtPath(dome_light_path)

    # Configure dome light properties
    dome_light.GetAttribute("inputs:color").Set((0.8, 0.8, 0.9))  # Cool daylight
    dome_light.GetAttribute("inputs:intensity").Set(300.0)

    # Add directional key light (sun-like)
    key_light_path = "/World/KeyLight"
    key_prim = stage.DefinePrim(key_light_path, "DistantLight")
    key_light = stage.GetPrimAtPath(key_light_path)

    key_light.GetAttribute("inputs:color").Set((1.0, 0.95, 0.9))  # Warm light
    key_light.GetAttribute("inputs:intensity").Set(500.0)
    key_light.GetAttribute("xformOp:rotateXYZ").Set((45, 30, 0))
```

### Advanced Rendering Settings

```python
def configure_advanced_rendering():
    """Configure advanced rendering settings for photorealism"""

    # Enable advanced rendering features
    settings = carb.settings.get_settings()

    # Enable RTX features
    settings.set("/rtx/antialiasing/enable", True)
    settings.set("/rtx/antialiasing/technique", "TAA")

    # Enable denoising
    settings.set("/rtx/denoise/enable", True)
    settings.set("/rtx/denoise/technique", "Optix")

    # Configure global illumination
    settings.set("/rtx/pathtracing/enable", True)
    settings.set("/rtx/pathtracing/maxBounces", 8)
    settings.set("/rtx/pathtracing/maxNonBounces", 4)

    # Enable subsurface scattering for skin rendering
    settings.set("/rtx/subsurface/enable", True)

    # Configure motion blur for dynamic scenes
    settings.set("/rtx/motionblur/enable", True)
    settings.set("/rtx/motionblur/quality", 2)  # High quality
```

## Synthetic Data Generation Pipeline

### Isaac Sim Perception Tools

Isaac Sim includes specialized tools for generating synthetic training data:

```python
import omni
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.synthetic_utils import plot
from omni.isaac.synthetic_utils.sensors import Camera
from omni.isaac.synthetic_utils.exporter import Exporter
from PIL import Image
import numpy as np
import os

class SyntheticDataGenerator:
    def __init__(self, output_dir="synthetic_data"):
        self.output_dir = output_dir
        self.camera = None
        self.scene_objects = []

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/labels", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)
        os.makedirs(f"{output_dir}/masks", exist_ok=True)

        # Initialize camera
        self.setup_camera()

    def setup_camera(self):
        """Setup camera for data capture"""
        from omni.isaac.core.prims import XFormPrim
        from omni.replicator.core import random_col
        import omni.replicator.core as rep

        # Create camera prim
        self.camera_prim = XFormPrim(
            prim_path="/World/Camera",
            position=np.array([2.0, 0.0, 1.5]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )

        # Configure camera properties
        self.camera = Camera(
            prim_path="/World/Camera",
            frequency=20,  # 20 Hz capture rate
            resolution=(640, 480)
        )

        # Enable RGB, depth, and segmentation
        self.camera.add_data_output("rgb", "data")
        self.camera.add_data_output("depth", "data")
        self.camera.add_data_output("instance_segmentation", "data")

    def capture_frame(self, frame_id):
        """Capture a single frame with annotations"""

        # Get RGB image
        rgb_data = self.camera.get_rgb_data()
        rgb_image = Image.fromarray(rgb_data, mode="RGB")
        rgb_image.save(f"{self.output_dir}/images/frame_{frame_id:06d}.png")

        # Get depth data
        depth_data = self.camera.get_depth_data()
        depth_image = Image.fromarray((depth_data * 255).astype(np.uint8))
        depth_image.save(f"{self.output_dir}/depth/frame_{frame_id:06d}_depth.png")

        # Get segmentation masks
        seg_data = self.camera.get_segmentation_data()
        seg_image = Image.fromarray(seg_data.astype(np.uint8))
        seg_image.save(f"{self.output_dir}/masks/frame_{frame_id:06d}_seg.png")

        # Generate label file
        self.generate_label_file(frame_id, seg_data)

    def generate_label_file(self, frame_id, segmentation_data):
        """Generate label file for the captured frame"""

        # Create label dictionary with object information
        labels = {
            "frame_id": frame_id,
            "objects": [],
            "camera_intrinsics": self.camera.get_intrinsics()
        }

        # Extract object information from segmentation
        unique_ids = np.unique(segmentation_data)

        for obj_id in unique_ids:
            if obj_id == 0:  # Skip background
                continue

            # Get object mask
            mask = (segmentation_data == obj_id)

            # Calculate bounding box
            y_coords, x_coords = np.where(mask)
            if len(x_coords) > 0 and len(y_coords) > 0:
                bbox = {
                    "x_min": int(np.min(x_coords)),
                    "y_min": int(np.min(y_coords)),
                    "x_max": int(np.max(x_coords)),
                    "y_max": int(np.max(y_coords)),
                    "class_id": int(obj_id),
                    "class_name": self.get_object_name(obj_id)
                }

                # Calculate center point
                bbox["center_x"] = int((bbox["x_min"] + bbox["x_max"]) / 2)
                bbox["center_y"] = int((bbox["y_min"] + bbox["y_max"]) / 2)

                labels["objects"].append(bbox)

        # Save label file
        import json
        with open(f"{self.output_dir}/labels/frame_{frame_id:06d}.json", 'w') as f:
            json.dump(labels, f, indent=2)

    def get_object_name(self, object_id):
        """Map object ID to object name"""
        # This would be populated based on your scene objects
        object_names = {
            1: "humanoid_robot",
            2: "table",
            3: "chair",
            4: "cup",
            5: "book",
            # Add more mappings as needed
        }
        return object_names.get(object_id, f"object_{object_id}")

    def generate_dataset(self, num_frames=1000):
        """Generate synthetic dataset with multiple frames"""

        for frame_id in range(num_frames):
            # Randomize scene configuration
            self.randomize_scene()

            # Capture frame
            self.capture_frame(frame_id)

            # Print progress
            if frame_id % 100 == 0:
                print(f"Captured {frame_id}/{num_frames} frames")

        print(f"Dataset generation complete! Generated {num_frames} frames in {self.output_dir}")

    def randomize_scene(self):
        """Randomize scene elements for domain randomization"""

        # Randomize object positions
        for obj_path in self.scene_objects:
            prim = omni.usd.get_context().get_stage().GetPrimAtPath(obj_path)
            if prim.IsValid():
                # Random position offset
                import random
                new_pos = [
                    random.uniform(-2.0, 2.0),  # X range
                    random.uniform(-1.5, 1.5),  # Y range
                    random.uniform(0.1, 1.0)    # Z range (height)
                ]

                # Apply new position
                prim.GetAttribute("xformOp:translate").Set(new_pos)

        # Randomize lighting conditions
        self.randomize_lighting()

        # Randomize camera position/view
        self.randomize_camera()

    def randomize_lighting(self):
        """Randomize lighting conditions"""
        stage = omni.usd.get_context().get_stage()

        # Randomize dome light intensity and color
        dome_light = stage.GetPrimAtPath("/World/DomeLight")
        if dome_light.IsValid():
            import random
            intensity = random.uniform(200, 500)
            color_temp = random.uniform(5000, 8000)  # Kelvin

            # Convert color temperature to RGB approximation
            color = self.color_temperature_to_rgb(color_temp)

            dome_light.GetAttribute("inputs:intensity").Set(intensity)
            dome_light.GetAttribute("inputs:color").Set(color)

    def randomize_camera(self):
        """Randomize camera position and orientation"""
        import random

        # Random camera position around the scene
        radius = random.uniform(1.5, 3.0)
        angle = random.uniform(0, 2 * 3.14159)
        height = random.uniform(1.0, 2.0)

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height

        self.camera_prim.set_world_pose(position=np.array([x, y, z]))

        # Point camera toward center
        set_camera_view(eye=np.array([x, y, z]), target=np.array([0, 0, 0.8]))

    def color_temperature_to_rgb(self, kelvin):
        """Convert color temperature in Kelvin to RGB values"""
        temp = kelvin / 100

        if temp <= 66:
            red = 255
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)

        blue = temp - 10
        if temp >= 66:
            blue = 138.5177312231 * np.log(blue) - 305.0447927307
        elif temp <= 19:
            blue = 0

        # Clamp values to 0-255 range
        red = max(0, min(255, red))
        green = max(0, min(255, green))
        blue = max(0, min(255, blue))

        return (red/255.0, green/255.0, blue/255.0)
```

### Domain Randomization Techniques

Domain randomization is crucial for creating robust perception systems:

```python
import random
import numpy as np

class DomainRandomizer:
    def __init__(self):
        self.randomization_ranges = {
            # Lighting randomization
            'light_intensity': (100, 800),
            'light_temperature': (3000, 8000),  # Kelvin
            'light_position_radius': (2.0, 5.0),

            # Material randomization
            'diffuse_color_range': (0.1, 1.0),
            'roughness_range': (0.1, 0.9),
            'metallic_range': (0.0, 0.2),

            # Camera randomization
            'camera_distance': (1.0, 4.0),
            'camera_height': (0.5, 2.5),
            'camera_noise': (0.0, 0.05),

            # Environmental randomization
            'floor_texture_scale': (0.5, 2.0),
            'object_scale_variance': (0.8, 1.2),
            'background_complexity': (0, 5)  # Number of background objects
        }

    def randomize_lighting(self, stage):
        """Apply lighting randomization"""
        # Randomize dome light
        dome_light = stage.GetPrimAtPath("/World/DomeLight")
        if dome_light.IsValid():
            intensity = random.uniform(*self.randomization_ranges['light_intensity'])
            temp = random.uniform(*self.randomization_ranges['light_temperature'])
            color = self.color_temperature_to_rgb(temp)

            dome_light.GetAttribute("inputs:intensity").Set(intensity)
            dome_light.GetAttribute("inputs:color").Set(color)

        # Randomize directional lights
        for i in range(3):  # Up to 3 additional lights
            light_path = f"/World/RandomLight_{i}"
            light_prim = stage.GetPrimAtPath(light_path)

            if not light_prim.IsValid():
                continue

            # Randomize properties
            intensity = random.uniform(100, 400)
            color_temp = random.uniform(4000, 7000)
            color = self.color_temperature_to_rgb(color_temp)

            light_prim.GetAttribute("inputs:intensity").Set(intensity)
            light_prim.GetAttribute("inputs:color").Set(color)

    def randomize_materials(self, stage):
        """Apply material randomization"""
        # Get all material prims in the scene
        material_paths = self.get_all_material_paths(stage)

        for path in material_paths:
            material_prim = stage.GetPrimAtPath(path)
            if material_prim.IsValid():
                self.randomize_material(material_prim)

    def randomize_material(self, material_prim):
        """Randomize a single material"""
        # Get shader
        shader_path = material_prim.GetPath().AppendChild("PreviewSurface")
        shader_prim = material_prim.GetStage().GetPrimAtPath(shader_path)

        if shader_prim.IsValid():
            shader = material_prim.GetStage().GetPrimAtPath(shader_path)

            # Randomize diffuse color
            if shader.HasAttribute("inputs:diffuseColor"):
                r = random.uniform(*self.randomization_ranges['diffuse_color_range'])
                g = random.uniform(*self.randomization_ranges['diffuse_color_range'])
                b = random.uniform(*self.randomization_ranges['diffuse_color_range'])
                shader.GetAttribute("inputs:diffuseColor").Set((r, g, b))

            # Randomize roughness
            if shader.HasAttribute("inputs:roughness"):
                roughness = random.uniform(*self.randomization_ranges['roughness_range'])
                shader.GetAttribute("inputs:roughness").Set(roughness)

            # Randomize metallic
            if shader.HasAttribute("inputs:metallic"):
                metallic = random.uniform(*self.randomization_ranges['metallic_range'])
                shader.GetAttribute("inputs:metallic").Set(metallic)

    def randomize_camera(self, camera_prim):
        """Apply camera randomization"""
        # Randomize camera position
        distance = random.uniform(*self.randomization_ranges['camera_distance'])
        angle = random.uniform(0, 2 * np.pi)
        height = random.uniform(*self.randomization_ranges['camera_height'])

        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        z = height

        camera_prim.GetAttribute("xformOp:translate").Set((x, y, z))

    def get_all_material_paths(self, stage):
        """Get all material paths in the scene"""
        material_paths = []

        # This is a simplified approach - in practice, you'd traverse the scene graph
        # to find all material prims
        for prim in stage.TraverseAll():
            if prim.GetTypeName() == "Material":
                material_paths.append(prim.GetPath().pathString)

        return material_paths

    def add_camera_noise(self, image_data):
        """Add realistic camera noise to captured images"""
        noise_level = random.uniform(0, self.randomization_ranges['camera_noise'][1])

        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, image_data.shape)
        noisy_image = np.clip(image_data.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return noisy_image
```

## Environment Generation for Training

### Procedural Environment Generation

```python
import random
import numpy as np

class ProceduralEnvironmentGenerator:
    def __init__(self):
        self.object_library = {
            'furniture': ['chair', 'table', 'sofa', 'bed', 'desk'],
            'kitchen': ['cup', 'plate', 'bowl', 'knife', 'fork'],
            'office': ['computer', 'phone', 'book', 'pen', 'paper'],
            'living_room': ['tv', 'remote', 'pillow', 'lamp', 'plant']
        }

        self.room_types = ['bedroom', 'kitchen', 'living_room', 'office', 'bathroom']

        # Define room dimensions
        self.room_sizes = {
            'bedroom': (4, 4),
            'kitchen': (5, 4),
            'living_room': (6, 5),
            'office': (4, 3),
            'bathroom': (3, 3)
        }

    def generate_environment(self, env_type='random'):
        """Generate a procedural environment"""
        if env_type == 'random':
            env_type = random.choice(self.room_types)

        # Create basic room structure
        room_size = self.room_sizes[env_type]
        self.create_room_structure(env_type, room_size)

        # Add furniture based on room type
        self.add_room_furniture(env_type)

        # Add objects and randomize their properties
        self.add_environment_objects(env_type)

        # Apply domain randomization
        self.apply_domain_randomization()

        return env_type, room_size

    def create_room_structure(self, room_type, size):
        """Create basic room structure"""
        stage = omni.usd.get_context().get_stage()

        # Create room container
        room_path = f"/World/{room_type.title()}Room"
        room_prim = stage.DefinePrim(room_path, "Xform")

        # Create floor
        floor_path = f"{room_path}/Floor"
        floor_prim = stage.DefinePrim(floor_path, "Mesh")

        # Create walls
        wall_height = 2.5
        for i, (pos, rot) in enumerate([
            ((size[0]/2, 0, wall_height/2), (0, 0, 0)),  # Right wall
            ((-size[0]/2, 0, wall_height/2), (0, 180, 0)),  # Left wall
            ((0, size[1]/2, wall_height/2), (0, 90, 0)),  # Front wall
            ((0, -size[1]/2, wall_height/2), (0, -90, 0))  # Back wall
        ]):
            wall_path = f"{room_path}/Wall_{i}"
            wall_prim = stage.DefinePrim(wall_path, "Mesh")

    def add_room_furniture(self, room_type):
        """Add furniture appropriate for room type"""
        if room_type not in self.object_library:
            return

        furniture_list = self.object_library[room_type]

        for i in range(random.randint(2, 5)):  # Add 2-5 furniture pieces
            furniture_type = random.choice(furniture_list)
            self.add_furniture(furniture_type, room_type)

    def add_furniture(self, furniture_type, room_type):
        """Add specific furniture to the room"""
        # This would load furniture from asset library
        # For now, we'll create simple geometric representations
        stage = omni.usd.get_context().get_stage()

        # Generate random position within room bounds
        room_size = self.room_sizes[room_type]
        x = random.uniform(-room_size[0]/2 + 0.5, room_size[0]/2 - 0.5)
        y = random.uniform(-room_size[1]/2 + 0.5, room_size[1]/2 - 0.5)
        z = 0  # On floor

        furniture_path = f"/World/{room_type.title()}Room/{furniture_type}_{random.randint(1000, 9999)}"
        furniture_prim = stage.DefinePrim(furniture_path, "Xform")
        furniture_prim.GetAttribute("xformOp:translate").Set((x, y, z))

        # Apply random rotation
        rotation = random.uniform(0, 360)
        furniture_prim.GetAttribute("xformOp:rotateY").Set(rotation)

    def add_environment_objects(self, room_type):
        """Add smaller objects to the environment"""
        # Add objects on surfaces like tables
        surface_objects = self.object_library.get(room_type, [])

        for _ in range(random.randint(3, 8)):
            if surface_objects:
                obj_type = random.choice(surface_objects)
                self.add_environment_object(obj_type, room_type)

    def add_environment_object(self, obj_type, room_type):
        """Add a specific object to the environment"""
        stage = omni.usd.get_context().get_stage()

        # Position on a surface (simplified - would check for actual surfaces in real implementation)
        room_size = self.room_sizes[room_type]
        x = random.uniform(-room_size[0]/2 + 0.2, room_size[0]/2 - 0.2)
        y = random.uniform(-room_size[1]/2 + 0.2, room_size[1]/2 - 0.2)
        z = random.uniform(0.1, 1.0)  # Height above ground

        obj_path = f"/World/{room_type.title()}Room/{obj_type}_{random.randint(1000, 9999)}"
        obj_prim = stage.DefinePrim(obj_path, "Xform")
        obj_prim.GetAttribute("xformOp:translate").Set((x, y, z))

        # Apply random scale
        scale = random.uniform(0.5, 1.5)
        obj_prim.GetAttribute("xformOp:scale").Set((scale, scale, scale))

    def apply_domain_randomization(self):
        """Apply domain randomization to the environment"""
        dr = DomainRandomizer()
        stage = omni.usd.get_context().get_stage()

        # Randomize lighting
        dr.randomize_lighting(stage)

        # Randomize materials
        dr.randomize_materials(stage)
```

## Data Quality Validation

### Synthetic Data Validation Techniques

```python
import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class SyntheticDataValidator:
    def __init__(self):
        self.metrics = {
            'sharpness': [],
            'color_distribution': [],
            'texture_complexity': [],
            'lighting_consistency': []
        }

    def validate_image_quality(self, image_path):
        """Validate the quality of a synthetic image"""
        img = cv2.imread(image_path)

        # Calculate sharpness using Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Calculate color distribution statistics
        color_stats = self.calculate_color_distribution(img)

        # Calculate texture complexity
        texture_complexity = self.calculate_texture_complexity(gray)

        # Store metrics
        self.metrics['sharpness'].append(sharpness)
        self.metrics['color_distribution'].append(color_stats)
        self.metrics['texture_complexity'].append(texture_complexity)

        return {
            'sharpness': sharpness,
            'color_stats': color_stats,
            'texture_complexity': texture_complexity,
            'is_valid': self.is_image_quality_acceptable(sharpness, texture_complexity)
        }

    def calculate_color_distribution(self, img):
        """Calculate color distribution statistics"""
        # Split image into color channels
        b, g, r = cv2.split(img)

        # Calculate statistics for each channel
        stats_dict = {}
        for i, channel in enumerate([b, g, r]):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            stats_dict[f'channel_{i}'] = {
                'mean': np.mean(channel),
                'std': np.std(channel),
                'skewness': stats.skew(channel.flatten()),
                'kurtosis': stats.kurtosis(channel.flatten())
            }

        return stats_dict

    def calculate_texture_complexity(self, gray_img):
        """Calculate texture complexity using local binary patterns"""
        # Simplified texture complexity measure
        # In practice, you might use more sophisticated methods like LBP or GLCM

        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return np.mean(gradient_magnitude)

    def is_image_quality_acceptable(self, sharpness, texture_complexity):
        """Determine if image quality is acceptable"""
        # Define thresholds based on real image statistics
        min_sharpness = 100  # Adjust based on your requirements
        min_texture_complexity = 10  # Adjust based on your requirements

        return sharpness > min_sharpness and texture_complexity > min_texture_complexity

    def validate_dataset_consistency(self, dataset_dir):
        """Validate consistency across the entire dataset"""
        import os
        import glob

        image_files = glob.glob(f"{dataset_dir}/images/*.png") + glob.glob(f"{dataset_dir}/images/*.jpg")

        for img_file in image_files:
            validation_result = self.validate_image_quality(img_file)

            if not validation_result['is_valid']:
                print(f"Warning: Low quality image detected: {img_file}")

        # Calculate overall dataset statistics
        overall_stats = self.calculate_dataset_statistics()
        print(f"Dataset Quality Report:")
        print(f"  Average Sharpness: {np.mean(self.metrics['sharpness']):.2f}")
        print(f"  Average Texture Complexity: {np.mean(self.metrics['texture_complexity']):.2f}")
        print(f"  Total Images: {len(self.metrics['sharpness'])}")

    def calculate_dataset_statistics(self):
        """Calculate overall statistics for the dataset"""
        stats_dict = {}

        if self.metrics['sharpness']:
            stats_dict['sharpness'] = {
                'mean': np.mean(self.metrics['sharpness']),
                'std': np.std(self.metrics['sharpness']),
                'min': np.min(self.metrics['sharpness']),
                'max': np.max(self.metrics['sharpness']),
                'median': np.median(self.metrics['sharpness'])
            }

        return stats_dict

    def compare_with_real_data(self, synthetic_dir, real_dir):
        """Compare synthetic data distribution with real data"""
        # This would involve more complex statistical comparison
        # such as two-sample Kolmogorov-Smirnov tests, etc.

        print("Comparing synthetic and real data distributions...")

        # Placeholder for actual comparison logic
        # In practice, you'd compare:
        # - Color distributions
        # - Texture characteristics
        # - Sharpness distributions
        # - Lighting conditions
        # - Object appearance

        return {
            'color_similarity': 0.85,  # Example value
            'texture_similarity': 0.78,  # Example value
            'overall_similarity': 0.82   # Example value
        }
```

## Performance Optimization for Data Generation

### Efficient Data Pipeline

```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import threading

class EfficientDataGenerator:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()

    def setup_parallel_generation(self):
        """Setup parallel data generation pipeline"""

        # Use ProcessPoolExecutor for CPU-intensive tasks
        # and ThreadPoolExecutor for I/O-bound tasks
        with ProcessPoolExecutor(max_workers=self.num_workers) as proc_executor:
            with ThreadPoolExecutor(max_workers=self.num_workers) as thread_executor:

                # Submit tasks for parallel processing
                futures = []

                for i in range(1000):  # Generate 1000 frames
                    future = proc_executor.submit(
                        self.generate_single_frame,
                        frame_id=i,
                        randomize=True
                    )
                    futures.append(future)

                # Collect results
                for future in futures:
                    result = future.result()
                    self.save_frame_data(result)

    def generate_single_frame(self, frame_id, randomize=True):
        """Generate a single frame with optional randomization"""

        # Apply randomization if requested
        if randomize:
            self.apply_randomization()

        # Capture data from Isaac Sim
        rgb_data = self.get_camera_data()
        depth_data = self.get_depth_data()
        seg_data = self.get_segmentation_data()

        return {
            'frame_id': frame_id,
            'rgb': rgb_data,
            'depth': depth_data,
            'segmentation': seg_data,
            'timestamp': self.get_simulation_time()
        }

    def apply_randomization(self):
        """Apply scene randomization"""
        # This would call domain randomization functions
        pass

    def get_camera_data(self):
        """Get RGB camera data"""
        # Placeholder - would interface with Isaac Sim camera
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def get_depth_data(self):
        """Get depth data"""
        # Placeholder - would interface with Isaac Sim depth sensor
        return np.random.rand(480, 640).astype(np.float32)

    def get_segmentation_data(self):
        """Get segmentation data"""
        # Placeholder - would interface with Isaac Sim segmentation
        return np.random.randint(0, 10, (480, 640), dtype=np.uint8)

    def get_simulation_time(self):
        """Get current simulation time"""
        # Placeholder
        return 0.0

    def save_frame_data(self, frame_data):
        """Save frame data to disk efficiently"""
        import os
        from PIL import Image
        import json

        output_dir = "efficient_dataset"
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)
        os.makedirs(f"{output_dir}/masks", exist_ok=True)

        frame_id = frame_data['frame_id']

        # Save RGB image
        rgb_img = Image.fromarray(frame_data['rgb'])
        rgb_img.save(f"{output_dir}/images/frame_{frame_id:06d}.png")

        # Save depth
        depth_img = Image.fromarray((frame_data['depth'] * 255).astype(np.uint8))
        depth_img.save(f"{output_dir}/depth/frame_{frame_id:06d}_depth.png")

        # Save segmentation
        seg_img = Image.fromarray(frame_data['segmentation'])
        seg_img.save(f"{output_dir}/masks/frame_{frame_id:06d}_seg.png")

        # Save metadata
        metadata = {
            'frame_id': frame_id,
            'timestamp': frame_data['timestamp'],
            'sensor_config': self.get_sensor_config()
        }

        with open(f"{output_dir}/metadata/frame_{frame_id:06d}.json", 'w') as f:
            json.dump(metadata, f)

    def get_sensor_config(self):
        """Get current sensor configuration"""
        return {
            'resolution': [640, 480],
            'fov': 60,
            'near_plane': 0.1,
            'far_plane': 10.0
        }
```

## Best Practices for Synthetic Data Generation

### 1. Quality Assurance
- Validate synthetic images against real-world statistics
- Use domain adaptation techniques to bridge sim-to-real gap
- Regularly compare synthetic and real data distributions

### 2. Computational Efficiency
- Use appropriate rendering quality settings for your needs
- Implement parallel processing for faster generation
- Optimize scene complexity to balance quality and performance

### 3. Dataset Diversity
- Implement comprehensive domain randomization
- Create varied scenarios and environments
- Include different lighting conditions and weather

### 4. Annotation Accuracy
- Ensure precise segmentation masks
- Validate bounding box accuracy
- Include 3D pose information where applicable

## Summary

This chapter covered photorealistic simulation and synthetic data generation for humanoid robotics using the NVIDIA Isaac platform. We explored rendering techniques, domain randomization, procedural environment generation, and quality validation methods. Synthetic data generation is essential for training robust perception systems that can operate effectively in real-world conditions.

## Exercises

1. Configure Isaac Sim for photorealistic rendering of a humanoid robot
2. Implement a domain randomization pipeline for synthetic data generation
3. Create a procedural environment generator for indoor scenes
4. Validate synthetic image quality against real-world datasets
5. Implement an efficient parallel data generation pipeline

## Further Reading

- NVIDIA Isaac Sim Synthetic Data Generation Guide
- "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"
- Photorealistic Rendering Techniques for Robotics

---

*Next: [Chapter 3: Isaac ROS Perception Pipelines](../chapter3-perception-pipelines/README.md)*