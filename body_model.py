# body_model.py - 3D Human Body Model for Fashion Design
"""
Parametric 3D human body model for fashion design and fit testing.
Provides realistic body measurements and 3D mesh for garment simulation.
"""

import numpy as np
import trimesh
from typing import Dict, Tuple, List
import math

class HumanBodyModel:
    """Parametric 3D human body model for fashion design"""
    
    def __init__(self, size: str = "M", gender: str = "unisex"):
        self.size = size
        self.gender = gender
        self.measurements = self._create_measurements()
        self.body_mesh = None
        self._generate_body_mesh()
    
    def _create_measurements(self) -> Dict[str, float]:
        """Create realistic body measurements in millimeters"""
        
        # Base measurements for Medium size (unisex)
        base_measurements = {
            # Heights (from floor)
            'total_height': 1700.0,
            'shoulder_height': 1420.0,
            'chest_height': 1280.0,
            'bust_height': 1260.0,  # For dress designs
            'waist_height': 1080.0,
            'hip_height': 920.0,
            'knee_height': 520.0,
            'ankle_height': 80.0,
            'neck_height': 1520.0,
            'armpit_height': 1320.0,
            'hem_height': 600.0,  # Standard jacket hem
            
            # Widths (full circumference converted to half-width for pattern)
            'shoulder_width': 460.0,  # Across shoulders
            'chest_width': 520.0,     # Chest circumference / 2π * π = half width
            'bust_width': 500.0,      # For dresses
            'bust_depth': 250.0,      # Bust depth front to back
            'waist_width': 420.0,     # Waist circumference / 2π * π 
            'hip_width': 550.0,       # Hip circumference / 2π * π
            'neck_width': 180.0,      # Neck circumference / 2π * π
            'arm_length': 620.0,      # Shoulder to wrist
            'sleeve_length': 650.0,   # Including shoulder ease
            
            # Depths (front to back)
            'chest_depth': 240.0,     # Chest depth front to back
            'waist_depth': 200.0,     # Waist depth
            'hip_depth': 260.0,       # Hip depth
            'neck_depth': 140.0,      # Neck depth
            'back_depth': 180.0,      # Back measurement
            'body_depth': 220.0,      # Average body depth
            
            # Additional measurements for construction
            'armhole_depth': 220.0,   # Armhole depth
            'torso_height': 340.0,    # Shoulder to waist
            'back_length': 420.0,     # Neck to waist back
            'front_length': 380.0,    # Neck to waist front
        }
        
        # Adjust for size
        size_multipliers = {
            'XS': 0.9,
            'S': 0.95,
            'M': 1.0,
            'L': 1.05,
            'XL': 1.1,
            'XXL': 1.15
        }
        
        multiplier = size_multipliers.get(self.size, 1.0)
        
        # Apply size scaling (height scales less than width/depth)
        scaled_measurements = {}
        for key, value in base_measurements.items():
            if 'height' in key:
                # Heights scale less
                scaled_measurements[key] = value * (0.95 + 0.05 * multiplier)
            else:
                # Widths and depths scale more
                scaled_measurements[key] = value * multiplier
        
        return scaled_measurements
    
    def get_measurements(self) -> Dict[str, float]:
        """Get all body measurements"""
        return self.measurements.copy()
    
    def get_measurement(self, name: str) -> float:
        """Get specific measurement"""
        return self.measurements.get(name, 0.0)
    
    def _generate_body_mesh(self):
        """Generate 3D body mesh using parametric modeling"""
        vertices = []
        faces = []
        
        # Generate body sections from bottom to top
        sections = self._generate_body_sections()
        
        # Create vertices for each section
        vertex_offset = 0
        for i, section in enumerate(sections):
            section_vertices = self._generate_section_vertices(section)
            vertices.extend(section_vertices)
            
            # Create faces between sections
            if i > 0:
                prev_count = len(sections[i-1]['points'])
                curr_count = len(section['points'])
                
                # Create triangular faces connecting sections
                section_faces = self._create_section_faces(
                    vertex_offset - prev_count, 
                    vertex_offset, 
                    prev_count, 
                    curr_count
                )
                faces.extend(section_faces)
            
            vertex_offset += len(section['points'])
        
        # Convert to numpy arrays
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        # Create trimesh object
        self.body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Smooth the mesh
        self.body_mesh = self.body_mesh.smoothed()
    
    def _generate_body_sections(self) -> List[Dict]:
        """Generate cross-sections of the body at different heights"""
        sections = []
        
        # Define key body sections with their shapes
        section_definitions = [
            # (height, front_width, back_width, front_depth, back_depth, shape_type)
            (0, 80, 80, 80, 80, 'circular'),  # Feet
            (self.measurements['ankle_height'], 100, 100, 100, 100, 'circular'),  # Ankle
            (self.measurements['knee_height'], 150, 140, 120, 110, 'oval'),  # Knee
            (self.measurements['hip_height'], 
             self.measurements['hip_width'], 
             self.measurements['hip_width'] * 0.9,
             self.measurements['hip_depth'] * 0.6,
             self.measurements['hip_depth'] * 0.4, 'body'),  # Hip
            (self.measurements['waist_height'], 
             self.measurements['waist_width'], 
             self.measurements['waist_width'] * 0.85,
             self.measurements['waist_depth'] * 0.5,
             self.measurements['waist_depth'] * 0.5, 'body'),  # Waist
            (self.measurements['chest_height'], 
             self.measurements['chest_width'], 
             self.measurements['chest_width'] * 0.9,
             self.measurements['chest_depth'] * 0.6,
             self.measurements['chest_depth'] * 0.4, 'body'),  # Chest
            (self.measurements['shoulder_height'], 
             self.measurements['shoulder_width'], 
             self.measurements['shoulder_width'] * 0.8,
             self.measurements['chest_depth'] * 0.4,
             self.measurements['back_depth'] * 0.4, 'shoulder'),  # Shoulder
            (self.measurements['neck_height'], 
             self.measurements['neck_width'], 
             self.measurements['neck_width'],
             self.measurements['neck_depth'] * 0.5,
             self.measurements['neck_depth'] * 0.5, 'circular'),  # Neck
        ]
        
        for height, front_width, back_width, front_depth, back_depth, shape_type in section_definitions:
            section = {
                'height': height,
                'front_width': front_width,
                'back_width': back_width,
                'front_depth': front_depth,
                'back_depth': back_depth,
                'shape_type': shape_type,
                'points': []
            }
            
            # Generate points around the perimeter
            section['points'] = self._generate_section_points(section)
            sections.append(section)
        
        return sections
    
    def _generate_section_points(self, section: Dict) -> List[Tuple[float, float]]:
        """Generate points around a body section perimeter"""
        points = []
        num_points = 32  # Points around perimeter
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            
            if section['shape_type'] == 'circular':
                # Simple circle
                radius = section['front_width'] / 2
                x = radius * math.cos(angle)
                z = radius * math.sin(angle)
            
            elif section['shape_type'] == 'oval':
                # Oval shape
                a = section['front_width'] / 2
                b = section['front_depth'] / 2
                x = a * math.cos(angle)
                z = b * math.sin(angle)
            
            elif section['shape_type'] == 'body':
                # Human body shape - more complex curve
                # Front and back have different curvatures
                if -math.pi/2 <= angle <= math.pi/2:
                    # Front half
                    width = section['front_width'] / 2
                    depth = section['front_depth'] / 2
                else:
                    # Back half
                    width = section['back_width'] / 2
                    depth = section['back_depth'] / 2
                
                # Create body-like curve
                cos_angle = math.cos(angle)
                sin_angle = math.sin(angle)
                
                # Modify for body shape
                if abs(cos_angle) > 0.7:  # Side areas
                    x = width * cos_angle
                    z = depth * sin_angle * 0.8  # Flatter sides
                else:  # Front and back
                    x = width * cos_angle * 0.9
                    z = depth * sin_angle
            
            elif section['shape_type'] == 'shoulder':
                # Shoulder shape - wider, flatter
                a = section['front_width'] / 2
                b = section['front_depth'] / 2
                
                cos_angle = math.cos(angle)
                sin_angle = math.sin(angle)
                
                # Flatten top and bottom
                if abs(sin_angle) > 0.8:
                    x = a * cos_angle
                    z = b * sin_angle * 0.6
                else:
                    x = a * cos_angle
                    z = b * sin_angle
            
            points.append((x, z))
        
        return points
    
    def _generate_section_vertices(self, section: Dict) -> List[Tuple[float, float, float]]:
        """Convert 2D section points to 3D vertices"""
        vertices = []
        height = section['height']
        
        for x, z in section['points']:
            vertices.append((x, height, z))
        
        return vertices
    
    def _create_section_faces(self, prev_start: int, curr_start: int, 
                             prev_count: int, curr_count: int) -> List[Tuple[int, int, int]]:
        """Create triangular faces between two body sections"""
        faces = []
        
        # Simple approach: assume same number of points in each section
        # In production, would handle different point counts
        if prev_count == curr_count:
            for i in range(prev_count):
                next_i = (i + 1) % prev_count
                
                # Two triangles per quad
                faces.append((
                    prev_start + i,
                    curr_start + i,
                    curr_start + next_i
                ))
                faces.append((
                    prev_start + i,
                    curr_start + next_i,
                    prev_start + next_i
                ))
        
        return faces
    
    def get_mesh(self) -> trimesh.Trimesh:
        """Get the 3D body mesh"""
        return self.body_mesh
    
    def get_key_points(self) -> Dict[str, Tuple[float, float, float]]:
        """Get key anatomical points for pattern construction"""
        m = self.measurements
        
        return {
            # Center front points
            'center_front_neck': (0, m['neck_height'], m['neck_depth']/2),
            'center_front_chest': (0, m['chest_height'], m['chest_depth']/2),
            'center_front_waist': (0, m['waist_height'], m['waist_depth']/2),
            'center_front_hip': (0, m['hip_height'], m['hip_depth']/2),
            
            # Center back points
            'center_back_neck': (0, m['neck_height'], -m['neck_depth']/2),
            'center_back_chest': (0, m['chest_height'], -m['back_depth']/2),
            'center_back_waist': (0, m['waist_height'], -m['waist_depth']/2),
            'center_back_hip': (0, m['hip_height'], -m['hip_depth']/2),
            
            # Side points
            'side_neck': (m['neck_width']/2, m['neck_height'], 0),
            'side_chest': (m['chest_width']/2, m['chest_height'], 0),
            'side_waist': (m['waist_width']/2, m['waist_height'], 0),
            'side_hip': (m['hip_width']/2, m['hip_height'], 0),
            
            # Shoulder points
            'shoulder_tip': (m['shoulder_width']/2, m['shoulder_height'], 0),
            'shoulder_neck': (m['neck_width']/2, m['shoulder_height'], 0),
            
            # Armhole points
            'armhole_front': (m['shoulder_width']/2, m['armpit_height'], m['chest_depth']/3),
            'armhole_back': (m['shoulder_width']/2, m['armpit_height'], -m['back_depth']/3),
            'armhole_side': (m['shoulder_width']/2, m['armpit_height'], 0),
            
            # Bust points (for dresses)
            'bust_point': (m['bust_width']/2, m['bust_height'], m['chest_depth']/2),
        }
    
    def get_construction_lines(self) -> Dict[str, List[Tuple[float, float, float]]]:
        """Get construction lines for pattern making"""
        key_points = self.get_key_points()
        
        construction_lines = {
            # Center front line
            'center_front': [
                key_points['center_front_neck'],
                key_points['center_front_chest'],
                key_points['center_front_waist'],
                key_points['center_front_hip']
            ],
            
            # Center back line
            'center_back': [
                key_points['center_back_neck'],
                key_points['center_back_chest'],
                key_points['center_back_waist'],
                key_points['center_back_hip']
            ],
            
            # Side seam line
            'side_seam': [
                key_points['armhole_side'],
                key_points['side_chest'],
                key_points['side_waist'],
                key_points['side_hip']
            ],
            
            # Shoulder line
            'shoulder_seam': [
                key_points['shoulder_neck'],
                key_points['shoulder_tip']
            ],
            
            # Armhole curve
            'armhole': [
                key_points['shoulder_tip'],
                key_points['armhole_back'],
                key_points['armhole_side'],
                key_points['armhole_front']
            ]
        }
        
        return construction_lines
    
    def check_garment_fit(self, garment_mesh: trimesh.Trimesh) -> Dict[str, float]:
        """Check how well a garment fits this body"""
        if garment_mesh is None:
            return {'fit_score': 0.0, 'ease': 0.0, 'intersections': 1.0}
        
        # Sample points on body surface
        body_points = self.body_mesh.sample(1000)
        
        # Find closest points on garment
        distances = []
        intersections = 0
        
        for body_point in body_points:
            # Find closest point on garment surface
            closest_point, distance, _ = trimesh.proximity.closest_point(
                garment_mesh, [body_point]
            )
            
            distances.append(distance[0])
            
            # Check for intersections (negative ease)
            if distance[0] < 5.0:  # Less than 5mm ease
                intersections += 1
        
        avg_ease = np.mean(distances)
        intersection_ratio = intersections / len(body_points)
        
        # Calculate fit score
        ideal_ease = 25.0  # 25mm ideal ease
        ease_score = 1.0 - abs(avg_ease - ideal_ease) / ideal_ease
        intersection_penalty = intersection_ratio * 2.0
        
        fit_score = max(0.0, ease_score - intersection_penalty)
        
        return {
            'fit_score': fit_score,
            'average_ease': avg_ease,
            'intersection_ratio': intersection_ratio,
            'min_ease': np.min(distances),
            'max_ease': np.max(distances)
        }
    
    def get_ease_map(self, garment_mesh: trimesh.Trimesh) -> np.ndarray:
        """Generate ease map showing fit quality across the body"""
        if garment_mesh is None:
            return np.array([])
        
        # Sample points systematically across body
        body_bounds = self.body_mesh.bounds
        resolution = 20  # Points per dimension
        
        ease_map = np.zeros((resolution, resolution, resolution))
        
        x_range = np.linspace(body_bounds[0][0], body_bounds[1][0], resolution)
        y_range = np.linspace(body_bounds[0][1], body_bounds[1][1], resolution)
        z_range = np.linspace(body_bounds[0][2], body_bounds[1][2], resolution)
        
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                for k, z in enumerate(z_range):
                    point = np.array([x, y, z])
                    
                    # Check if point is inside or near body
                    body_distance = trimesh.proximity.signed_distance(self.body_mesh, [point])
                    
                    if abs(body_distance[0]) < 50:  # Within 50mm of body surface
                        # Find distance to garment
                        garment_distance = trimesh.proximity.signed_distance(garment_mesh, [point])
                        
                        # Calculate ease (positive = good, negative = intersection)
                        ease = garment_distance[0] - abs(body_distance[0])
                        ease_map[i, j, k] = ease
        
        return ease_map
    
    def visualize_body(self) -> None:
        """Visualize the body model"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot body mesh
        vertices = self.body_mesh.vertices
        faces = self.body_mesh.faces
        
        # Plot sample faces for visualization
        for face in faces[::10]:  # Every 10th face for performance
            triangle = vertices[face]
            ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                           alpha=0.6, color='lightblue')
        
        # Plot key points
        key_points = self.get_key_points()
        for name, point in key_points.items():
            ax.scatter(*point, c='red', s=20)
            ax.text(point[0], point[1], point[2], name, fontsize=8)
        
        # Plot construction lines
        construction_lines = self.get_construction_lines()
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        
        for i, (name, line) in enumerate(construction_lines.items()):
            line_array = np.array(line)
            ax.plot(line_array[:, 0], line_array[:, 1], line_array[:, 2], 
                   color=colors[i % len(colors)], linewidth=2, label=name)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Human Body Model - Size {self.size}')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = 300
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([0, 1800])
        ax.set_zlim([-max_range, max_range])
        
        plt.tight_layout()
        plt.show()
    
    def export_measurements(self, filename: str):
        """Export measurements to file for pattern making software"""
        import json
        
        export_data = {
            'size': self.size,
            'gender': self.gender,
            'measurements': self.measurements,
            'key_points': {name: list(point) for name, point in self.get_key_points().items()},
            'construction_lines': {
                name: [list(point) for point in line] 
                for name, line in self.get_construction_lines().items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Body measurements exported to {filename}")

# Example usage and testing
if __name__ == "__main__":
    # Create different body sizes
    sizes = ['S', 'M', 'L']
    
    for size in sizes:
        print(f"\n=== Size {size} Body Model ===")
        body = HumanBodyModel(size=size)
        
        measurements = body.get_measurements()
        print(f"Chest circumference: {measurements['chest_width'] * 2:.0f}mm")
        print(f"Waist circumference: {measurements['waist_width'] * 2:.0f}mm")
        print(f"Hip circumference: {measurements['hip_width'] * 2:.0f}mm")
        print(f"Height: {measurements['total_height']:.0f}mm")
        
        # Export measurements
        body.export_measurements(f"body_measurements_{size}.json")
    
    # Visualize medium body
    body_m = HumanBodyModel(size='M')
    body_m.visualize_body()