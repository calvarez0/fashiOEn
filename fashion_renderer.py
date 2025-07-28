# fashion_renderer.py - Enhanced 3D Fashion Visualization with Beautiful Dresses
"""
Enhanced 3D visualization for fashion designs with beautiful, realistic dress rendering.
Creates stunning dress visualizations that evolve through generations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import math
import random

@dataclass
class RenderSettings:
    """Settings for 3D rendering"""
    resolution: Tuple[int, int] = (1200, 900)
    background_color: str = 'white'
    lighting_intensity: float = 1.0
    show_wireframe: bool = False
    show_body: bool = True
    garment_opacity: float = 0.85
    body_opacity: float = 0.3
    dress_quality: str = 'high'  # 'low', 'medium', 'high'

class EnhancedFashion3DRenderer:
    """Enhanced 3D renderer for beautiful fashion visualization"""
    
    def __init__(self, settings: Optional[RenderSettings] = None):
        self.settings = settings or RenderSettings()
        
        # Predefined beautiful dress styles
        self.dress_styles = {
            'a_line': {'waist_factor': 0.7, 'hem_flare': 1.8, 'curves': 'smooth'},
            'fit_and_flare': {'waist_factor': 0.6, 'hem_flare': 2.0, 'curves': 'dramatic'},
            'straight': {'waist_factor': 0.9, 'hem_flare': 1.1, 'curves': 'minimal'},
            'mermaid': {'waist_factor': 0.5, 'hem_flare': 1.6, 'curves': 'body_hugging'},
            'ball_gown': {'waist_factor': 0.6, 'hem_flare': 2.5, 'curves': 'fairy_tale'}
        }
        
        # Color palettes for beautiful dresses
        self.color_palettes = {
            'elegant': [(0.2, 0.1, 0.4), (0.1, 0.2, 0.5), (0.4, 0.1, 0.3)],  # Deep purples/blues
            'romantic': [(1.0, 0.8, 0.9), (0.9, 0.7, 0.8), (1.0, 0.9, 0.95)],  # Soft pinks
            'sophisticated': [(0.15, 0.15, 0.15), (0.3, 0.3, 0.3), (0.1, 0.1, 0.2)],  # Blacks/grays
            'vibrant': [(0.8, 0.2, 0.4), (0.2, 0.7, 0.3), (0.9, 0.5, 0.1)],  # Bold colors
            'sunset': [(1.0, 0.6, 0.2), (1.0, 0.4, 0.3), (0.9, 0.3, 0.4)],  # Warm oranges/reds
            'ocean': [(0.1, 0.4, 0.7), (0.2, 0.6, 0.8), (0.0, 0.3, 0.6)],  # Blues
            'forest': [(0.2, 0.5, 0.3), (0.3, 0.6, 0.2), (0.1, 0.4, 0.2)]   # Greens
        }
    
    def create_beautiful_dress(self, design_params: Dict, body_measurements: Dict, design_id: int = 1) -> Dict:
        """Create a beautiful dress based on design parameters"""
        
        # Extract or generate dress characteristics
        fitness = design_params.get('fitness', random.uniform(0.3, 0.9))
        generation = design_params.get('generation', 0)
        
        # Choose dress style based on fitness (higher fitness = more elegant styles)
        if fitness > 0.8:
            style_name = random.choice(['ball_gown', 'mermaid', 'fit_and_flare'])
            palette_name = random.choice(['elegant', 'romantic', 'sophisticated'])
        elif fitness > 0.6:
            style_name = random.choice(['a_line', 'fit_and_flare'])
            palette_name = random.choice(['romantic', 'vibrant', 'sunset'])
        else:
            style_name = random.choice(['straight', 'a_line'])
            palette_name = random.choice(['vibrant', 'ocean', 'forest'])
        
        style = self.dress_styles[style_name]
        colors = self.color_palettes[palette_name]
        
        # Create dress geometry
        dress_geometry = self._create_dress_geometry(body_measurements, style, fitness)
        
        # Add design variations based on construction parameters
        if 'construction_params' in design_params:
            dress_geometry = self._apply_construction_variations(dress_geometry, design_params['construction_params'])
        
        return {
            'geometry': dress_geometry,
            'style': style_name,
            'colors': colors,
            'fitness': fitness,
            'generation': generation,
            'design_id': design_id,
            'details': self._generate_dress_details(style, fitness)
        }
    
    def _create_dress_geometry(self, measurements: Dict, style: Dict, fitness: float) -> Dict:
        """Create detailed dress geometry"""
        
        # Base measurements
        shoulder_width = measurements.get('shoulder_width', 460)
        bust_width = measurements.get('bust_width', 500) 
        waist_width = measurements.get('waist_width', 420)
        hip_width = measurements.get('hip_width', 550)
        
        # Heights
        shoulder_height = measurements.get('shoulder_height', 1420)
        bust_height = measurements.get('bust_height', 1260)
        waist_height = measurements.get('waist_height', 1080)
        hip_height = measurements.get('hip_height', 920)
        hem_height = hip_height - 400  # Knee-length base
        
        # Apply style factors
        waist_factor = style['waist_factor']
        hem_flare = style['hem_flare']
        
        # Calculate dress measurements (including ease and style modifications)
        dress_shoulder = shoulder_width * 0.9  # Slightly off-shoulder
        dress_bust = bust_width + 60 + (fitness * 40)  # More fitted with higher fitness
        dress_waist = waist_width * waist_factor + 40
        dress_hip = hip_width + 80
        dress_hem = dress_hip * hem_flare
        
        # Create sophisticated dress silhouette with multiple control points
        front_points = self._create_dress_front_silhouette(
            dress_shoulder, dress_bust, dress_waist, dress_hip, dress_hem,
            shoulder_height, bust_height, waist_height, hip_height, hem_height,
            style, fitness
        )
        
        back_points = self._create_dress_back_silhouette(
            front_points, measurements, style, fitness
        )
        
        # Create 3D surface triangles
        triangles = self._create_dress_surface_triangles(front_points, back_points)
        
        return {
            'front_points': front_points,
            'back_points': back_points,
            'triangles': triangles,
            'measurements': {
                'shoulder': dress_shoulder,
                'bust': dress_bust, 
                'waist': dress_waist,
                'hip': dress_hip,
                'hem': dress_hem
            }
        }
    
    def _create_dress_front_silhouette(self, shoulder_w, bust_w, waist_w, hip_w, hem_w,
                                     shoulder_h, bust_h, waist_h, hip_h, hem_h, 
                                     style, fitness):
        """Create sophisticated front dress silhouette"""
        
        # Depth values for 3D effect
        neckline_depth = 25 + (fitness * 15)  # Deeper neckline with higher fitness
        bust_depth = 35 + (fitness * 20)
        waist_depth = 20 + (fitness * 10)
        hip_depth = 25
        hem_depth = 20
        
        # Create neckline (varies by style and fitness)
        neckline_height = shoulder_h + 20
        neckline_width = 40 + (fitness * 30)  # Wider neckline for higher fitness
        
        points = []
        
        # Neckline - create elegant curve
        neckline_points = self._create_neckline_curve(neckline_width, neckline_height, neckline_depth)
        points.extend(neckline_points)
        
        # Shoulder line - graceful slope
        points.append((shoulder_w/2, shoulder_h - 10, bust_depth * 0.8))
        
        # Armhole - elegant curve
        armhole_points = self._create_armhole_curve(shoulder_w/2, shoulder_h, bust_h, bust_w, bust_depth)
        points.extend(armhole_points)
        
        # Bust area - enhanced for higher fitness
        bust_enhancement = fitness * 0.3
        points.append((bust_w/2 * (1 + bust_enhancement), bust_h, bust_depth))
        
        # Waist - create cinched effect
        waist_curve_points = self._create_waist_curve(bust_w, waist_w, bust_h, waist_h, waist_depth, style)
        points.extend(waist_curve_points)
        
        # Hip area - smooth transition
        hip_curve_points = self._create_hip_curve(waist_w, hip_w, waist_h, hip_h, hip_depth, style)
        points.extend(hip_curve_points)
        
        # Hem - final flare
        hem_curve_points = self._create_hem_curve(hip_w, hem_w, hip_h, hem_h, hem_depth, style, fitness)
        points.extend(hem_curve_points)
        
        # Close to center front
        points.append((0, hem_h, hem_depth))
        points.append((0, neckline_height, neckline_depth))
        
        return points
    
    def _create_neckline_curve(self, width, height, depth):
        """Create elegant neckline curve"""
        points = []
        
        # Center front neckline
        points.append((0, height - 30, depth))  # Slight scoop
        
        # Curved neckline
        num_points = 5
        for i in range(1, num_points):
            t = i / num_points
            x = width * t * t  # Quadratic curve for elegance
            y = height - 30 + (t * 20)  # Slight rise
            z = depth * (1 - t * 0.3)  # Slight curve back
            points.append((x, y, z))
        
        return points
    
    def _create_armhole_curve(self, shoulder_x, shoulder_y, bust_y, bust_x, depth):
        """Create elegant armhole curve"""
        points = []
        
        # Armhole depth
        armhole_bottom = bust_y + 40
        
        # Create smooth armhole curve
        num_points = 4
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # Parametric curve for natural armhole shape
            x = shoulder_x + (bust_x/2 - shoulder_x) * (t * t)
            y = shoulder_y + (armhole_bottom - shoulder_y) * t
            z = depth * (0.8 + 0.2 * math.sin(t * math.pi))  # Slight 3D curve
            
            points.append((x, y, z))
        
        return points
    
    def _create_waist_curve(self, bust_w, waist_w, bust_h, waist_h, depth, style):
        """Create waist curve based on style"""
        points = []
        
        # Number of curve points based on style
        if style['curves'] == 'dramatic':
            num_points = 6
            curve_factor = 1.3
        elif style['curves'] == 'smooth':
            num_points = 4
            curve_factor = 1.1
        else:
            num_points = 3
            curve_factor = 1.05
        
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # Smooth transition from bust to waist
            width = bust_w/2 + (waist_w/2 - bust_w/2) * (t * curve_factor)
            height = bust_h + (waist_h - bust_h) * t
            z = depth * (1 + 0.1 * math.sin(t * math.pi))  # Slight curve
            
            points.append((width, height, z))
        
        return points
    
    def _create_hip_curve(self, waist_w, hip_w, waist_h, hip_h, depth, style):
        """Create hip curve"""
        points = []
        
        num_points = 3
        for i in range(num_points):
            t = i / (num_points - 1)
            
            width = waist_w/2 + (hip_w/2 - waist_w/2) * t
            height = waist_h + (hip_h - waist_h) * t
            z = depth
            
            points.append((width, height, z))
        
        return points
    
    def _create_hem_curve(self, hip_w, hem_w, hip_h, hem_h, depth, style, fitness):
        """Create hem curve with style-specific flare"""
        points = []
        
        # Flare characteristics
        hem_flare = style['hem_flare']
        
        if style['curves'] == 'fairy_tale':  # Ball gown
            num_points = 8
            # Dramatic flare with curve
            for i in range(num_points):
                t = i / (num_points - 1)
                
                # Exponential flare for ball gown effect
                width_factor = 1 + (hem_flare - 1) * (t * t)
                width = hip_w/2 * width_factor
                height = hip_h + (hem_h - hip_h) * t
                
                # Add curve for fullness
                z = depth + 30 * math.sin(t * math.pi) * fitness
                
                points.append((width, height, z))
                
        elif style['curves'] == 'body_hugging':  # Mermaid
            num_points = 6
            for i in range(num_points):
                t = i / (num_points - 1)
                
                if t < 0.7:  # Tight until knee area
                    width = hip_w/2 * (1 + 0.1 * t)
                else:  # Then flare dramatically
                    flare_start = 0.7
                    flare_t = (t - flare_start) / (1 - flare_start)
                    width = hip_w/2 * (1.1 + (hem_flare - 1.1) * (flare_t * flare_t * flare_t))
                
                height = hip_h + (hem_h - hip_h) * t
                z = depth
                
                points.append((width, height, z))
        else:  # A-line and other styles
            num_points = 4
            for i in range(num_points):
                t = i / (num_points - 1)
                
                width = hip_w/2 + (hem_w/2 - hip_w/2) * t
                height = hip_h + (hem_h - hip_h) * t
                z = depth
                
                points.append((width, height, z))
        
        return points
    
    def _create_dress_back_silhouette(self, front_points, measurements, style, fitness):
        """Create back dress silhouette"""
        back_points = []
        
        back_depth_offset = -measurements.get('back_depth', 180) / 2 - 20
        
        for point in front_points:
            x, y, z = point
            # Mirror to back with slight modifications
            back_x = x * 0.95  # Slightly narrower back
            back_y = y
            back_z = back_depth_offset + z * 0.3  # Move to back
            
            back_points.append((back_x, back_y, back_z))
        
        return back_points
    
    def _create_dress_surface_triangles(self, front_points, back_points):
        """Create triangular mesh for dress surface"""
        triangles = []
        
        # Create triangles between front and back
        min_points = min(len(front_points), len(back_points))
        
        # Front surface triangles
        for i in range(len(front_points) - 2):
            triangle = [front_points[i], front_points[i+1], front_points[i+2]]
            triangles.append(triangle)
        
        # Back surface triangles  
        for i in range(len(back_points) - 2):
            triangle = [back_points[i], back_points[i+1], back_points[i+2]]
            triangles.append(triangle)
        
        # Side triangles connecting front and back
        for i in range(min_points - 1):
            # Right side
            if i < len(front_points) - 1 and i < len(back_points) - 1:
                # Two triangles per quad
                tri1 = [front_points[i], back_points[i], front_points[i+1]]
                tri2 = [back_points[i], back_points[i+1], front_points[i+1]]
                triangles.extend([tri1, tri2])
        
        return triangles
    
    def _apply_construction_variations(self, geometry, construction_params):
        """Apply construction parameter variations to dress"""
        
        # Apply ease variations
        if hasattr(construction_params, 'ease_amounts'):
            ease = construction_params.ease_amounts
            
            # Modify dress measurements based on ease
            for point_list in [geometry['front_points'], geometry['back_points']]:
                for i, (x, y, z) in enumerate(point_list):
                    # Adjust width based on height level
                    if 'waist' in ease:
                        waist_height = 1080  # approximate
                        if abs(y - waist_height) < 100:  # Near waist
                            adjustment = ease['waist'] / 1000  # Convert to factor
                            point_list[i] = (x * (1 + adjustment), y, z)
        
        return geometry
    
    def _generate_dress_details(self, style, fitness):
        """Generate dress details based on style and fitness"""
        details = {
            'neckline_style': 'scoop',
            'sleeve_style': 'sleeveless',
            'waist_treatment': 'fitted',
            'hem_style': 'straight',
            'embellishments': []
        }
        
        # Higher fitness gets more elegant details
        if fitness > 0.8:
            details['embellishments'] = ['beading', 'elegant_draping']
            details['neckline_style'] = 'sweetheart'
        elif fitness > 0.6:
            details['embellishments'] = ['subtle_texture']
            details['waist_treatment'] = 'belted'
        
        return details
    
    def render_beautiful_dress(self, ax, dress_data, body_measurements, show_mannequin=True):
        """Render a beautiful dress on the axes"""
        
        # Draw mannequin if requested
        if show_mannequin:
            self._draw_elegant_mannequin(ax, body_measurements)
        
        # Get dress geometry and colors
        geometry = dress_data['geometry']
        colors = dress_data['colors']
        style = dress_data['style']
        fitness = dress_data['fitness']
        
        # Choose color based on fitness
        color_index = min(int(fitness * len(colors)), len(colors) - 1)
        primary_color = colors[color_index]
        accent_color = [c * 0.8 for c in primary_color]  # Darker accent
        
        try:
            # Render dress triangles with beautiful shading
            triangles = geometry['triangles']
            
            if triangles:
                # Create main dress surface
                dress_collection = Poly3DCollection(triangles, alpha=self.settings.garment_opacity, linewidths=0.5)
                dress_collection.set_facecolor(primary_color)
                dress_collection.set_edgecolor(accent_color)
                ax.add_collection3d(dress_collection)
                
                # Add highlights for premium effect
                if fitness > 0.7:
                    self._add_dress_highlights(ax, geometry, primary_color, fitness)
            
            # Add dress details
            self._add_dress_details(ax, dress_data, body_measurements)
            
        except Exception as e:
            print(f"   Advanced dress rendering failed: {e}")
            # Fallback to simple but elegant rendering
            self._render_simple_elegant_dress(ax, dress_data, body_measurements)
    
    def _draw_elegant_mannequin(self, ax, measurements):
        """Draw an elegant fashion mannequin"""
        
        # Head - more refined
        head_y = measurements.get('neck_height', 1520) + 120
        head_size = 300
        ax.scatter(0, head_y, 0, c='bisque', s=head_size, alpha=0.9, 
                  marker='o', edgecolors='tan', linewidth=1)
        
        # Neck - graceful
        neck_start = measurements.get('neck_height', 1520)
        neck_points = np.array([[0, neck_start, 0], [0, head_y - 40, 0]])
        ax.plot(neck_points[:, 0], neck_points[:, 1], neck_points[:, 2], 
               color='bisque', linewidth=12, alpha=0.9, solid_capstyle='round')
        
        # Body outline - elegant proportions
        shoulder_w = measurements.get('shoulder_width', 460) / 2
        bust_w = measurements.get('bust_width', 500) / 2
        waist_w = measurements.get('waist_width', 420) / 2
        hip_w = measurements.get('hip_width', 550) / 2
        
        shoulder_h = measurements.get('shoulder_height', 1420)
        bust_h = measurements.get('bust_height', 1260)
        waist_h = measurements.get('waist_height', 1080)
        hip_h = measurements.get('hip_height', 920)
        
        # Right side silhouette
        body_right = np.array([
            [shoulder_w, shoulder_h, 0],
            [bust_w, bust_h, 0],
            [waist_w, waist_h, 0],
            [hip_w, hip_h, 0],
            [hip_w * 0.6, hip_h - 300, 0]  # Legs
        ])
        
        # Left side silhouette
        body_left = np.array([
            [-shoulder_w, shoulder_h, 0],
            [-bust_w, bust_h, 0], 
            [-waist_w, waist_h, 0],
            [-hip_w, hip_h, 0],
            [-hip_w * 0.6, hip_h - 300, 0]
        ])
        
        # Draw body with elegant lines
        ax.plot(body_right[:, 0], body_right[:, 1], body_right[:, 2], 
               color='tan', linewidth=4, alpha=0.8, solid_capstyle='round')
        ax.plot(body_left[:, 0], body_left[:, 1], body_left[:, 2], 
               color='tan', linewidth=4, alpha=0.8, solid_capstyle='round')
        
        # Arms - graceful positioning
        # Right arm
        right_arm = np.array([
            [shoulder_w, shoulder_h, 0],
            [shoulder_w + 100, shoulder_h - 150, -30],
            [shoulder_w + 80, waist_h + 50, -20]
        ])
        ax.plot(right_arm[:, 0], right_arm[:, 1], right_arm[:, 2], 
               color='tan', linewidth=5, alpha=0.7, solid_capstyle='round')
        
        # Left arm
        left_arm = np.array([
            [-shoulder_w, shoulder_h, 0],
            [-shoulder_w - 100, shoulder_h - 150, -30],
            [-shoulder_w - 80, waist_h + 50, -20]
        ])
        ax.plot(left_arm[:, 0], left_arm[:, 1], left_arm[:, 2], 
               color='tan', linewidth=5, alpha=0.7, solid_capstyle='round')
    
    def _add_dress_highlights(self, ax, geometry, color, fitness):
        """Add highlights and details for premium dresses"""
        
        # Add subtle highlights along seam lines
        front_points = geometry['front_points']
        
        # Waist highlight
        waist_points = [p for p in front_points if 1000 < p[1] < 1150]  # Waist area
        if len(waist_points) >= 2:
            waist_line = np.array(waist_points[:2])
            ax.plot(waist_line[:, 0], waist_line[:, 1], waist_line[:, 2], 
                   color=[min(c + 0.3, 1.0) for c in color], linewidth=3, alpha=0.9)
        
        # Neckline highlight
        neck_points = [p for p in front_points if p[1] > 1400]  # Neckline area
        if len(neck_points) >= 2:
            neck_line = np.array(neck_points[:3])
            ax.plot(neck_line[:, 0], neck_line[:, 1], neck_line[:, 2], 
                   color=[min(c + 0.4, 1.0) for c in color], linewidth=2, alpha=0.8)
    
    def _add_dress_details(self, ax, dress_data, measurements):
        """Add dress details like belts, trim, etc."""
        
        details = dress_data['details']
        geometry = dress_data['geometry']
        colors = dress_data['colors']
        fitness = dress_data['fitness']
        
        # Add waist belt for certain styles
        if details['waist_treatment'] == 'belted' and fitness > 0.5:
            waist_height = measurements.get('waist_height', 1080)
            waist_width = geometry['measurements']['waist']
            
            # Belt line
            belt_points = np.array([
                [-waist_width/2, waist_height, 25],
                [waist_width/2, waist_height, 25]
            ])
            
            belt_color = [c * 0.5 for c in colors[0]]  # Darker belt
            ax.plot(belt_points[:, 0], belt_points[:, 1], belt_points[:, 2], 
                   color=belt_color, linewidth=8, alpha=0.9, solid_capstyle='round')
        
        # Add embellishments
        if 'beading' in details['embellishments']:
            self._add_beading_effect(ax, geometry, colors, fitness)
    
    def _add_beading_effect(self, ax, geometry, colors, fitness):
        """Add beading effect for high-fitness dresses"""
        
        front_points = geometry['front_points']
        
        # Add sparkle points along neckline and waist
        sparkle_color = [min(c + 0.5, 1.0) for c in colors[0]]
        
        for point in front_points[::3]:  # Every 3rd point
            if point[1] > 1200:  # Upper bodice area
                # Small sparkle
                ax.scatter(point[0], point[1], point[2], 
                          c=[sparkle_color], s=20, alpha=0.8, marker='*')
    
    def _render_simple_elegant_dress(self, ax, dress_data, measurements):
        """Fallback elegant dress rendering"""
        
        colors = dress_data['colors']
        fitness = dress_data['fitness']
        
        # Simple but elegant dress shape
        shoulder_w = measurements.get('shoulder_width', 460) / 2
        bust_w = measurements.get('bust_width', 500) / 2 + 40
        waist_w = measurements.get('waist_width', 420) / 2 + 20
        hip_w = measurements.get('hip_width', 550) / 2 + 60
        hem_w = hip_w * (1.2 + fitness * 0.3)
        
        shoulder_h = measurements.get('shoulder_height', 1420)
        bust_h = measurements.get('bust_height', 1260)
        waist_h = measurements.get('waist_height', 1080)
        hip_h = measurements.get('hip_height', 920)
        hem_h = hip_h - 350
        
        # Front silhouette
        front_dress = np.array([
            [0, shoulder_h + 10, 30],  # Neckline
            [shoulder_w * 0.8, shoulder_h, 30],
            [bust_w, bust_h, 35],
            [waist_w, waist_h, 25],
            [hip_w, hip_h, 25],
            [hem_w, hem_h, 20],
            [0, hem_h, 20]
        ])
        
        # Back silhouette
        back_dress = front_dress.copy()
        back_dress[:, 2] = -back_dress[:, 2] - 40  # Move to back
        back_dress[:, 0] *= 0.95  # Slightly narrower
        
        # Draw dress as filled polygons
        color = colors[min(int(fitness * len(colors)), len(colors) - 1)]
        
        # Front dress surface
        ax.plot(front_dress[:, 0], front_dress[:, 1], front_dress[:, 2], 
               color=color, linewidth=6, alpha=0.9, solid_capstyle='round')
        ax.fill(front_dress[:, 0], front_dress[:, 1], front_dress[:, 2], 
               color=color, alpha=0.7)
        
        # Back dress surface
        back_color = [c * 0.8 for c in color]
        ax.plot(back_dress[:, 0], back_dress[:, 1], back_dress[:, 2], 
               color=back_color, linewidth=5, alpha=0.8, solid_capstyle='round')
        
        # Connect sides for 3D effect
        for i in range(len(front_dress) - 1):
            side_line = np.array([front_dress[i], back_dress[i]])
            ax.plot(side_line[:, 0], side_line[:, 1], side_line[:, 2], 
                   color=[c * 0.9 for c in color], linewidth=2, alpha=0.6)
    
    def render_design_comparison(self, designs: List[Dict], 
                               body_model, save_path: Optional[str] = None):
        """Render multiple beautiful designs for comparison"""
        
        n_designs = min(len(designs), 9)  # Show up to 9 designs
        cols = 3
        rows = (n_designs + cols - 1) // cols
        
        fig = plt.figure(figsize=(16, 6 * rows))
        fig.patch.set_facecolor('white')
        
        body_measurements = body_model.get_measurements()
        
        for i in range(n_designs):
            design = designs[i]
            
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            ax.set_facecolor('white')
            
            # Create beautiful dress from design data
            dress_data = self.create_beautiful_dress(design, body_measurements, i + 1)
            
            # Render the dress
            self.render_beautiful_dress(ax, dress_data, body_measurements)
            
            # Setup axes for best viewing
            self._setup_dress_axes(ax, body_measurements)
            
            # Title with design info
            fitness = design.get('fitness', 0)
            generation = design.get('generation', 0)
            style = dress_data['style']
            
            ax.set_title(f'Design {i+1} - {style.replace("_", " ").title()}\n'
                        f'Gen: {generation}, Fitness: {fitness:.3f}', 
                        fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle('3D Fashion Evolution - Beautiful Dress Collection', 
                     fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        plt.show()
        
        return fig
    
    def _setup_dress_axes(self, ax, measurements):
        """Setup 3D axes optimized for dress viewing"""
        
        # Set limits for optimal dress viewing
        max_range = 500
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([600, 1600])  # Focus on dress area
        ax.set_zlim([-200, 200])
        
        # Set viewing angle for best dress presentation
        ax.view_init(elev=15, azim=30)
        
        # Clean up axes for fashion presentation
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        
        # Hide grid and ticks for cleaner look
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Make axes less prominent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make pane edges lighter
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
    
    def create_evolution_showcase(self, generations_data: List[List[Dict]], 
                                 body_model, save_path: str):
        """Create a beautiful showcase of evolution progress"""
        
        if not generations_data:
            return
        
        # Create figure showing evolution progression
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
        fig.patch.set_facecolor('white')
        
        body_measurements = body_model.get_measurements()
        
        # Show best dress from different generations
        generation_indices = [0, len(generations_data)//4, len(generations_data)//2, 
                            3*len(generations_data)//4, len(generations_data)-1]
        
        for idx, gen_idx in enumerate(generation_indices):
            if gen_idx < len(generations_data) and idx < 5:
                generation_designs = generations_data[gen_idx]
                
                if generation_designs:
                    # Get best design from this generation
                    best_design = max(generation_designs, key=lambda d: d.get('fitness', 0))
                    
                    # Determine subplot position
                    row = idx // 3
                    col = idx % 3
                    ax = axes[row, col]
                    
                    # Create and render beautiful dress
                    dress_data = self.create_beautiful_dress(best_design, body_measurements)
                    self.render_beautiful_dress(ax, dress_data, body_measurements)
                    
                    # Setup axes
                    self._setup_dress_axes(ax, body_measurements)
                    
                    # Title
                    fitness = best_design.get('fitness', 0)
                    ax.set_title(f'Generation {gen_idx + 1}\n'
                               f'Best Fitness: {fitness:.3f}', 
                               fontsize=12, fontweight='bold')
        
        # Use last subplot for evolution summary
        summary_ax = axes[1, 2]
        summary_ax.remove()  # Remove 3D axes
        summary_ax = fig.add_subplot(2, 3, 6)  # Add 2D axes
        
        # Plot fitness evolution
        gen_numbers = list(range(1, len(generations_data) + 1))
        best_fitnesses = []
        avg_fitnesses = []
        
        for generation in generations_data:
            if generation:
                fitnesses = [d.get('fitness', 0) for d in generation]
                best_fitnesses.append(max(fitnesses))
                avg_fitnesses.append(sum(fitnesses) / len(fitnesses))
            else:
                best_fitnesses.append(0)
                avg_fitnesses.append(0)
        
        summary_ax.plot(gen_numbers, best_fitnesses, 'r-', linewidth=3, 
                       marker='o', markersize=8, label='Best Fitness')
        summary_ax.plot(gen_numbers, avg_fitnesses, 'b-', linewidth=2, 
                       marker='s', markersize=6, label='Average Fitness')
        
        summary_ax.set_xlabel('Generation', fontsize=12)
        summary_ax.set_ylabel('Fitness Score', fontsize=12)
        summary_ax.set_title('Evolution Progress', fontsize=14, fontweight='bold')
        summary_ax.legend()
        summary_ax.grid(True, alpha=0.3)
        summary_ax.set_ylim(0, 1)
        
        plt.suptitle('Fashion Evolution Showcase - From Basic to Beautiful', 
                     fontsize=20, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        return fig
    
    def render_pattern_layout_beautiful(self, patterns: List[Dict], 
                                      fabric_width: float = 1500,
                                      save_path: Optional[str] = None):
        """Render beautiful pattern layout with fashion illustration style"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        fig.patch.set_facecolor('white')
        
        # Left: Pattern layout
        current_x = 50
        current_y = 50
        row_height = 0
        
        # Beautiful color palette for patterns
        pattern_colors = [
            '#FF6B9D',  # Pink
            '#4ECDC4',  # Teal
            '#45B7D1',  # Blue
            '#96CEB4',  # Green
            '#FECA57',  # Yellow
            '#FF9FF3',  # Magenta
            '#54A0FF',  # Light blue
            '#5F27CD'   # Purple
        ]
        
        for i, pattern in enumerate(patterns):
            if 'vertices' not in pattern or len(pattern['vertices']) < 3:
                continue
            
            vertices = np.array(pattern['vertices'])
            color = pattern_colors[i % len(pattern_colors)]
            
            # Calculate pattern dimensions
            pattern_width = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
            pattern_height = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
            
            # Check if pattern fits in current row
            if current_x + pattern_width > fabric_width - 50:
                current_x = 50
                current_y += row_height + 50
                row_height = 0
            
            # Translate pattern to layout position
            offset_x = current_x - np.min(vertices[:, 0])
            offset_y = current_y - np.min(vertices[:, 1])
            layout_vertices = vertices + [offset_x, offset_y]
            
            # Draw pattern with beautiful styling
            layout_vertices_closed = np.vstack([layout_vertices, layout_vertices[0]])
            ax1.plot(layout_vertices_closed[:, 0], layout_vertices_closed[:, 1], 
                    color=color, linewidth=3, alpha=0.9)
            ax1.fill(layout_vertices[:, 0], layout_vertices[:, 1], 
                    color=color, alpha=0.4)
            
            # Add pattern name with elegant typography
            center_x = np.mean(layout_vertices[:, 0])
            center_y = np.mean(layout_vertices[:, 1])
            ax1.text(center_x, center_y, pattern['name'], 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             alpha=0.8, edgecolor=color))
            
            current_x += pattern_width + 50
            row_height = max(row_height, pattern_height)
        
        # Fabric boundary with elegant styling
        fabric_height = current_y + row_height + 50
        ax1.plot([0, fabric_width, fabric_width, 0, 0], 
                [0, 0, fabric_height, fabric_height, 0], 
                'k-', linewidth=4, alpha=0.8)
        
        ax1.set_xlim(-50, fabric_width + 50)
        ax1.set_ylim(-50, fabric_height + 50)
        ax1.set_aspect('equal')
        ax1.set_title('Pattern Layout', fontsize=16, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.2)
        
        # Right: 3D visualization of how patterns become dress
        ax2.remove()
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        # Show a beautiful dress made from these patterns
        dummy_measurements = {
            'shoulder_width': 460, 'bust_width': 500, 'waist_width': 420,
            'hip_width': 550, 'shoulder_height': 1420, 'bust_height': 1260,
            'waist_height': 1080, 'hip_height': 920, 'neck_height': 1520
        }
        
        dummy_design = {'fitness': 0.8, 'generation': 1}
        dress_data = self.create_beautiful_dress(dummy_design, dummy_measurements)
        self.render_beautiful_dress(ax2, dress_data, dummy_measurements)
        self._setup_dress_axes(ax2, dummy_measurements)
        
        ax2.set_title('3D Dress Result', fontsize=16, fontweight='bold', pad=20)
        
        # Calculate and display efficiency
        total_pattern_area = sum(pattern.get('area', 0) for pattern in patterns)
        fabric_area = fabric_width * fabric_height
        efficiency = (total_pattern_area / fabric_area) * 100 if fabric_area > 0 else 0
        
        # Add efficiency info
        efficiency_text = f"Layout Efficiency: {efficiency:.1f}%\n"
        efficiency_text += f"Fabric: {fabric_width:.0f}mm × {fabric_height:.0f}mm\n"
        efficiency_text += f"Patterns: {len(patterns)} pieces"
        
        fig.text(0.5, 0.02, efficiency_text, ha='center', fontsize=12, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('From 2D Patterns to 3D Fashion', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        plt.show()
        
        return {
            'efficiency': efficiency,
            'fabric_dimensions': (fabric_width, fabric_height),
            'total_area': fabric_area,
            'pattern_area': total_pattern_area
        }

# Example usage and integration
if __name__ == "__main__":
    from body_model import HumanBodyModel
    
    # Create body model and renderer
    body = HumanBodyModel(size='M')
    renderer = EnhancedFashion3DRenderer()
    
    # Create sample designs
    sample_designs = [
        {'fitness': 0.9, 'generation': 5, 'construction_params': None},
        {'fitness': 0.7, 'generation': 3, 'construction_params': None},
        {'fitness': 0.5, 'generation': 1, 'construction_params': None},
    ]
    
    # Render beautiful comparison
    renderer.render_design_comparison(sample_designs, body, "beautiful_dresses.png")
    
    print("Enhanced Fashion Renderer ready for beautiful dress generation!")