# fashion_renderer.py - 3D Fashion Visualization and Rendering
"""
Advanced 3D visualization for fashion designs with professional rendering capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RenderSettings:
    """Settings for 3D rendering"""
    resolution: Tuple[int, int] = (800, 600)
    background_color: str = 'white'
    lighting_intensity: float = 1.0
    show_wireframe: bool = False
    show_body: bool = True
    garment_opacity: float = 0.8
    body_opacity: float = 0.3

class Fashion3DRenderer:
    """Advanced 3D renderer for fashion visualization"""
    
    def __init__(self, settings: Optional[RenderSettings] = None):
        self.settings = settings or RenderSettings()
    
    def render_design_comparison(self, designs: List[Dict], 
                               body_model, save_path: Optional[str] = None):
        """Render multiple designs for comparison"""
        
        n_designs = len(designs)
        cols = min(4, n_designs)
        rows = (n_designs + cols - 1) // cols
        
        fig = plt.figure(figsize=(5 * cols, 5 * rows))
        
        for i, design in enumerate(designs):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            
            # Render single design
            self._render_single_design(ax, design, body_model)
            
            fitness = design.get('fitness', 0)
            generation = design.get('generation', 0)
            ax.set_title(f'Design {i+1}\nGen: {generation}, Fit: {fitness:.3f}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def _render_single_design(self, ax, design, body_model):
        """Render a single design on given axes"""
        
        # Render body if enabled
        if self.settings.show_body:
            self._render_body(ax, body_model)
        
        # Render garment
        if 'mesh_3d' in design and design['mesh_3d'] is not None:
            self._render_garment_mesh(ax, design['mesh_3d'])
        
        # Set axis properties
        self._setup_3d_axes(ax)
    
    def _render_body(self, ax, body_model):
        """Render body model"""
        try:
            body_mesh = body_model.get_mesh()
            if body_mesh is not None:
                # Sample vertices for performance
                vertices = body_mesh.vertices[::20]
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          c='gray', alpha=self.settings.body_opacity, s=1)
        except Exception as e:
            print(f"Warning: Could not render body: {e}")
    
    def _render_garment_mesh(self, ax, mesh):
        """Render garment mesh"""
        try:
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Render faces
            for face in faces[::5]:  # Sample for performance
                if len(face) >= 3 and all(i < len(vertices) for i in face[:3]):
                    triangle = vertices[face[:3]]
                    ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                                   alpha=self.settings.garment_opacity, 
                                   color='lightblue', edgecolor='navy', linewidth=0.1)
            
            if self.settings.show_wireframe:
                # Add wireframe
                for face in faces[::10]:
                    if len(face) >= 3 and all(i < len(vertices) for i in face[:3]):
                        triangle = vertices[face[:3]]
                        # Close the triangle
                        triangle_closed = np.vstack([triangle, triangle[0]])
                        ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], 
                               triangle_closed[:, 2], 'k-', alpha=0.3, linewidth=0.5)
        
        except Exception as e:
            print(f"Warning: Could not render garment mesh: {e}")
    
    def _setup_3d_axes(self, ax):
        """Setup 3D axes with proper scaling and labels"""
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # Set equal aspect ratio
        max_range = 400
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([0, 1800])
        ax.set_zlim([-max_range, max_range])
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
    
    def create_evolution_animation(self, generation_data: List[List[Dict]], 
                                 body_model, save_path: str):
        """Create animation showing evolution over generations"""
        
        try:
            from matplotlib.animation import FuncAnimation
        except ImportError:
            print("Animation requires matplotlib.animation")
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def animate(frame):
            ax.clear()
            if frame < len(generation_data):
                designs = generation_data[frame]
                if designs:
                    # Show best design of generation
                    best_design = max(designs, key=lambda d: d.get('fitness', 0))
                    self._render_single_design(ax, best_design, body_model)
                    
                    fitness = best_design.get('fitness', 0)
                    ax.set_title(f'Evolution - Generation {frame+1}\nBest Fitness: {fitness:.3f}')
        
        anim = FuncAnimation(fig, animate, frames=len(generation_data), 
                           interval=1000, repeat=True)
        
        try:
            anim.save(save_path, writer='pillow', fps=1)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Could not save animation: {e}")
        
        plt.show()
    
    def export_3d_model(self, design: Dict, filename: str):
        """Export 3D model in various formats"""
        
        if 'mesh_3d' not in design or design['mesh_3d'] is None:
            print("No 3D mesh to export")
            return
        
        mesh = design['mesh_3d']
        
        try:
            # Export based on file extension
            if filename.endswith('.obj'):
                mesh.export(filename)
            elif filename.endswith('.stl'):
                mesh.export(filename)
            elif filename.endswith('.ply'):
                mesh.export(filename)
            else:
                # Default to OBJ
                mesh.export(filename + '.obj')
            
            print(f"3D model exported to {filename}")
            
        except Exception as e:
            print(f"Export failed: {e}")
    
    def render_pattern_layout(self, patterns: List[Dict], 
                            fabric_width: float = 1500,
                            save_path: Optional[str] = None):
        """Render 2D pattern layout for cutting"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Simulate pattern layout
        current_x = 50  # Start margin
        current_y = 50
        row_height = 0
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(patterns)))
        
        for i, pattern in enumerate(patterns):
            if 'vertices' not in pattern:
                continue
            
            vertices = np.array(pattern['vertices'])
            if len(vertices) < 3:
                continue
            
            # Calculate pattern dimensions
            pattern_width = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
            pattern_height = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
            
            # Check if pattern fits in current row
            if current_x + pattern_width > fabric_width - 50:
                # Move to next row
                current_x = 50
                current_y += row_height + 50
                row_height = 0
            
            # Translate pattern to layout position
            offset_x = current_x - np.min(vertices[:, 0])
            offset_y = current_y - np.min(vertices[:, 1])
            
            layout_vertices = vertices + [offset_x, offset_y]
            
            # Plot pattern
            layout_vertices_closed = np.vstack([layout_vertices, layout_vertices[0]])
            ax.plot(layout_vertices_closed[:, 0], layout_vertices_closed[:, 1], 
                   color=colors[i], linewidth=2, label=pattern['name'])
            ax.fill(layout_vertices[:, 0], layout_vertices[:, 1], 
                   color=colors[i], alpha=0.3)
            
            # Add pattern name
            center_x = np.mean(layout_vertices[:, 0])
            center_y = np.mean(layout_vertices[:, 1])
            ax.text(center_x, center_y, pattern['name'], 
                   ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Update position for next pattern
            current_x += pattern_width + 50
            row_height = max(row_height, pattern_height)
        
        # Draw fabric boundary
        fabric_height = current_y + row_height + 50
        ax.plot([0, fabric_width, fabric_width, 0, 0], 
               [0, 0, fabric_height, fabric_height, 0], 
               'k-', linewidth=3, label='Fabric Edge')
        
        ax.set_xlim(-50, fabric_width + 50)
        ax.set_ylim(-50, fabric_height + 50)
        ax.set_aspect('equal')
        ax.set_xlabel('Width (mm)')
        ax.set_ylabel('Length (mm)')
        ax.set_title(f'Pattern Layout - Fabric Width: {fabric_width}mm')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Calculate efficiency
        total_pattern_area = sum(pattern.get('area', 0) for pattern in patterns)
        fabric_area = fabric_width * fabric_height
        efficiency = (total_pattern_area / fabric_area) * 100 if fabric_area > 0 else 0
        
        print(f"Layout Efficiency: {efficiency:.1f}%")
        print(f"Fabric Usage: {fabric_width:.0f}mm x {fabric_height:.0f}mm")
        
        return {
            'efficiency': efficiency,
            'fabric_dimensions': (fabric_width, fabric_height),
            'total_area': fabric_area,
            'pattern_area': total_pattern_area
        }

# Example usage
if __name__ == "__main__":
    renderer = Fashion3DRenderer()
    print("Fashion 3D Renderer initialized successfully!")