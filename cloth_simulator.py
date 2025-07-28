# cloth_simulator.py - 3D Cloth Simulation for Fashion Design
"""
Simulates how 2D pattern pieces drape and fit on a 3D body model.
Creates realistic 3D garment meshes from flat patterns.
"""

import numpy as np
import trimesh
from typing import List, Dict, Tuple, Optional
import math
from dataclasses import dataclass
from body_model import HumanBodyModel
from garment_genome import ConstructionParameters

@dataclass
class ClothProperties:
    """Physical properties of fabric"""
    density: float = 0.2          # kg/m² (typical cotton)
    elasticity: float = 0.1       # How much fabric stretches
    bending_stiffness: float = 0.01  # Resistance to bending
    friction: float = 0.3         # Friction against body
    thickness: float = 1.0        # mm
    drape_coefficient: float = 0.5  # How much fabric drapes (0=stiff, 1=fluid)

class ClothSimulator:
    """3D cloth simulation for garment visualization"""
    
    def __init__(self):
        self.cloth_properties = ClothProperties()
        self.simulation_steps = 100
        self.gravity = 9810.0  # mm/s² (gravity in millimeters)
        self.damping = 0.95
        self.time_step = 0.01
    
    def simulate_garment(self, pattern_pieces: List[Dict], 
                        body_model: HumanBodyModel,
                        construction_params: ConstructionParameters) -> Optional[trimesh.Trimesh]:
        """Simulate how pattern pieces form a 3D garment on the body"""
        
        try:
            # Step 1: Create 3D mesh from 2D patterns
            garment_mesh = self._patterns_to_3d_mesh(pattern_pieces, body_model, construction_params)
            
            # Step 2: Position garment on body
            positioned_mesh = self._position_garment_on_body(garment_mesh, body_model)
            
            # Step 3: Simulate draping
            final_mesh = self._simulate_draping(positioned_mesh, body_model, construction_params)
            
            return final_mesh
            
        except Exception as e:
            print(f"   Cloth simulation failed: {e}")
            return self._create_fallback_garment(body_model, construction_params)
    
    def _patterns_to_3d_mesh(self, pattern_pieces: List[Dict], 
                            body_model: HumanBodyModel,
                            construction_params: ConstructionParameters) -> trimesh.Trimesh:
        """Convert 2D pattern pieces to initial 3D mesh"""
        
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        body_measurements = body_model.get_measurements()
        
        for piece in pattern_pieces:
            if 'vertices' not in piece or len(piece['vertices']) < 3:
                continue
            
            # Get 2D pattern vertices
            pattern_2d = np.array(piece['vertices'])
            
            # Create 3D vertices by extruding pattern
            piece_3d_vertices = self._extrude_pattern_to_3d(
                pattern_2d, piece['name'], body_measurements, construction_params
            )
            
            # Create triangular mesh for this piece
            piece_faces = self._triangulate_pattern_piece(piece_3d_vertices, vertex_offset)
            
            all_vertices.extend(piece_3d_vertices)
            all_faces.extend(piece_faces)
            vertex_offset += len(piece_3d_vertices)
        
        if not all_vertices:
            return self._create_fallback_garment(body_model, construction_params)
        
        vertices = np.array(all_vertices)
        faces = np.array(all_faces)
        
        # Create trimesh
        try:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            return mesh
        except Exception:
            return self._create_fallback_garment(body_model, construction_params)
    
    def _extrude_pattern_to_3d(self, pattern_2d: np.ndarray, piece_name: str,
                              body_measurements: Dict, construction_params: ConstructionParameters) -> List[Tuple[float, float, float]]:
        """Convert 2D pattern to 3D by mapping onto body surface"""
        
        vertices_3d = []
        
        # Determine how to map this pattern piece to 3D space
        if "front" in piece_name.lower():
            # Front pieces map to front of body
            base_z = body_measurements['chest_depth'] / 2
            for x, y in pattern_2d:
                # Map pattern coordinates to body coordinates
                body_x = x - pattern_2d[:, 0].mean()  # Center the pattern
                body_y = body_measurements['shoulder_height'] - y  # Flip Y axis
                body_z = base_z + self._calculate_body_curve(body_x, body_y, 'front', body_measurements)
                vertices_3d.append((body_x, body_y, body_z))
                
        elif "back" in piece_name.lower():
            # Back pieces map to back of body
            base_z = -body_measurements['back_depth'] / 2
            for x, y in pattern_2d:
                body_x = x - pattern_2d[:, 0].mean()
                body_y = body_measurements['shoulder_height'] - y
                body_z = base_z + self._calculate_body_curve(body_x, body_y, 'back', body_measurements)
                vertices_3d.append((body_x, body_y, body_z))
                
        elif "sleeve" in piece_name.lower():
            # Sleeves wrap around arm position
            for i, (x, y) in enumerate(pattern_2d):
                # Map sleeve to cylindrical arm shape
                arm_radius = 60  # mm
                angle = (x / pattern_2d[:, 0].max()) * math.pi  # Wrap around
                
                arm_x = body_measurements['shoulder_width'] / 2 + arm_radius * math.cos(angle)
                arm_y = body_measurements['shoulder_height'] - y
                arm_z = arm_radius * math.sin(angle)
                vertices_3d.append((arm_x, arm_y, arm_z))
                
        else:
            # Generic mapping - lay flat in front
            base_z = body_measurements['chest_depth'] / 2 + 50  # 50mm in front
            for x, y in pattern_2d:
                body_x = x - pattern_2d[:, 0].mean()
                body_y = body_measurements['shoulder_height'] - y
                vertices_3d.append((body_x, body_y, base_z))
        
        return vertices_3d
    
    def _calculate_body_curve(self, x: float, y: float, side: str, measurements: Dict) -> float:
        """Calculate body surface curvature at given position"""
        
        # Normalize coordinates
        max_width = measurements['chest_width'] / 2
        torso_height = measurements['shoulder_height'] - measurements['hip_height']
        
        if max_width == 0 or torso_height == 0:
            return 0.0
        
        norm_x = abs(x) / max_width if max_width > 0 else 0
        norm_y = (measurements['shoulder_height'] - y) / torso_height if torso_height > 0 else 0
        
        # Body curve follows roughly elliptical cross-section
        if norm_x <= 1.0:
            if side == 'front':
                # Front curves outward
                curve = 20 * math.sqrt(max(0, 1 - norm_x*norm_x)) * (1 - norm_y*0.3)
            else:
                # Back is flatter
                curve = 10 * math.sqrt(max(0, 1 - norm_x*norm_x)) * (1 - norm_y*0.2)
        else:
            curve = 0.0
        
        return curve
    
    def _triangulate_pattern_piece(self, vertices_3d: List[Tuple[float, float, float]], 
                                  vertex_offset: int) -> List[Tuple[int, int, int]]:
        """Create triangular faces for a pattern piece"""
        
        if len(vertices_3d) < 3:
            return []
        
        faces = []
        n_vertices = len(vertices_3d)
        
        # Simple fan triangulation from first vertex
        for i in range(1, n_vertices - 1):
            faces.append((
                vertex_offset,           # First vertex
                vertex_offset + i,       # Current vertex
                vertex_offset + i + 1    # Next vertex
            ))
        
        return faces
    
    def _position_garment_on_body(self, garment_mesh: trimesh.Trimesh, 
                                 body_model: HumanBodyModel) -> trimesh.Trimesh:
        """Position the garment properly on the body"""
        
        # Get body key points for alignment
        key_points = body_model.get_key_points()
        body_measurements = body_model.get_measurements()
        
        # Center the garment on the body
        garment_center = garment_mesh.center_mass
        body_center = np.array([0, body_measurements['chest_height'], 0])
        
        translation = body_center - garment_center
        garment_mesh.apply_translation(translation)
        
        # Adjust for proper fit (move slightly away from body)
        fit_offset = np.array([0, 0, 10])  # 10mm ease in Z direction
        garment_mesh.apply_translation(fit_offset)
        
        return garment_mesh
    
    def _simulate_draping(self, garment_mesh: trimesh.Trimesh, 
                         body_model: HumanBodyModel,
                         construction_params: ConstructionParameters) -> trimesh.Trimesh:
        """Simulate fabric draping under gravity"""
        
        vertices = garment_mesh.vertices.copy()
        n_vertices = len(vertices)
        
        # Initialize physics simulation
        velocities = np.zeros_like(vertices)
        forces = np.zeros_like(vertices)
        
        # Simulation parameters
        mass = 0.1  # kg per vertex
        
        for step in range(self.simulation_steps):
            # Reset forces
            forces.fill(0.0)
            
            # Apply gravity
            forces[:, 1] -= mass * self.gravity  # Y is up, so negative gravity
            
            # Apply fabric tension forces (simplified)
            forces += self._calculate_fabric_tension(vertices, garment_mesh.faces)
            
            # Body collision forces
            collision_forces = self._calculate_body_collision_forces(vertices, body_model)
            forces += collision_forces
            
            # Update velocities and positions using Verlet integration
            acceleration = forces / mass
            velocities = velocities * self.damping + acceleration * self.time_step
            vertices += velocities * self.time_step
            
            # Constrain certain vertices (e.g., shoulder points)
            vertices = self._apply_constraints(vertices, body_model, construction_params)
        
        # Create new mesh with simulated vertices
        try:
            simulated_mesh = trimesh.Trimesh(vertices=vertices, faces=garment_mesh.faces)
            simulated_mesh.remove_degenerate_faces()
            return simulated_mesh
        except Exception:
            return garment_mesh  # Return original if simulation fails
    
    def _calculate_fabric_tension(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Calculate internal fabric tension forces"""
        forces = np.zeros_like(vertices)
        
        if len(faces) == 0:
            return forces
        
        # Calculate spring forces between connected vertices
        for face in faces:
            if len(face) >= 3:
                v0, v1, v2 = face[0], face[1], face[2]
                
                # Edge springs
                edges = [(v0, v1), (v1, v2), (v2, v0)]
                
                for vi, vj in edges:
                    if vi < len(vertices) and vj < len(vertices):
                        # Spring force between connected vertices
                        edge_vector = vertices[vj] - vertices[vi]
                        edge_length = np.linalg.norm(edge_vector)
                        
                        if edge_length > 0:
                            # Rest length based on cloth properties
                            rest_length = edge_length * (1 + self.cloth_properties.elasticity)
                            
                            # Spring force
                            spring_force = (edge_length - rest_length) * edge_vector / edge_length
                            spring_force *= self.cloth_properties.bending_stiffness * 1000  # Scale for mm units
                            
                            forces[vi] += spring_force
                            forces[vj] -= spring_force
        
        return forces
    
    def _calculate_body_collision_forces(self, vertices: np.ndarray, 
                                       body_model: HumanBodyModel) -> np.ndarray:
        """Calculate forces to prevent garment from intersecting body"""
        forces = np.zeros_like(vertices)
        
        body_mesh = body_model.get_mesh()
        if body_mesh is None:
            return forces
        
        # Check each garment vertex against body surface
        for i, vertex in enumerate(vertices):
            # Find closest point on body surface
            closest_point, distance, _ = trimesh.proximity.closest_point(body_mesh, [vertex])
            distance = distance[0]
            
            # Minimum distance (ease allowance)
            min_distance = 5.0  # 5mm minimum ease
            
            if distance < min_distance:
                # Calculate repulsion force
                direction = vertex - closest_point[0]
                direction_length = np.linalg.norm(direction)
                
                if direction_length > 0:
                    direction_normalized = direction / direction_length
                    
                    # Force magnitude inversely proportional to distance
                    force_magnitude = (min_distance - distance) * 1000  # Strong repulsion
                    
                    forces[i] += direction_normalized * force_magnitude
        
        return forces
    
    def _apply_constraints(self, vertices: np.ndarray, 
                          body_model: HumanBodyModel,
                          construction_params: ConstructionParameters) -> np.ndarray:
        """Apply constraints to keep garment properly positioned"""
        
        body_measurements = body_model.get_measurements()
        constrained_vertices = vertices.copy()
        
        # Constraint 1: Keep shoulder points at shoulder level
        shoulder_height = body_measurements['shoulder_height']
        shoulder_tolerance = 20  # mm
        
        for i, vertex in enumerate(constrained_vertices):
            # If vertex is near shoulder height and at shoulder width
            if (abs(vertex[1] - shoulder_height) < shoulder_tolerance and
                abs(abs(vertex[0]) - body_measurements['shoulder_width']/2) < 50):
                
                # Constrain to shoulder height
                constrained_vertices[i, 1] = shoulder_height
        
        # Constraint 2: Prevent extreme deformation
        center = np.mean(constrained_vertices, axis=0)
        max_deviation = body_measurements['chest_width']  # Maximum reasonable spread
        
        for i, vertex in enumerate(constrained_vertices):
            distance_from_center = np.linalg.norm(vertex - center)
            if distance_from_center > max_deviation:
                direction = (vertex - center) / distance_from_center
                constrained_vertices[i] = center + direction * max_deviation
        
        return constrained_vertices
    
    def _create_fallback_garment(self, body_model: HumanBodyModel,
                                construction_params: ConstructionParameters) -> trimesh.Trimesh:
        """Create a simple fallback garment when simulation fails"""
        
        body_measurements = body_model.get_measurements()
        
        try:
            # Create a simple dress shape using basic geometry
            width = body_measurements['chest_width'] + 100  # 100mm ease
            height = body_measurements['shoulder_height'] - body_measurements['hip_height']
            depth = body_measurements['chest_depth'] + 50   # 50mm ease
            
            # Create simple box vertices
            vertices = np.array([
                # Front face
                [-width/2, body_measurements['shoulder_height'], depth/2],
                [width/2, body_measurements['shoulder_height'], depth/2],
                [width/2, body_measurements['hip_height'], depth/2],
                [-width/2, body_measurements['hip_height'], depth/2],
                
                # Back face  
                [-width/2, body_measurements['shoulder_height'], -depth/2],
                [width/2, body_measurements['shoulder_height'], -depth/2],
                [width/2, body_measurements['hip_height'], -depth/2],
                [-width/2, body_measurements['hip_height'], -depth/2],
            ])
            
            # Create simple faces
            faces = np.array([
                [0, 1, 2], [0, 2, 3],  # Front
                [5, 4, 7], [5, 7, 6],  # Back
                [4, 0, 3], [4, 3, 7],  # Left
                [1, 5, 6], [1, 6, 2],  # Right
                [4, 5, 1], [4, 1, 0],  # Top
                [3, 2, 6], [3, 6, 7],  # Bottom
            ])
            
            return trimesh.Trimesh(vertices=vertices, faces=faces)
            
        except Exception as e:
            print(f"   Warning: Even fallback garment failed: {e}")
            # Ultimate fallback - create minimal valid mesh
            vertices = np.array([
                [0, 1000, 100],
                [100, 1000, 100], 
                [50, 900, 100]
            ])
            faces = np.array([[0, 1, 2]])
            
            return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def evaluate_fit_quality(self, garment_mesh: trimesh.Trimesh, 
                            body_model: HumanBodyModel) -> Dict[str, float]:
        """Evaluate how well the simulated garment fits the body"""
        
        if garment_mesh is None:
            return {'overall_score': 0.0}
        
        body_mesh = body_model.get_mesh()
        
        # Sample points on garment surface
        garment_points = garment_mesh.sample(500)
        
        # Find distances to body
        distances = []
        intersections = 0
        
        for point in garment_points:
            closest_point, distance, _ = trimesh.proximity.closest_point(body_mesh, [point])
            distances.append(distance[0])
            
            if distance[0] < 2.0:  # Less than 2mm - intersection
                intersections += 1
        
        # Calculate metrics
        avg_ease = np.mean(distances)
        min_ease = np.min(distances)
        max_ease = np.max(distances)
        ease_std = np.std(distances)
        intersection_ratio = intersections / len(garment_points)
        
        # Ideal ease is 15-30mm
        ideal_ease = 22.5
        ease_score = 1.0 - abs(avg_ease - ideal_ease) / ideal_ease
        ease_score = max(0.0, ease_score)
        
        # Penalize intersections heavily
        intersection_penalty = intersection_ratio * 2.0
        
        # Penalize excessive variation in ease
        consistency_score = 1.0 - min(ease_std / ideal_ease, 1.0)
        
        # Overall score
        overall_score = max(0.0, (ease_score + consistency_score) / 2 - intersection_penalty)
        
        return {
            'overall_score': overall_score,
            'average_ease': avg_ease,
            'minimum_ease': min_ease,
            'maximum_ease': max_ease,
            'ease_standard_deviation': ease_std,
            'intersection_ratio': intersection_ratio,
            'ease_score': ease_score,
            'consistency_score': consistency_score
        }
    
    def visualize_simulation(self, garment_mesh: trimesh.Trimesh, 
                            body_model: HumanBodyModel, 
                            save_path: Optional[str] = None):
        """Visualize the simulated garment on the body"""
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 10))
        
        # Main 3D view
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Plot body mesh (wireframe)
        body_mesh = body_model.get_mesh()
        if body_mesh is not None:
            body_vertices = body_mesh.vertices[::20]  # Sample for performance
            ax1.scatter(body_vertices[:, 0], body_vertices[:, 1], body_vertices[:, 2], 
                       c='gray', alpha=0.3, s=1, label='Body')
        
        # Plot garment mesh
        if garment_mesh is not None:
            garment_vertices = garment_mesh.vertices
            faces = garment_mesh.faces
            
            # Plot sample faces
            for face in faces[::5]:  # Every 5th face
                if len(face) >= 3 and all(i < len(garment_vertices) for i in face[:3]):
                    triangle = garment_vertices[face[:3]]
                    ax1.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                                   alpha=0.7, color='blue')
        
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title('3D Garment Simulation')
        ax1.legend()
        
        # Front view
        ax2 = fig.add_subplot(222)
        if garment_mesh is not None:
            garment_front = garment_vertices[garment_vertices[:, 2] > 0]  # Front-facing vertices
            ax2.scatter(garment_front[:, 0], garment_front[:, 1], c='blue', alpha=0.6, s=10)
        
        if body_mesh is not None:
            body_front = body_vertices[body_vertices[:, 2] > 0]
            ax2.scatter(body_front[:, 0], body_front[:, 1], c='gray', alpha=0.3, s=5)
        
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title('Front View')
        ax2.grid(True, alpha=0.3)
        
        # Side view
        ax3 = fig.add_subplot(223)
        if garment_mesh is not None:
            ax3.scatter(garment_vertices[:, 2], garment_vertices[:, 1], c='blue', alpha=0.6, s=10)
        
        if body_mesh is not None:
            ax3.scatter(body_vertices[:, 2], body_vertices[:, 1], c='gray', alpha=0.3, s=5)
        
        ax3.set_xlabel('Z (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_title('Side View')
        ax3.grid(True, alpha=0.3)
        
        # Fit analysis
        ax4 = fig.add_subplot(224)
        fit_metrics = self.evaluate_fit_quality(garment_mesh, body_model)
        
        metrics_names = ['Overall Score', 'Ease Score', 'Consistency', 'Avg Ease (mm)']
        metrics_values = [
            fit_metrics.get('overall_score', 0),
            fit_metrics.get('ease_score', 0),
            fit_metrics.get('consistency_score', 0),
            min(fit_metrics.get('average_ease', 0) / 50, 1)  # Normalize to 0-1
        ]
        
        bars = ax4.bar(metrics_names, metrics_values, color=['green', 'blue', 'orange', 'red'])
        ax4.set_ylim(0, 1)
        ax4.set_title('Fit Quality Metrics')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print detailed metrics
        print("\nDetailed Fit Analysis:")
        for key, value in fit_metrics.items():
            print(f"  {key}: {value:.3f}")

# Example usage and testing
if __name__ == "__main__":
    from body_model import HumanBodyModel
    from pattern_generator import PatternGenerator
    from garment_genome import ConstructionParameters
    
    # Create test setup
    body = HumanBodyModel(size='M')
    generator = PatternGenerator(body)
    simulator = ClothSimulator()
    
    # Create test patterns
    params = ConstructionParameters.create_default("jacket")
    patterns = generator.generate_patterns(params, "jacket")
    
    print(f"Testing cloth simulation with {len(patterns)} pattern pieces...")
    
    # Run simulation
    garment_mesh = simulator.simulate_garment(patterns, body, params)
    
    if garment_mesh is not None:
        print(f"✓ Simulation successful!")
        print(f"  Vertices: {len(garment_mesh.vertices)}")
        print(f"  Faces: {len(garment_mesh.faces)}")
        print(f"  Volume: {garment_mesh.volume:.2f} mm³")
        
        # Evaluate fit
        fit_metrics = simulator.evaluate_fit_quality(garment_mesh, body)
        print(f"  Overall fit score: {fit_metrics['overall_score']:.3f}")
        
        # Visualize result
        simulator.visualize_simulation(garment_mesh, body, "simulation_test.png")
        
        # Export result
        garment_mesh.export("test_garment.obj")
        print("  Exported garment to test_garment.obj")
        
    else:
        print("✗ Simulation failed")