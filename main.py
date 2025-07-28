# main.py - 3D Fashion CPPN Evolution System
"""
3D Fashion CPPN Evolution System
Generates actual garment construction parameters that can be sewn and worn.
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import math

# Import our custom modules
try:
    from garment_genome import GarmentGenome, GarmentCPPN, ConstructionParameters
    from body_model import HumanBodyModel
    from pattern_generator import PatternGenerator
    from cloth_simulator import ClothSimulator
    from fashion_renderer import Fashion3DRenderer
    from evolution_engine import FashionEvolutionEngine
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all files are in the same directory:")
    print("- main.py")
    print("- garment_genome.py") 
    print("- body_model.py")
    print("- pattern_generator.py")
    print("- cloth_simulator.py")
    print("- fashion_renderer.py")
    print("- evolution_engine.py")
    exit(1)

class GarmentType(Enum):
    JACKET = "jacket"
    DRESS = "dress"
    SHIRT = "shirt"
    PANTS = "pants"
    SKIRT = "skirt"

@dataclass
class FashionDesign:
    """Complete fashion design with 3D representation"""
    genome: GarmentGenome
    construction_params: ConstructionParameters
    pattern_pieces: List[Dict]
    mesh_3d: Optional[trimesh.Trimesh] = None
    fitness: float = 0.0
    generation: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'genome': self.genome.to_dict(),
            'construction_params': self.construction_params.to_dict(),
            'pattern_pieces': self.pattern_pieces,
            'fitness': self.fitness,
            'generation': self.generation
        }

class Fashion3DSystem:
    """Main system orchestrating 3D fashion evolution"""
    
    def __init__(self, garment_type: GarmentType = GarmentType.JACKET):
        self.garment_type = garment_type
        self.body_model = HumanBodyModel()
        self.pattern_generator = PatternGenerator(self.body_model)
        self.cloth_simulator = ClothSimulator()
        self.renderer = Fashion3DRenderer()
        self.evolution_engine = FashionEvolutionEngine()
        
        print(f"🎨 Initialized 3D Fashion System for {garment_type.value}")
        print(f"📐 Body model: {self.body_model.get_measurements()}")
    
    def create_design_from_genome(self, genome: GarmentGenome) -> FashionDesign:
        """Transform a genome into a complete 3D fashion design"""
        
        # Step 1: Generate construction parameters from CPPN
        cppn = GarmentCPPN(genome, self.garment_type.value)
        construction_params = self._generate_construction_parameters(cppn)
        
        # Step 2: Generate pattern pieces
        pattern_pieces = self.pattern_generator.generate_patterns(
            construction_params, self.garment_type.value
        )
        
        # Step 3: Simulate 3D garment
        mesh_3d = self.cloth_simulator.simulate_garment(
            pattern_pieces, self.body_model, construction_params
        )
        
        # Debug info
        if mesh_3d is not None:
            print(f"     ✓ 3D mesh: {len(mesh_3d.vertices)} vertices, {len(mesh_3d.faces)} faces")
        else:
            print(f"     ✗ 3D mesh creation failed")
        
        # Step 4: Create complete design
        design = FashionDesign(
            genome=genome,
            construction_params=construction_params,
            pattern_pieces=pattern_pieces,
            mesh_3d=mesh_3d
        )
        
        return design
    
    def _generate_construction_parameters(self, cppn: GarmentCPPN) -> ConstructionParameters:
        """Generate construction parameters using body-aware CPPN evaluation"""
        
        # Get key body measurements for CPPN input
        measurements = self.body_model.get_measurements()
        
        # Evaluate CPPN at key construction points
        construction_points = self._get_construction_evaluation_points()
        
        params = {}
        for point_name, (x, y, z) in construction_points.items():
            try:
                # Normalize coordinates relative to body with safety checks
                chest_width = measurements.get('chest_width', 520)
                waist_height = measurements.get('waist_height', 1080)
                torso_height = measurements.get('torso_height', 340)
                body_depth = measurements.get('body_depth', 220)
                
                norm_x = x / chest_width if chest_width > 0 else 0
                norm_y = (y - waist_height) / torso_height if torso_height > 0 else 0
                norm_z = z / body_depth if body_depth > 0 else 0
                
                # Evaluate CPPN
                outputs = cppn.evaluate(norm_x, norm_y, norm_z)
                params[point_name] = outputs
            except Exception as e:
                print(f"   Warning: Failed to evaluate CPPN at {point_name}: {e}")
                # Provide default outputs
                params[point_name] = {
                    'ease': 0.0,
                    'dart': 0.0,
                    'curve': 0.0,
                    'suppression': 0.0,
                    'slope': 0.0,
                    'asymmetry': 0.0
                }
        
        try:
            return ConstructionParameters.from_cppn_outputs(params, self.garment_type.value)
        except Exception as e:
            print(f"   Warning: Failed to create construction parameters: {e}")
            return ConstructionParameters.create_default(self.garment_type.value)
    
    def _get_construction_evaluation_points(self) -> Dict[str, Tuple[float, float, float]]:
        """Get key 3D points where CPPN should be evaluated for construction"""
        measurements = self.body_model.get_measurements()
        
        if self.garment_type == GarmentType.JACKET:
            return {
                'shoulder_point': (measurements['shoulder_width']/2, measurements['shoulder_height'], 0),
                'chest_point': (measurements['chest_width']/2, measurements['chest_height'], measurements['chest_depth']/2),
                'waist_point': (measurements['waist_width']/2, measurements['waist_height'], measurements['waist_depth']/2),
                'hem_point': (measurements['hip_width']/2, measurements['hip_height'], measurements['hip_depth']/2),
                'lapel_point': (measurements['chest_width']/4, measurements['chest_height'] + 50, measurements['chest_depth']/2),
                'armhole_front': (measurements['shoulder_width']/2, measurements['armpit_height'], measurements['chest_depth']/3),
                'armhole_back': (measurements['shoulder_width']/2, measurements['armpit_height'], -measurements['back_depth']/3),
                'side_seam': (measurements['chest_width']/2, measurements['waist_height'], 0),
            }
        elif self.garment_type == GarmentType.DRESS:
            return {
                'neckline': (0, measurements['neck_height'], measurements.get('neck_depth', 140)/2),
                'bust_point': (measurements.get('bust_width', 500)/2, measurements.get('bust_height', 1260), measurements.get('bust_depth', 250)/2),
                'waist_point': (measurements['waist_width']/2, measurements['waist_height'], measurements['waist_depth']/2),
                'hip_point': (measurements['hip_width']/2, measurements['hip_height'], measurements['hip_depth']/2),
                'hem_point': (measurements.get('hip_width', 550)/2 * 1.2, measurements.get('hem_height', 600), 0),
                'shoulder_point': (measurements['shoulder_width']/2, measurements['shoulder_height'], 0),
            }
        else:
            # Default construction points
            return {
                'center_front': (0, measurements['chest_height'], measurements['chest_depth']/2),
                'center_back': (0, measurements['chest_height'], -measurements['back_depth']/2),
                'side_seam': (measurements['chest_width']/2, measurements['waist_height'], 0),
            }
    
    def evaluate_design_wearability(self, design: FashionDesign) -> float:
        """Evaluate how wearable and realistic a design is"""
        score = 0.0
        
        # Check if garment fits properly
        fit_score = self._evaluate_fit(design)
        score += fit_score * 0.4
        
        # Check construction feasibility
        construction_score = self._evaluate_construction(design)
        score += construction_score * 0.3
        
        # Check aesthetic appeal
        aesthetic_score = self._evaluate_aesthetics(design)
        score += aesthetic_score * 0.3
        
        return score
    
    def _evaluate_fit(self, design: FashionDesign) -> float:
        """Evaluate how well the garment fits the body"""
        if design.mesh_3d is None:
            return 0.0
        
        # Check for intersections with body
        body_mesh = self.body_model.get_mesh()
        
        # Simple collision detection
        garment_vertices = design.mesh_3d.vertices
        body_vertices = body_mesh.vertices
        
        # Check minimum distance to body (should have ease)
        min_distances = []
        for garment_vertex in garment_vertices[::10]:  # Sample every 10th vertex for performance
            distances = np.linalg.norm(body_vertices - garment_vertex, axis=1)
            min_distances.append(np.min(distances))
        
        avg_ease = np.mean(min_distances)
        
        # Ideal ease is 2-5cm depending on garment type
        ideal_ease = 30.0 if self.garment_type == GarmentType.JACKET else 20.0
        ease_score = 1.0 - abs(avg_ease - ideal_ease) / ideal_ease
        
        return max(0.0, ease_score)
    
    def _evaluate_construction(self, design: FashionDesign) -> float:
        """Evaluate if the design can actually be constructed"""
        score = 0.0
        
        # Check if pattern pieces are reasonable
        for piece in design.pattern_pieces:
            # Check for reasonable proportions
            if 'vertices' in piece:
                vertices = np.array(piece['vertices'])
                if len(vertices) > 2:
                    # Calculate area - should not be too small or too large
                    area = self._calculate_polygon_area(vertices)
                    if 1000 < area < 200000:  # Reasonable area in mm²
                        score += 0.2
                    
                    # Check for reasonable perimeter
                    perimeter = self._calculate_polygon_perimeter(vertices)
                    if 200 < perimeter < 2000:  # Reasonable perimeter in mm
                        score += 0.2
        
        # Check seam feasibility
        params = design.construction_params
        if hasattr(params, 'seam_allowances'):
            if all(5 <= allowance <= 20 for allowance in params.seam_allowances.values()):
                score += 0.3
        
        # Check for reasonable ease amounts
        if hasattr(params, 'ease_amounts'):
            if all(0 <= ease <= 100 for ease in params.ease_amounts.values()):
                score += 0.3
        
        return min(1.0, score)
    
    def _evaluate_aesthetics(self, design: FashionDesign) -> float:
        """Evaluate aesthetic appeal of the design"""
        score = 0.0
        
        # Proportion harmony (golden ratio, etc.)
        if design.mesh_3d:
            bounds = design.mesh_3d.bounds
            width = bounds[1][0] - bounds[0][0]
            height = bounds[1][1] - bounds[0][1]
            
            if height > 0:
                ratio = width / height
                golden_ratio = 1.618
                proportion_score = 1.0 - abs(ratio - golden_ratio) / golden_ratio
                score += proportion_score * 0.5
        
        # Symmetry vs asymmetry balance
        params = design.construction_params
        if hasattr(params, 'asymmetry_factor'):
            # Sweet spot for fashion asymmetry
            asymmetry_score = 1.0 - abs(params.asymmetry_factor - 0.15) / 0.15
            score += max(0, asymmetry_score) * 0.3
        
        # Complexity - not too simple, not too complex
        genome_complexity = len(design.genome.nodes) + len(design.genome.connections)
        complexity_score = 1.0 - abs(genome_complexity - 20) / 20
        score += max(0, complexity_score) * 0.2
        
        return min(1.0, score)
    
    def _calculate_polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate area of polygon using shoelace formula"""
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] 
                            for i in range(-1, len(x)-1)))
    
    def _calculate_polygon_perimeter(self, vertices: np.ndarray) -> float:
        """Calculate perimeter of polygon"""
        perimeter = 0.0
        for i in range(len(vertices)):
            next_i = (i + 1) % len(vertices)
            perimeter += np.linalg.norm(vertices[next_i] - vertices[i])
        return perimeter
    
    def run_evolution(self, generations: int = 10, population_size: int = 16):
        """Run the complete 3D fashion evolution process"""
        
        # Create timestamped folder for results
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = f"fashion_evolution_{timestamp}"
        import os
        os.makedirs(results_folder, exist_ok=True)
        print(f"📁 Results will be saved to: {results_folder}")
        
        print(f"\n🧬 Starting 3D Fashion Evolution")
        print(f"   Generations: {generations}")
        print(f"   Population: {population_size}")
        print(f"   Garment Type: {self.garment_type.value}")
        
        # Initialize population
        population = self.evolution_engine.create_initial_population(
            population_size, self.garment_type.value
        )
        
        for generation in range(generations):
            print(f"\n👗 Generation {generation + 1}/{generations}")
            print("-" * 40)
            
            # Create designs from genomes
            designs = []
            for i, genome in enumerate(population):
                print(f"   Creating design {i+1}/{len(population)}...")
                try:
                    design = self.create_design_from_genome(genome)
                    design.generation = generation
                    designs.append(design)
                except Exception as e:
                    print(f"   ⚠️  Failed to create design {i+1}: {e}")
                    # Create a minimal fallback design
                    fallback_design = FashionDesign(
                        genome=genome,
                        construction_params=ConstructionParameters.create_default(self.garment_type.value),
                        pattern_pieces=[],
                        fitness=0.0,
                        generation=generation
                    )
                    designs.append(fallback_design)
            
            # Evaluate designs
            print("   Evaluating designs...")
            for design in designs:
                design.fitness = self.evaluate_design_wearability(design)
            
            # Display results
            self.display_generation(designs, generation, results_folder)
            
            # Select best designs for breeding
            designs.sort(key=lambda d: d.fitness, reverse=True)
            selected_designs = designs[:population_size//2]
            
            print(f"   Best fitness: {selected_designs[0].fitness:.3f}")
            print(f"   Average fitness: {np.mean([d.fitness for d in designs]):.3f}")
            
            # Save generation data
            self._save_generation_data(designs, generation, results_folder)
            
            # Evolve to next generation
            if generation < generations - 1:
                # Copy fitness from designs back to genomes for evolution
                for design in designs:
                    design.genome.fitness = design.fitness
                
                population = self.evolution_engine.evolve_population(
                    [d.genome for d in selected_designs],
                    population_size
                )
        
        print("\n🎉 Evolution complete!")
        
        # Save final results
        self.save_best_design(designs, os.path.join(results_folder, f"best_{self.garment_type.value}_design"))
        
        return designs
    
    def _save_generation_data(self, designs: List[FashionDesign], generation: int, results_folder: str):
        """Save generation data to JSON"""
        import os
        
        generation_data = {
            'generation': generation + 1,
            'designs': [design.to_dict() for design in designs],
            'best_fitness': max(d.fitness for d in designs),
            'average_fitness': np.mean([d.fitness for d in designs]),
            'design_count': len(designs)
        }
        
        filename = os.path.join(results_folder, f"generation_{generation + 1}_data.json")
        with open(filename, 'w') as f:
            json.dump(generation_data, f, indent=2, default=str)
        
        print(f"   💾 Saved generation data to {filename}")
    
    def display_generation(self, designs: List[FashionDesign], generation: int, results_folder: str = "."):
        """Display the current generation of designs"""
        import os
        
        # Create visualization
        n_designs = min(len(designs), 9)  # Show top 9
        fig = plt.figure(figsize=(15, 10))
        
        print(f"   Creating visualization with {n_designs} designs...")
        
        for i in range(n_designs):
            design = designs[i]
            
            # Create subplot for 3D visualization
            ax = fig.add_subplot(3, 3, i+1, projection='3d')
            
            # Always draw a realistic dress
            self._draw_realistic_dress(ax, design, i+1)
            
            ax.set_title(f'Design {i+1}\nFitness: {design.fitness:.3f}', fontsize=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set equal aspect ratio
            max_range = 400  # mm
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([500, 1600])
            ax.set_zlim([-max_range, max_range])
            
            # Set viewing angle for better dress visibility
            ax.view_init(elev=10, azim=45)
        
        plt.suptitle(f'3D Fashion Evolution - Generation {generation + 1}', fontsize=16)
        plt.tight_layout()
        
        # Save the image
        filename = os.path.join(results_folder, f'generation_{generation + 1}_3d.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   💾 Saved visualization to {filename}")
        
        # Force display and keep window open
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to ensure rendering
        
        # Also create a simple summary image
        self._create_summary_image(designs, generation, results_folder)
        
        plt.close(fig)  # Close to prevent memory issues
    
    def _draw_realistic_dress(self, ax, design: FashionDesign, design_number: int):
        """Draw a clear, recognizable dress shape on a human mannequin"""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        measurements = self.body_model.get_measurements()
        
        # First, draw a clear human mannequin
        self._draw_human_mannequin(ax, measurements)
        
        # Get dress parameters from construction params or use defaults
        if hasattr(design, 'construction_params') and design.construction_params:
            params = design.construction_params
            bust_ease = params.ease_amounts.get('bust', 80)
            waist_ease = params.ease_amounts.get('waist', 60) 
            hip_ease = params.ease_amounts.get('hip', 100)
            hem_curve = getattr(params, 'hem_curve', 0)
        else:
            bust_ease = 80
            waist_ease = 60
            hip_ease = 100
            hem_curve = 0
        
        # Calculate dress measurements (body + ease)
        bust_width = (measurements.get('bust_width', 500) + bust_ease) / 2
        waist_width = (measurements['waist_width'] + waist_ease) / 2
        hip_width = (measurements['hip_width'] + hip_ease) / 2
        
        # Add variation based on fitness - higher fitness = more flattering proportions
        fitness_factor = design.fitness
        waist_suppression = 0.8 + (fitness_factor * 0.4)  # 0.8 to 1.2 multiplier
        hem_flare = 1.2 + (fitness_factor * 0.3)  # 1.2 to 1.5 multiplier
        
        waist_width *= waist_suppression
        hem_width = hip_width * hem_flare
        
        # Dress key heights
        neckline_height = measurements['neck_height'] - 30
        bust_height = measurements.get('bust_height', 1260)
        waist_height = measurements['waist_height']
        hip_height = measurements['hip_height']
        hem_height = hip_height - 350  # Knee-length dress
        
        # Create dress front silhouette - clear A-line shape
        dress_front_points = [
            # Neckline (scooped)
            (-measurements['neck_width']/4, neckline_height, measurements['neck_depth']/2 + 25),
            (measurements['neck_width']/4, neckline_height, measurements['neck_depth']/2 + 25),
            
            # Shoulders/armholes
            (bust_width * 0.9, neckline_height + 30, measurements['chest_depth']/2 + 20),
            (-bust_width * 0.9, neckline_height + 30, measurements['chest_depth']/2 + 20),
            
            # Bust area - fuller
            (bust_width, bust_height, measurements.get('bust_depth', 250)/2 + 20),
            (-bust_width, bust_height, measurements.get('bust_depth', 250)/2 + 20),
            
            # Waist - fitted
            (waist_width, waist_height, measurements['waist_depth']/2 + 15),
            (-waist_width, waist_height, measurements['waist_depth']/2 + 15),
            
            # Hip - slightly flared
            (hip_width * 1.1, hip_height, measurements['hip_depth']/2 + 15),
            (-hip_width * 1.1, hip_height, measurements['hip_depth']/2 + 15),
            
            # Hem - A-line flare with curve variation
            (hem_width + hem_curve, hem_height, measurements['hip_depth']/2 + 10),
            (-hem_width - hem_curve, hem_height, measurements['hip_depth']/2 + 10),
        ]
        
        # Create dress back (slightly different proportions)
        dress_back_points = []
        for point in dress_front_points:
            x, y, z = point
            # Move to back, make slightly narrower
            back_z = -measurements['back_depth']/2 - 15
            back_x = x * 0.95  # Slightly narrower back
            dress_back_points.append((back_x, y, back_z))
        
        # Convert to numpy arrays
        dress_front = np.array(dress_front_points)
        dress_back = np.array(dress_back_points)
        
        # Color scheme based on fitness - better dresses get better colors
        base_colors = [
            [1.0, 0.7, 0.8],    # Pink
            [0.8, 0.6, 1.0],    # Lavender  
            [0.6, 0.8, 1.0],    # Light blue
            [0.8, 1.0, 0.6],    # Light green
            [1.0, 0.9, 0.6],    # Light yellow
        ]
        
        color_index = min(int(fitness_factor * len(base_colors)), len(base_colors) - 1)
        dress_color = base_colors[color_index]
        
        # Draw dress as clean surfaces
        try:
            # Create front dress panel
            front_triangles = self._create_dress_triangles(dress_front)
            if front_triangles:
                front_poly = Poly3DCollection(front_triangles, alpha=0.85, linewidths=1.0)
                front_poly.set_facecolor(dress_color)
                front_poly.set_edgecolor([c * 0.7 for c in dress_color])
                ax.add_collection3d(front_poly)
            
            # Create back dress panel
            back_triangles = self._create_dress_triangles(dress_back)
            if back_triangles:
                back_poly = Poly3DCollection(back_triangles, alpha=0.75, linewidths=0.8)
                back_color = [c * 0.85 for c in dress_color]  # Slightly darker back
                back_poly.set_facecolor(back_color)
                back_poly.set_edgecolor([c * 0.6 for c in back_color])
                ax.add_collection3d(back_poly)
            
            # Connect front and back with side panels for 3D effect
            side_triangles = self._create_side_panels(dress_front, dress_back)
            if side_triangles:
                side_poly = Poly3DCollection(side_triangles, alpha=0.6, linewidths=0.5)
                side_color = [c * 0.9 for c in dress_color]
                side_poly.set_facecolor(side_color)
                side_poly.set_edgecolor([c * 0.5 for c in side_color])
                ax.add_collection3d(side_poly)
                
        except Exception as e:
            print(f"   Warning: Advanced dress rendering failed: {e}")
            # Fallback to simple wireframe
            ax.plot(dress_front[:, 0], dress_front[:, 1], dress_front[:, 2], 
                   color=dress_color, linewidth=4, alpha=0.9, label='Dress Front')
            ax.plot(dress_back[:, 0], dress_back[:, 1], dress_back[:, 2], 
                   color=[c * 0.8 for c in dress_color], linewidth=3, alpha=0.8, label='Dress Back')
        
        # Add dress details
        self._add_dress_details(ax, dress_front, dress_color, fitness_factor)
        
        # Clear, readable label
        label_color = 'navy' if sum(dress_color) > 2.0 else 'white'
        ax.text(0, measurements['neck_height'] + 150, 100, 
               f'DRESS {design_number}\nFitness: {design.fitness:.3f}', 
               ha='center', va='center', fontsize=11, fontweight='bold', 
               color=label_color,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    def _draw_human_mannequin(self, ax, measurements):
        """Draw a clear human mannequin"""
        
        # Head
        head_center = [0, measurements['neck_height'] + 120, 0]
        ax.scatter(*head_center, c='peachpuff', s=400, alpha=0.8, marker='o', edgecolors='brown', linewidth=1)
        
        # Neck
        neck_points = [
            [0, measurements['neck_height'], 0],
            [0, measurements['neck_height'] + 80, 0]
        ]
        neck_line = np.array(neck_points)
        ax.plot(neck_line[:, 0], neck_line[:, 1], neck_line[:, 2], 
               color='peachpuff', linewidth=8, alpha=0.8)
        
        # Body outline - cleaner mannequin shape
        body_points = [
            # Shoulders
            [-measurements['shoulder_width']/2, measurements['shoulder_height'], 0],
            [measurements['shoulder_width']/2, measurements['shoulder_height'], 0],
            
            # Chest
            [measurements['chest_width']/2, measurements['chest_height'], 0],
            [-measurements['chest_width']/2, measurements['chest_height'], 0],
            
            # Waist  
            [measurements['waist_width']/2, measurements['waist_height'], 0],
            [-measurements['waist_width']/2, measurements['waist_height'], 0],
            
            # Hips
            [measurements['hip_width']/2, measurements['hip_height'], 0],
            [-measurements['hip_width']/2, measurements['hip_height'], 0],
            
            # Legs (simplified)
            [measurements['hip_width']/4, measurements['hip_height'] - 300, 0],
            [-measurements['hip_width']/4, measurements['hip_height'] - 300, 0],
        ]
        
        # Draw body outline
        body_outline = np.array(body_points)
        ax.plot(body_outline[::2, 0], body_outline[::2, 1], body_outline[::2, 2], 
               color='tan', linewidth=3, alpha=0.7, label='Body')  # Right side
        ax.plot(body_outline[1::2, 0], body_outline[1::2, 1], body_outline[1::2, 2], 
               color='tan', linewidth=3, alpha=0.7)  # Left side
        
        # Arms (simple)
        # Right arm
        right_arm = np.array([
            [measurements['shoulder_width']/2, measurements['shoulder_height'], 0],
            [measurements['shoulder_width']/2 + 80, measurements['shoulder_height'] - 100, 0],
            [measurements['shoulder_width']/2 + 60, measurements['waist_height'], 0]
        ])
        ax.plot(right_arm[:, 0], right_arm[:, 1], right_arm[:, 2], 
               color='tan', linewidth=4, alpha=0.6)
        
        # Left arm  
        left_arm = np.array([
            [-measurements['shoulder_width']/2, measurements['shoulder_height'], 0],
            [-measurements['shoulder_width']/2 - 80, measurements['shoulder_height'] - 100, 0],
            [-measurements['shoulder_width']/2 - 60, measurements['waist_height'], 0]
        ])
        ax.plot(left_arm[:, 0], left_arm[:, 1], left_arm[:, 2], 
               color='tan', linewidth=4, alpha=0.6)

    def _create_dress_triangles(self, dress_points):
        """Create triangular mesh for dress surface"""
        triangles = []
        
        # Create triangular strips for smooth dress surface
        for i in range(0, len(dress_points) - 3, 2):
            if i + 3 < len(dress_points):
                # Create two triangles for each "band" of the dress
                triangle1 = [dress_points[i], dress_points[i+1], dress_points[i+2]]
                triangle2 = [dress_points[i+1], dress_points[i+2], dress_points[i+3]]
                triangles.extend([triangle1, triangle2])
        
        return triangles

    def _create_side_panels(self, front_points, back_points):
        """Create side panels connecting front and back of dress"""
        side_triangles = []
        
        # Right side panels
        for i in range(0, len(front_points) - 1, 2):
            if i + 1 < len(front_points) and i + 1 < len(back_points):
                # Right side
                front_right = front_points[i] if front_points[i][0] > 0 else front_points[i+1]
                back_right = back_points[i] if back_points[i][0] > 0 else back_points[i+1]
                
                if i + 2 < len(front_points):
                    front_right_next = front_points[i+2] if front_points[i+2][0] > 0 else front_points[i+3] if i+3 < len(front_points) else front_points[i+2]
                    back_right_next = back_points[i+2] if back_points[i+2][0] > 0 else back_points[i+3] if i+3 < len(back_points) else back_points[i+2]
                    
                    # Create side panel triangles
                    triangle1 = [front_right, back_right, front_right_next]
                    triangle2 = [back_right, back_right_next, front_right_next]
                    side_triangles.extend([triangle1, triangle2])
        
        return side_triangles

    def _add_dress_details(self, ax, dress_front, dress_color, fitness_factor):
        """Add dress details like neckline, waistline"""
        
        # Neckline detail
        neckline_points = dress_front[:2]  # First two points are neckline
        if len(neckline_points) == 2:
            ax.plot([neckline_points[0][0], neckline_points[1][0]], 
                   [neckline_points[0][1], neckline_points[1][1]], 
                   [neckline_points[0][2], neckline_points[1][2]], 
                   color='darkviolet', linewidth=3, alpha=0.9)
        
        # Waistline detail (if fitness is high enough)
        if fitness_factor > 0.3:
            waist_points = dress_front[6:8]  # Waist area points
            if len(waist_points) == 2:
                ax.plot([waist_points[0][0], waist_points[1][0]], 
                       [waist_points[0][1], waist_points[1][1]], 
                       [waist_points[0][2], waist_points[1][2]], 
                       color='purple', linewidth=2, alpha=0.7, linestyle='--')
        
        # Hemline
        hem_points = dress_front[-2:]  # Last two points are hemline
        if len(hem_points) == 2:
            ax.plot([hem_points[0][0], hem_points[1][0]], 
                   [hem_points[0][1], hem_points[1][1]], 
                   [hem_points[0][2], hem_points[1][2]], 
                   color='darkmagenta', linewidth=2, alpha=0.8)
    
    def _create_summary_image(self, designs: List[FashionDesign], generation: int, results_folder: str):
        """Create a simple summary visualization"""
        import os
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot 1: Fitness distribution
        fitnesses = [d.fitness for d in designs]
        ax1.bar(range(len(fitnesses)), fitnesses, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Design Number')
        ax1.set_ylabel('Fitness Score')
        ax1.set_title(f'Generation {generation + 1} - Fitness Scores')
        ax1.grid(True, alpha=0.3)
        
        # Highlight best design
        if fitnesses:
            best_idx = fitnesses.index(max(fitnesses))
            ax1.bar(best_idx, fitnesses[best_idx], color='gold', alpha=0.9)
        
        # Plot 2: Simple dress silhouettes
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(0, 6)
        ax2.set_aspect('equal')
        
        # Draw simple dress shapes for top 5 designs
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for i, design in enumerate(designs[:5]):
            x_offset = (i - 2) * 1.2  # Spread them out
            
            # Simple dress outline
            dress_x = [x_offset - 0.3, x_offset + 0.3, x_offset + 0.4, x_offset - 0.4, x_offset - 0.3]
            dress_y = [5.5, 5.5, 1, 1, 5.5]  # Shoulders to hem
            
            ax2.plot(dress_x, dress_y, color=colors[i], linewidth=3, alpha=0.8)
            ax2.fill(dress_x, dress_y, color=colors[i], alpha=0.3)
            ax2.text(x_offset, 0.5, f'D{i+1}\n{design.fitness:.2f}', 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax2.set_title('Top 5 Dress Designs')
        ax2.set_xlabel('Design Variations')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        plt.tight_layout()
        summary_filename = os.path.join(results_folder, f'generation_{generation + 1}_summary.png')
        plt.savefig(summary_filename, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"   📊 Saved summary to {summary_filename}")
        
        plt.show(block=False)
        plt.pause(0.1)
        plt.close(fig)  # Close to prevent memory issues
    
    def save_best_design(self, designs: List[FashionDesign], filename: str):
        """Save the best design with all its data"""
        best_design = max(designs, key=lambda d: d.fitness)
        
        # Save design data
        design_data = best_design.to_dict()
        
        with open(f"{filename}.json", 'w') as f:
            json.dump(design_data, f, indent=2, default=str)
        
        # Save 3D mesh if available
        if best_design.mesh_3d is not None:
            best_design.mesh_3d.export(f"{filename}.obj")
        
        print(f"💾 Saved best design to {filename}")

def main():
    """Main execution function"""
    print("🎨 3D Fashion CPPN Evolution System")
    print("=" * 50)
    
    # Choose garment type
    garment_types = list(GarmentType)
    print("Available garment types:")
    for i, gt in enumerate(garment_types):
        print(f"  {i+1}. {gt.value}")
    
    try:
        choice = input("Choose garment type (1-5, default=1): ").strip()
        choice = int(choice) if choice else 1
        garment_type = garment_types[choice - 1]
    except (ValueError, IndexError):
        garment_type = GarmentType.JACKET
        print("Using default: jacket")
    
    # Initialize system
    system = Fashion3DSystem(garment_type)
    
    # Run evolution
    final_designs = system.run_evolution(generations=5, population_size=12)
    
    # Save best result
    system.save_best_design(final_designs, f"best_{garment_type.value}_design")
    
    print(f"\n✨ Created {len(final_designs)} unique 3D fashion designs!")

if __name__ == "__main__":
    main()