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
        cppn = GarmentCPPN(genome, self.garment_type)
        construction_params = self._generate_construction_parameters(cppn)
        
        # Step 2: Generate pattern pieces
        pattern_pieces = self.pattern_generator.generate_patterns(
            construction_params, self.garment_type
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
            return ConstructionParameters.from_cppn_outputs(params, self.garment_type)
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
        """Draw a realistic dress shape on the body"""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        measurements = self.body_model.get_measurements()
        
        # Draw body mannequin first
        body_outline = np.array([
            [0, measurements['neck_height'], measurements['neck_depth']/2],           # Neck
            [-measurements['shoulder_width']/2, measurements['shoulder_height'], 0], # Left shoulder
            [-measurements['chest_width']/2, measurements['chest_height'], measurements['chest_depth']/2],  # Left chest
            [-measurements['waist_width']/2, measurements['waist_height'], measurements['waist_depth']/2],  # Left waist
            [-measurements['hip_width']/2, measurements['hip_height'], measurements['hip_depth']/2],        # Left hip
            [0, measurements['hip_height'] - 200, measurements['hip_depth']/2],      # Bottom center
            [measurements['hip_width']/2, measurements['hip_height'], measurements['hip_depth']/2],         # Right hip
            [measurements['waist_width']/2, measurements['waist_height'], measurements['waist_depth']/2],   # Right waist
            [measurements['chest_width']/2, measurements['chest_height'], measurements['chest_depth']/2],   # Right chest
            [measurements['shoulder_width']/2, measurements['shoulder_height'], 0],  # Right shoulder
            [0, measurements['neck_height'], measurements['neck_depth']/2]           # Close shape
        ])
        
        # Plot body outline in light gray
        ax.plot(body_outline[:, 0], body_outline[:, 1], body_outline[:, 2], 
               color='lightgray', linewidth=1, alpha=0.7, linestyle='-')
        
        # Add head as a simple circle at the top
        head_center = [0, measurements['neck_height'] + 100, 0]
        ax.scatter(*head_center, c='lightgray', s=200, alpha=0.7, marker='o')
        
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
        bust_width = (measurements['bust_width'] + bust_ease) / 2
        waist_width = (measurements['waist_width'] + waist_ease) / 2
        hip_width = (measurements['hip_width'] + hip_ease) / 2
        hem_width = hip_width * 1.3  # A-line flare
        
        # Dress heights
        shoulder_height = measurements['shoulder_height']
        bust_height = measurements['bust_height']
        waist_height = measurements['waist_height']
        hip_height = measurements['hip_height']
        hem_height = measurements['hip_height'] - 400  # Knee-length dress
        
        # Create dress silhouette points
        dress_front = np.array([
            # Neckline
            [-measurements['neck_width']/3, measurements['neck_height'] - 50, measurements['neck_depth']/2 + 30],
            [measurements['neck_width']/3, measurements['neck_height'] - 50, measurements['neck_depth']/2 + 30],
            
            # Shoulder/armhole area
            [bust_width * 0.8, shoulder_height - 50, measurements['chest_depth']/2 + 20],
            [-bust_width * 0.8, shoulder_height - 50, measurements['chest_depth']/2 + 20],
            
            # Bust area
            [bust_width, bust_height, measurements['bust_depth']/2 + 20],
            [-bust_width, bust_height, measurements['bust_depth']/2 + 20],
            
            # Waist
            [waist_width, waist_height, measurements['waist_depth']/2 + 15],
            [-waist_width, waist_height, measurements['waist_depth']/2 + 15],
            
            # Hip
            [hip_width, hip_height, measurements['hip_depth']/2 + 15],
            [-hip_width, hip_height, measurements['hip_depth']/2 + 15],
            
            # Hem with curve
            [hem_width, hem_height + hem_curve, measurements['hip_depth']/2 + 10],
            [-hem_width, hem_height + hem_curve, measurements['hip_depth']/2 + 10],
        ])
        
        # Create back of dress (slightly different)
        dress_back = dress_front.copy()
        dress_back[:, 2] = -measurements['back_depth']/2 - 15  # Move to back
        
        # Color based on fitness
        color_intensity = design.fitness
        dress_color = [0.8, 0.4 + color_intensity * 0.4, 0.9 - color_intensity * 0.3]  # Pink to purple gradient
        
        # Create dress surfaces - simplified to avoid triangulation issues
        try:
            # Create front surface as a single polygon
            front_poly = Poly3DCollection([dress_front], alpha=0.8, linewidths=0.5)
            front_poly.set_facecolor(dress_color)
            front_poly.set_edgecolor('purple')
            ax.add_collection3d(front_poly)
            
            # Create back surface
            back_poly = Poly3DCollection([dress_back], alpha=0.6, linewidths=0.3)
            back_color = [c * 0.8 for c in dress_color]  # Darker back
            back_poly.set_facecolor(back_color)
            back_poly.set_edgecolor('darkmagenta')
            ax.add_collection3d(back_poly)
            
        except Exception as e:
            print(f"   Warning: Could not create dress surfaces: {e}")
            # Fallback to wireframe
            ax.plot(dress_front[:, 0], dress_front[:, 1], dress_front[:, 2], 
                   color='purple', linewidth=3, alpha=0.9, label='Dress')
        
        # Plot dress outline for clarity
        dress_outline = np.array([
            dress_front[0], dress_front[2], dress_front[4], dress_front[6], 
            dress_front[8], dress_front[10], dress_front[11], dress_front[9],
            dress_front[7], dress_front[5], dress_front[3], dress_front[1], dress_front[0]
        ])
        
        ax.plot(dress_outline[:, 0], dress_outline[:, 1], dress_outline[:, 2], 
               color='purple', linewidth=2, alpha=0.9)
        
        # Add dress label
        ax.text(0, measurements['shoulder_height'] + 100, 50, 
               f'Dress {design_number}\nFit: {design.fitness:.3f}', 
               ha='center', va='center', fontsize=9, fontweight='bold', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
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
    
    def display_generation(self, designs: List[FashionDesign], generation: int):
        """Display the current generation of designs"""
        # Create visualization
        n_designs = min(len(designs), 9)  # Show top 9
        fig = plt.figure(figsize=(15, 10))
        
        for i in range(n_designs):
            design = designs[i]
            
            # Create subplot for 3D visualization
            ax = fig.add_subplot(3, 3, i+1, projection='3d')
            
            if design.mesh_3d is not None:
                # Plot garment mesh as actual surfaces
                vertices = design.mesh_3d.vertices
                faces = design.mesh_3d.faces
                
                if len(vertices) > 0 and len(faces) > 0:
                    # Create 3D surface plot
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                    
                    # Collect all triangular faces
                    triangles = []
                    for face in faces:
                        if len(face) >= 3 and all(0 <= i < len(vertices) for i in face[:3]):
                            triangle_verts = vertices[face[:3]]
                            triangles.append(triangle_verts)
                    
                    if triangles:
                        # Create 3D surface collection
                        poly3d = Poly3DCollection(triangles, alpha=0.7, linewidths=0.5)
                        poly3d.set_facecolor('lightblue')
                        poly3d.set_edgecolor('navy')
                        ax.add_collection3d(poly3d)
                        
                        # Also plot some key vertices for structure
                        ax.scatter(vertices[::5, 0], vertices[::5, 1], vertices[::5, 2], 
                                 c='darkblue', alpha=0.8, s=10)
                    else:
                        # Fallback: show as wireframe
                        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                                 c='blue', alpha=0.8, s=20)
                        
                        # Draw wireframe
                        for face in faces[::2]:
                            if len(face) >= 3 and all(0 <= i < len(vertices) for i in face[:3]):
                                triangle = vertices[face[:3]]
                                # Close the triangle
                                triangle_closed = np.vstack([triangle, triangle[0]])
                                ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], 
                                       triangle_closed[:, 2], 'b-', alpha=0.8, linewidth=2)
                else:
                    ax.text(0, 1000, 0, f'Design {i+1}\n(Empty Mesh)', ha='center', va='center')
            else:
                # Create a simple visual representation if no mesh
                # Draw a basic dress shape
                chest_width = 260  # Half of chest measurement
                hip_width = 330    # Half of hip measurement  
                shoulder_height = 1420
                waist_height = 1080
                hip_height = 920
                hem_height = 600
                
                # Create simple dress silhouette
                dress_points = np.array([
                    # Left side
                    [-chest_width/2, shoulder_height, 100],  # Left shoulder
                    [-chest_width/2, waist_height, 100],     # Left waist
                    [-hip_width/2, hip_height, 100],         # Left hip
                    [-hip_width/2, hem_height, 100],         # Left hem
                    # Right side  
                    [hip_width/2, hem_height, 100],          # Right hem
                    [hip_width/2, hip_height, 100],          # Right hip
                    [chest_width/2, waist_height, 100],      # Right waist
                    [chest_width/2, shoulder_height, 100],   # Right shoulder
                    [-chest_width/2, shoulder_height, 100]   # Close the shape
                ])
                
                # Plot dress outline
                ax.plot(dress_points[:, 0], dress_points[:, 1], dress_points[:, 2], 
                       'b-', linewidth=3, alpha=0.8, label='Dress Shape')
                
                # Fill the dress area (front face)
                front_face = dress_points[:-1]  # Remove duplicate point
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                poly = Poly3DCollection([front_face], alpha=0.5)
                poly.set_facecolor('lightblue')
                poly.set_edgecolor('navy')
                ax.add_collection3d(poly)
                
                # Add back face for 3D effect
                back_points = dress_points.copy()
                back_points[:, 2] = -100  # Move to back
                back_face = back_points[:-1]
                poly_back = Poly3DCollection([back_face], alpha=0.3)
                poly_back.set_facecolor('lightgray')
                poly_back.set_edgecolor('gray')
                ax.add_collection3d(poly_back)
                
                ax.text(0, 800, 200, f'Dress {i+1}', ha='center', va='center', 
                       fontsize=10, fontweight='bold')
            
            # Plot body silhouette for reference - draw actual body shape
            measurements = self.body_model.get_measurements()
            
            # Create simple body mannequin silhouette
            body_outline = np.array([
                # Front body outline
                [0, measurements['neck_height'], measurements['neck_depth']/2],           # Neck
                [-measurements['shoulder_width']/2, measurements['shoulder_height'], 0], # Left shoulder
                [-measurements['chest_width']/2, measurements['chest_height'], measurements['chest_depth']/2],  # Left chest
                [-measurements['waist_width']/2, measurements['waist_height'], measurements['waist_depth']/2],  # Left waist
                [-measurements['hip_width']/2, measurements['hip_height'], measurements['hip_depth']/2],        # Left hip
                [0, measurements['hip_height'] - 200, measurements['hip_depth']/2],      # Bottom center
                [measurements['hip_width']/2, measurements['hip_height'], measurements['hip_depth']/2],         # Right hip
                [measurements['waist_width']/2, measurements['waist_height'], measurements['waist_depth']/2],   # Right waist
                [measurements['chest_width']/2, measurements['chest_height'], measurements['chest_depth']/2],   # Right chest
                [measurements['shoulder_width']/2, measurements['shoulder_height'], 0],  # Right shoulder
                [0, measurements['neck_height'], measurements['neck_depth']/2]           # Close shape
            ])
            
            # Plot body outline in light gray
            ax.plot(body_outline[:, 0], body_outline[:, 1], body_outline[:, 2], 
                   color='lightgray', linewidth=1, alpha=0.7, linestyle='-')
            
            # Add head as a simple circle at the top
            head_center = [0, measurements['neck_height'] + 100, 0]
            ax.scatter(*head_center, c='lightgray', s=200, alpha=0.7, marker='o')
            
            if design.mesh_3d is not None:
                # If we have a mesh, try to plot it as dress surfaces
                vertices = design.mesh_3d.vertices
                faces = design.mesh_3d.faces
                
                if len(vertices) > 0 and len(faces) > 0:
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                    
                    # Collect triangular faces for the dress
                    triangles = []
                    for face in faces[::2]:  # Sample every other face for performance
                        if len(face) >= 3 and all(0 <= i < len(vertices) for i in face[:3]):
                            triangle_verts = vertices[face[:3]]
                            triangles.append(triangle_verts)
                    
                    if triangles:
                        # Create dress surface
                        poly3d = Poly3DCollection(triangles, alpha=0.8, linewidths=0.5)
                        
                        # Color based on fitness - better dresses are more blue
                        color_intensity = design.fitness
                        dress_color = [0.2, 0.4 + color_intensity * 0.6, 0.8 + color_intensity * 0.2]
                        
                        poly3d.set_facecolor(dress_color)
                        poly3d.set_edgecolor('navy')
                        ax.add_collection3d(poly3d)
                        
                        ax.text(0, measurements['shoulder_height'] + 150, 0, 
                               f'Dress {i+1}', ha='center', va='center', 
                               fontsize=9, fontweight='bold', color='darkblue')
                    else:
                        # Fallback to creating a beautiful dress shape
                        self._draw_realistic_dress(ax, measurements, design, i)
                else:
                    # Create a beautiful dress shape
                    self._draw_realistic_dress(ax, measurements, design, i)
            else:
                # Create a beautiful dress shape
                self._draw_realistic_dress(ax, measurements, design, i)
    
    def _draw_realistic_dress(self, ax, measurements, design, i):
        """Draw a realistic dress shape on the body"""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
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
        bust_width = (measurements['bust_width'] + bust_ease) / 2
        waist_width = (measurements['waist_width'] + waist_ease) / 2
        hip_width = (measurements['hip_width'] + hip_ease) / 2
        hem_width = hip_width * 1.3  # A-line flare
        
        # Dress heights
        shoulder_height = measurements['shoulder_height']
        bust_height = measurements['bust_height']
        waist_height = measurements['waist_height']
        hip_height = measurements['hip_height']
        hem_height = measurements['hip_height'] - 400  # Knee-length dress
        
        # Create dress silhouette points
        dress_front = np.array([
            # Neckline
            [-measurements['neck_width']/3, measurements['neck_height'] - 50, measurements['neck_depth']/2 + 30],
            [measurements['neck_width']/3, measurements['neck_height'] - 50, measurements['neck_depth']/2 + 30],
            
            # Shoulder/armhole area
            [bust_width * 0.8, shoulder_height - 50, measurements['chest_depth']/2 + 20],
            [-bust_width * 0.8, shoulder_height - 50, measurements['chest_depth']/2 + 20],
            
            # Bust area
            [bust_width, bust_height, measurements['bust_depth']/2 + 20],
            [-bust_width, bust_height, measurements['bust_depth']/2 + 20],
            
            # Waist
            [waist_width, waist_height, measurements['waist_depth']/2 + 15],
            [-waist_width, waist_height, measurements['waist_depth']/2 + 15],
            
            # Hip
            [hip_width, hip_height, measurements['hip_depth']/2 + 15],
            [-hip_width, hip_height, measurements['hip_depth']/2 + 15],
            
            # Hem with curve
            [hem_width, hem_height + hem_curve, measurements['hip_depth']/2 + 10],
            [-hem_width, hem_height + hem_curve, measurements['hip_depth']/2 + 10],
        ])
        
        # Create back of dress (slightly different)
        dress_back = dress_front.copy()
        dress_back[:, 2] = -measurements['back_depth']/2 - 15  # Move to back
        
        # Create dress surfaces
        
        # Front panels
        front_triangles = []
        for idx in range(len(dress_front) - 2):
            if idx % 2 == 0:  # Left side
                triangle = [dress_front[idx], dress_front[idx+2], dress_front[idx+1]]
                front_triangles.append(triangle)
            
        # Side panels connecting front to back
        side_triangles = []
        for j in range(0, len(dress_front) - 1, 2):
            if j + 1 < len(dress_front):
                # Right side panel
                side_triangles.extend([
                    [dress_front[j], dress_back[j], dress_front[j+1]],
                    [dress_back[j], dress_back[j+1], dress_front[j+1]]
                ])
                
                # Left side panel  
                if j + 2 < len(dress_front):
                    side_triangles.extend([
                        [dress_front[j+1], dress_back[j+1], dress_front[j+2]],
                        [dress_back[j+1], dress_back[j+2], dress_front[j+2]]
                    ])
        
        # Color based on fitness
        color_intensity = design.fitness
        dress_color = [0.8, 0.4 + color_intensity * 0.4, 0.9 - color_intensity * 0.3]  # Pink to purple gradient
        
        # Plot front surface
        if front_triangles:
            front_poly = Poly3DCollection(front_triangles, alpha=0.8, linewidths=0.5)
            front_poly.set_facecolor(dress_color)
            front_poly.set_edgecolor('purple')
            ax.add_collection3d(front_poly)
        
        # Plot side surfaces
        if side_triangles:
            side_poly = Poly3DCollection(side_triangles, alpha=0.7, linewidths=0.3)
            side_color = [c * 0.8 for c in dress_color]  # Darker sides
            side_poly.set_facecolor(side_color)
            side_poly.set_edgecolor('darkmagenta')
            ax.add_collection3d(side_poly)
        
        # Plot dress outline for clarity
        dress_outline = np.array([
            dress_front[0], dress_front[2], dress_front[4], dress_front[6], 
            dress_front[8], dress_front[10], dress_front[11], dress_front[9],
            dress_front[7], dress_front[5], dress_front[3], dress_front[1], dress_front[0]
        ])
        
        ax.plot(dress_outline[:, 0], dress_outline[:, 1], dress_outline[:, 2], 
               color='purple', linewidth=2, alpha=0.9)
        
        # Add dress label
        ax.text(0, measurements['shoulder_height'] + 100, 50, 
               f'Dress {i+1}\nFit: {design.fitness:.3f}', 
               ha='center', va='center', fontsize=9, fontweight='bold', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
        ax.set_title(f'Design {i+1}\nFitness: {design.fitness:.3f}', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        max_range = 300  # mm
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, 600])
        
        plt.suptitle(f'3D Fashion Evolution - Generation {generation + 1}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'generation_{generation + 1}_3d.png', dpi=150, bbox_inches='tight')
        plt.show()
    
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