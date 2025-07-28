# main.py - Enhanced 3D Fashion CPPN Evolution System with Beautiful Dresses
"""
Enhanced 3D Fashion CPPN Evolution System
Generates beautiful, realistic dress visualizations that evolve over generations.
Creates stunning fashion designs that look like they belong in a design studio.
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
import os
import datetime

# Import our custom modules
try:
    from garment_genome import GarmentGenome, GarmentCPPN, ConstructionParameters
    from body_model import HumanBodyModel
    from pattern_generator import PatternGenerator
    from cloth_simulator import ClothSimulator
    from fashion_renderer import EnhancedFashion3DRenderer  # Use enhanced renderer
    from evolution_engine import FashionEvolutionEngine
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all files are in the same directory and fashion_renderer.py is updated")
    exit(1)

class GarmentType(Enum):
    JACKET = "jacket"
    DRESS = "dress"
    SHIRT = "shirt"
    PANTS = "pants"
    SKIRT = "skirt"

@dataclass
class FashionDesign:
    """Complete fashion design with enhanced visualization data"""
    genome: GarmentGenome
    construction_params: ConstructionParameters
    pattern_pieces: List[Dict]
    mesh_3d: Optional[trimesh.Trimesh] = None
    fitness: float = 0.0
    generation: int = 0
    style_characteristics: Dict = field(default_factory=dict)
    color_palette: List = field(default_factory=list)
    design_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'genome': self.genome.to_dict(),
            'construction_params': self.construction_params.to_dict(),
            'pattern_pieces': self.pattern_pieces,
            'fitness': self.fitness,
            'generation': self.generation,
            'style_characteristics': self.style_characteristics,
            'color_palette': self.color_palette,
            'design_notes': self.design_notes
        }

class Enhanced3DFashionSystem:
    """Enhanced system for beautiful 3D fashion evolution"""
    
    def __init__(self, garment_type: GarmentType = GarmentType.DRESS):
        self.garment_type = garment_type
        self.body_model = HumanBodyModel()
        self.pattern_generator = PatternGenerator(self.body_model)
        self.cloth_simulator = ClothSimulator()
        self.renderer = EnhancedFashion3DRenderer()  # Use enhanced renderer
        self.evolution_engine = FashionEvolutionEngine()
        
        # Fashion design parameters
        self.design_themes = [
            'elegant_evening', 'casual_chic', 'business_professional', 
            'bohemian_artistic', 'modern_minimalist', 'romantic_vintage'
        ]
        
        print(f"✨ Enhanced 3D Fashion System Initialized")
        print(f"🎨 Garment Type: {garment_type.value}")
        print(f"👗 Ready to create beautiful fashion designs!")
    
    def create_enhanced_design_from_genome(self, genome: GarmentGenome) -> FashionDesign:
        """Transform a genome into a beautiful fashion design"""
        
        # Step 1: Generate enhanced construction parameters
        cppn = GarmentCPPN(genome, self.garment_type.value)
        construction_params = self._generate_enhanced_construction_parameters(cppn)
        
        # Step 2: Determine style characteristics based on genome
        style_characteristics = self._analyze_genome_style(genome)
        
        # Step 3: Generate color palette based on fitness and style
        color_palette = self._generate_color_palette(genome.fitness, style_characteristics)
        
        # Step 4: Generate pattern pieces with style influence
        pattern_pieces = self.pattern_generator.generate_patterns(
            construction_params, self.garment_type.value
        )
        
        # Step 5: Enhanced pattern processing for better visualization
        pattern_pieces = self._enhance_patterns_for_visualization(pattern_pieces, style_characteristics)
        
        # Step 6: Create or simulate 3D mesh (with fallback to procedural generation)
        mesh_3d = self._create_enhanced_3d_mesh(pattern_pieces, construction_params, style_characteristics)
        
        # Step 7: Generate design notes
        design_notes = self._generate_design_notes(construction_params, style_characteristics, genome.fitness)
        
        # Create enhanced design
        design = FashionDesign(
            genome=genome,
            construction_params=construction_params,
            pattern_pieces=pattern_pieces,
            mesh_3d=mesh_3d,
            style_characteristics=style_characteristics,
            color_palette=color_palette,
            design_notes=design_notes
        )
        
        return design
    
    def _generate_enhanced_construction_parameters(self, cppn: GarmentCPPN) -> ConstructionParameters:
        """Generate enhanced construction parameters with fashion sensibility"""
        
        measurements = self.body_model.get_measurements()
        construction_points = self._get_fashion_evaluation_points()
        
        params = {}
        for point_name, (x, y, z) in construction_points.items():
            try:
                # Normalize coordinates with fashion-aware scaling
                chest_width = measurements.get('chest_width', 520)
                waist_height = measurements.get('waist_height', 1080)
                torso_height = measurements.get('torso_height', 340)
                body_depth = measurements.get('body_depth', 220)
                
                norm_x = x / chest_width if chest_width > 0 else 0
                norm_y = (y - waist_height) / torso_height if torso_height > 0 else 0
                norm_z = z / body_depth if body_depth > 0 else 0
                
                # Evaluate CPPN with fashion-specific inputs
                outputs = cppn.evaluate(norm_x, norm_y, norm_z)
                
                # Add fashion-specific enhancements
                outputs['style_factor'] = self._calculate_style_factor(norm_x, norm_y, norm_z)
                outputs['elegance'] = self._calculate_elegance_factor(outputs)
                
                params[point_name] = outputs
                
            except Exception as e:
                print(f"   Warning: CPPN evaluation failed for {point_name}: {e}")
                params[point_name] = self._get_default_fashion_params()
        
        try:
            enhanced_params = ConstructionParameters.from_cppn_outputs(params, self.garment_type.value)
            return self._apply_fashion_enhancements(enhanced_params, params)
        except Exception as e:
            print(f"   Warning: Using default construction parameters: {e}")
            return ConstructionParameters.create_default(self.garment_type.value)
    
    def _get_fashion_evaluation_points(self) -> Dict[str, Tuple[float, float, float]]:
        """Get fashion-specific evaluation points for dress design"""
        measurements = self.body_model.get_measurements()
        
        if self.garment_type == GarmentType.DRESS:
            return {
                'neckline_style': (0, measurements['neck_height'], measurements.get('neck_depth', 140)/2),
                'bust_definition': (measurements.get('bust_width', 500)/2, measurements.get('bust_height', 1260), measurements.get('bust_depth', 250)/2),
                'waist_cinch': (measurements['waist_width']/2, measurements['waist_height'], measurements['waist_depth']/2),
                'hip_flow': (measurements['hip_width']/2, measurements['hip_height'], measurements['hip_depth']/2),
                'hem_drama': (measurements.get('hip_width', 550)/2 * 1.3, measurements.get('hem_height', 600), 0),
                'shoulder_elegance': (measurements['shoulder_width']/2, measurements['shoulder_height'], 0),
                'side_silhouette': (measurements['hip_width']/2, measurements['waist_height'], 0),
                'back_sophistication': (0, measurements['chest_height'], -measurements['back_depth']/2),
            }
        else:
            # Default points for other garments
            return {
                'center_front': (0, measurements['chest_height'], measurements['chest_depth']/2),
                'center_back': (0, measurements['chest_height'], -measurements['back_depth']/2),
                'side_seam': (measurements['chest_width']/2, measurements['waist_height'], 0),
            }
    
    def _calculate_style_factor(self, x: float, y: float, z: float) -> float:
        """Calculate style factor based on position"""
        # Combine coordinates to create style influence
        return math.sin(x * math.pi) * math.cos(y * math.pi) * 0.5 + 0.5
    
    def _calculate_elegance_factor(self, outputs: Dict) -> float:
        """Calculate elegance factor from CPPN outputs"""
        # Combine multiple outputs to determine elegance
        elegance = 0.0
        for key, value in outputs.items():
            if isinstance(value, (int, float)):
                elegance += abs(value) * 0.1
        return min(elegance, 1.0)
    
    def _get_default_fashion_params(self) -> Dict:
        """Get default fashion parameters"""
        return {
            'ease': random.uniform(-0.3, 0.3),
            'dart': random.uniform(-0.2, 0.2),
            'curve': random.uniform(-0.4, 0.4),
            'suppression': random.uniform(-0.2, 0.2),
            'slope': random.uniform(-0.1, 0.1),
            'asymmetry': random.uniform(-0.1, 0.1),
            'style_factor': random.uniform(0.3, 0.7),
            'elegance': random.uniform(0.4, 0.8)
        }
    
    def _apply_fashion_enhancements(self, params: ConstructionParameters, cppn_params: Dict) -> ConstructionParameters:
        """Apply fashion-specific enhancements to construction parameters"""
        
        # Calculate overall style score
        style_scores = []
        for point_params in cppn_params.values():
            if 'style_factor' in point_params:
                style_scores.append(point_params['style_factor'])
        
        overall_style = sum(style_scores) / len(style_scores) if style_scores else 0.5
        
        # Enhance ease amounts for better fit
        if hasattr(params, 'ease_amounts'):
            for key in params.ease_amounts:
                # More sophisticated ease calculation
                base_ease = params.ease_amounts[key]
                style_modifier = (overall_style - 0.5) * 20  # -10 to +10mm
                params.ease_amounts[key] = max(base_ease + style_modifier, 10)  # Minimum 10mm ease
        
        # Enhance waist suppression for better silhouette
        if hasattr(params, 'waist_suppression'):
            elegance_scores = [p.get('elegance', 0.5) for p in cppn_params.values()]
            avg_elegance = sum(elegance_scores) / len(elegance_scores) if elegance_scores else 0.5
            params.waist_suppression = avg_elegance * 0.4  # 0 to 0.4 suppression
        
        return params
    
    def _analyze_genome_style(self, genome: GarmentGenome) -> Dict:
        """Analyze genome to determine style characteristics"""
        
        # Analyze genome structure for style cues
        num_nodes = len(genome.nodes)
        num_connections = len(genome.connections)
        complexity = (num_nodes + num_connections) / 30.0  # Normalize complexity
        
        # Analyze activation functions for style personality
        activations = [node.get('activation', 'linear') for node in genome.nodes]
        activation_diversity = len(set(activations)) / max(len(activations), 1)
        
        # Determine style characteristics
        style_chars = {
            'complexity': min(complexity, 1.0),
            'diversity': activation_diversity,
            'elegance': genome.fitness,  # Higher fitness = more elegant
            'drama': complexity * activation_diversity,
            'sophistication': genome.fitness * activation_diversity,
        }
        
        # Determine primary style theme
        if style_chars['elegance'] > 0.8 and style_chars['sophistication'] > 0.6:
            style_chars['theme'] = 'elegant_evening'
        elif style_chars['drama'] > 0.7:
            style_chars['theme'] = 'bohemian_artistic'
        elif style_chars['complexity'] < 0.3:
            style_chars['theme'] = 'modern_minimalist'
        elif style_chars['elegance'] > 0.6:
            style_chars['theme'] = 'romantic_vintage'
        else:
            style_chars['theme'] = 'casual_chic'
        
        return style_chars
    
    def _generate_color_palette(self, fitness: float, style_characteristics: Dict) -> List:
        """Generate beautiful color palette based on design characteristics"""
        
        theme = style_characteristics.get('theme', 'casual_chic')
        elegance = style_characteristics.get('elegance', 0.5)
        
        # Theme-based color palettes
        theme_palettes = {
            'elegant_evening': [
                [(0.1, 0.1, 0.2), (0.2, 0.1, 0.3), (0.3, 0.2, 0.4)],  # Deep purples
                [(0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3)],  # Elegant blacks
                [(0.2, 0.1, 0.1), (0.3, 0.2, 0.2), (0.4, 0.3, 0.3)]   # Deep burgundy
            ],
            'romantic_vintage': [
                [(1.0, 0.8, 0.9), (0.9, 0.7, 0.8), (1.0, 0.9, 0.95)],  # Soft pinks
                [(0.9, 0.8, 0.7), (0.8, 0.7, 0.6), (0.95, 0.85, 0.75)], # Cream/beige
                [(0.7, 0.8, 0.9), (0.6, 0.7, 0.8), (0.8, 0.85, 0.9)]   # Soft blues
            ],
            'modern_minimalist': [
                [(1.0, 1.0, 1.0), (0.95, 0.95, 0.95), (0.9, 0.9, 0.9)], # Pure whites
                [(0.8, 0.8, 0.8), (0.7, 0.7, 0.7), (0.6, 0.6, 0.6)],   # Clean grays
                [(0.9, 0.9, 0.8), (0.85, 0.85, 0.75), (0.8, 0.8, 0.7)] # Off-whites
            ],
            'bohemian_artistic': [
                [(0.8, 0.5, 0.2), (0.7, 0.4, 0.1), (0.9, 0.6, 0.3)],   # Warm oranges
                [(0.6, 0.3, 0.5), (0.5, 0.2, 0.4), (0.7, 0.4, 0.6)],   # Deep purples
                [(0.3, 0.6, 0.4), (0.2, 0.5, 0.3), (0.4, 0.7, 0.5)]    # Earth greens
            ],
            'casual_chic': [
                [(0.4, 0.6, 0.8), (0.3, 0.5, 0.7), (0.5, 0.7, 0.9)],   # Casual blues
                [(0.8, 0.6, 0.4), (0.7, 0.5, 0.3), (0.9, 0.7, 0.5)],   # Warm tans
                [(0.6, 0.8, 0.6), (0.5, 0.7, 0.5), (0.7, 0.9, 0.7)]    # Fresh greens
            ],
            'business_professional': [
                [(0.2, 0.2, 0.3), (0.1, 0.1, 0.2), (0.3, 0.3, 0.4)],   # Navy blues
                [(0.3, 0.3, 0.3), (0.2, 0.2, 0.2), (0.4, 0.4, 0.4)],   # Charcoal grays
                [(0.2, 0.3, 0.2), (0.1, 0.2, 0.1), (0.3, 0.4, 0.3)]    # Forest greens
            ]
        }
        
        # Select palette based on theme and fitness
        theme_options = theme_palettes.get(theme, theme_palettes['casual_chic'])
        palette_index = min(int(elegance * len(theme_options)), len(theme_options) - 1)
        
        return theme_options[palette_index]
    
    def _enhance_patterns_for_visualization(self, patterns: List[Dict], style_characteristics: Dict) -> List[Dict]:
        """Enhance pattern pieces for better visualization"""
        
        enhanced_patterns = []
        
        for pattern in patterns:
            enhanced_pattern = pattern.copy()
            
            # Add style-specific modifications to vertices if needed
            if 'vertices' in pattern and len(pattern['vertices']) > 0:
                vertices = np.array(pattern['vertices'])
                
                # Apply style-based modifications
                drama_factor = style_characteristics.get('drama', 0.5)
                if drama_factor > 0.7:  # High drama = more flowing shapes
                    # Add slight curve to straight edges
                    enhanced_pattern['style_modifications'] = 'dramatic_curves'
                elif drama_factor < 0.3:  # Low drama = cleaner lines
                    enhanced_pattern['style_modifications'] = 'clean_lines'
                
                enhanced_pattern['style_score'] = style_characteristics.get('elegance', 0.5)
            
            enhanced_patterns.append(enhanced_pattern)
        
        return enhanced_patterns
    
    def _create_enhanced_3d_mesh(self, patterns: List[Dict], construction_params: ConstructionParameters, 
                                style_characteristics: Dict) -> Optional[trimesh.Trimesh]:
        """Create enhanced 3D mesh with fallback to beautiful procedural generation"""
        
        try:
            # Try the cloth simulator first
            mesh_3d = self.cloth_simulator.simulate_garment(
                patterns, self.body_model, construction_params
            )
            
            if mesh_3d is not None and len(mesh_3d.vertices) > 10:
                return mesh_3d
                
        except Exception as e:
            print(f"   Cloth simulation failed: {e}")
        
        # Fallback to procedural mesh generation based on style
        return self._create_procedural_dress_mesh(construction_params, style_characteristics)
    
    def _create_procedural_dress_mesh(self, construction_params: ConstructionParameters, 
                                    style_characteristics: Dict) -> trimesh.Trimesh:
        """Create beautiful procedural dress mesh"""
        
        measurements = self.body_model.get_measurements()
        
        # Generate dress shape based on style
        theme = style_characteristics.get('theme', 'casual_chic')
        elegance = style_characteristics.get('elegance', 0.5)
        drama = style_characteristics.get('drama', 0.5)
        
        # Define dress parameters
        if theme == 'elegant_evening':
            waist_factor = 0.6
            hem_flare = 2.0 + drama * 0.5
            length_factor = 1.2
        elif theme == 'bohemian_artistic':
            waist_factor = 0.7
            hem_flare = 2.2 + drama * 0.8
            length_factor = 1.1
        elif theme == 'modern_minimalist':
            waist_factor = 0.8
            hem_flare = 1.2
            length_factor = 1.0
        else:  # casual_chic, romantic_vintage, business_professional
            waist_factor = 0.7
            hem_flare = 1.5 + drama * 0.3
            length_factor = 1.0
        
        # Create dress geometry
        vertices, faces = self._generate_dress_mesh_geometry(
            measurements, waist_factor, hem_flare, length_factor, elegance
        )
        
        try:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.remove_degenerate_faces()
            return mesh
        except Exception as e:
            print(f"   Procedural mesh creation failed: {e}")
            return self._create_minimal_dress_mesh(measurements)
    
    def _generate_dress_mesh_geometry(self, measurements: Dict, waist_factor: float, 
                                    hem_flare: float, length_factor: float, elegance: float):
        """Generate detailed dress mesh geometry"""
        
        # Base measurements
        shoulder_w = measurements.get('shoulder_width', 460) / 2
        bust_w = (measurements.get('bust_width', 500) + 60) / 2  # Add ease
        waist_w = (measurements.get('waist_width', 420) * waist_factor + 40) / 2
        hip_w = (measurements.get('hip_width', 550) + 80) / 2
        hem_w = hip_w * hem_flare
        
        # Heights
        shoulder_h = measurements.get('shoulder_height', 1420)
        bust_h = measurements.get('bust_height', 1260)
        waist_h = measurements.get('waist_height', 1080)
        hip_h = measurements.get('hip_height', 920)
        hem_h = hip_h - (350 * length_factor)
        
        # Create dress silhouette points
        front_silhouette = [
            (0, shoulder_h + 20, 30),  # Neckline center
            (shoulder_w * 0.3, shoulder_h + 10, 35),  # Neckline side
            (shoulder_w * 0.8, shoulder_h - 10, 30),  # Shoulder
            (bust_w, bust_h + 30, 40),  # Armhole
            (bust_w * (1 + elegance * 0.2), bust_h, 45),  # Bust
            (waist_w, waist_h, 35),  # Waist
            (hip_w, hip_h, 30),  # Hip
            (hem_w, hem_h, 25),  # Hem
            (0, hem_h, 25),  # Center front hem
        ]
        
        # Mirror for back (slightly different proportions)
        back_silhouette = []
        for x, y, z in front_silhouette:
            back_x = x * 0.95
            back_z = -z - 30
            back_silhouette.append((back_x, y, back_z))
        
        # Generate mesh vertices and faces
        vertices = []
        faces = []
        
        # Add front and back silhouette points
        vertices.extend(front_silhouette)
        vertices.extend(back_silhouette)
        
        # Create triangular faces
        n_front = len(front_silhouette)
        n_back = len(back_silhouette)
        
        # Front triangles
        for i in range(n_front - 2):
            faces.append([0, i + 1, i + 2])
        
        # Back triangles
        for i in range(n_back - 2):
            faces.append([n_front, n_front + i + 1, n_front + i + 2])
        
        # Side triangles connecting front and back
        for i in range(min(n_front, n_back) - 1):
            # Right side
            faces.append([i, n_front + i, i + 1])
            faces.append([n_front + i, n_front + i + 1, i + 1])
        
        return np.array(vertices), np.array(faces)
    
    def _create_minimal_dress_mesh(self, measurements: Dict) -> trimesh.Trimesh:
        """Create minimal fallback dress mesh"""
        
        # Very simple dress shape
        w = measurements.get('hip_width', 550) / 2 + 100
        h_top = measurements.get('shoulder_height', 1420)
        h_bottom = measurements.get('hip_height', 920) - 300
        
        vertices = np.array([
            [-w/2, h_top, 30], [w/2, h_top, 30], [w, h_bottom, 20],
            [-w, h_bottom, 20], [0, h_top, -30], [w/2, h_bottom, -20], [-w/2, h_bottom, -20]
        ])
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [1, 4, 5], [0, 3, 6], [0, 6, 4]
        ])
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _generate_design_notes(self, construction_params: ConstructionParameters, 
                             style_characteristics: Dict, fitness: float) -> List[str]:
        """Generate descriptive design notes"""
        
        notes = []
        theme = style_characteristics.get('theme', 'casual_chic')
        elegance = style_characteristics.get('elegance', 0.5)
        
        # Theme-based descriptions
        theme_descriptions = {
            'elegant_evening': "Sophisticated evening wear with refined silhouette",
            'romantic_vintage': "Soft, feminine design with vintage charm",
            'modern_minimalist': "Clean lines and contemporary sophistication", 
            'bohemian_artistic': "Free-flowing design with artistic flair",
            'casual_chic': "Effortlessly stylish everyday elegance",
            'business_professional': "Polished, professional wardrobe staple"
        }
        
        notes.append(theme_descriptions.get(theme, "Beautiful custom design"))
        
        # Fit and construction notes
        if hasattr(construction_params, 'ease_amounts'):
            avg_ease = sum(construction_params.ease_amounts.values()) / len(construction_params.ease_amounts)
            if avg_ease > 80:
                notes.append("Relaxed, comfortable fit")
            elif avg_ease < 50:
                notes.append("Fitted, tailored silhouette")
            else:
                notes.append("Classic, flattering fit")
        
        # Quality assessment
        if fitness > 0.8:
            notes.append("Premium design quality")
        elif fitness > 0.6:
            notes.append("Well-crafted design")
        else:
            notes.append("Experimental design concept")
        
        # Style details
        if elegance > 0.7:
            notes.append("Elegant draping and proportions")
        
        return notes
    
    def evaluate_design_beauty(self, design: FashionDesign) -> float:
        """Evaluate design beauty and wearability"""
        
        score = 0.0
        
        # Base fitness contribution (40%)
        score += design.fitness * 0.4
        
        # Style coherence (30%)
        style_score = self._evaluate_style_coherence(design)
        score += style_score * 0.3
        
        # Visual appeal (20%)
        visual_score = self._evaluate_visual_appeal(design)
        score += visual_score * 0.2
        
        # Construction quality (10%)
        construction_score = self._evaluate_construction_quality(design)
        score += construction_score * 0.1
        
        return min(score, 1.0)
    
    def _evaluate_style_coherence(self, design: FashionDesign) -> float:
        """Evaluate how well style elements work together"""
        
        style_chars = design.style_characteristics
        theme = style_chars.get('theme', 'casual_chic')
        elegance = style_chars.get('elegance', 0.5)
        complexity = style_chars.get('complexity', 0.5)
        
        # Check if complexity matches theme expectations
        theme_complexity_expectations = {
            'elegant_evening': (0.6, 1.0),
            'modern_minimalist': (0.0, 0.4),
            'bohemian_artistic': (0.7, 1.0),
            'romantic_vintage': (0.4, 0.8),
            'casual_chic': (0.2, 0.7),
            'business_professional': (0.3, 0.6)
        }
        
        expected_range = theme_complexity_expectations.get(theme, (0.3, 0.7))
        if expected_range[0] <= complexity <= expected_range[1]:
            coherence = 1.0
        else:
            # Calculate how far off we are
            if complexity < expected_range[0]:
                coherence = complexity / expected_range[0]
            else:
                coherence = expected_range[1] / complexity
        
        # Elegance should generally align with fitness
        elegance_alignment = 1.0 - abs(elegance - design.fitness)
        
        return (coherence + elegance_alignment) / 2
    
    def _evaluate_visual_appeal(self, design: FashionDesign) -> float:
        """Evaluate visual appeal of the design"""
        
        score = 0.0
        
        # Color palette harmony
        colors = design.color_palette
        if len(colors) >= 2:
            # Check color harmony (simplified)
            color_variance = np.var([sum(color) for color in colors])
            if 0.1 <= color_variance <= 0.3:  # Good color variation
                score += 0.4
            else:
                score += 0.2
        
        # Style characteristics appeal
        style_chars = design.style_characteristics
        drama = style_chars.get('drama', 0.5)
        sophistication = style_chars.get('sophistication', 0.5)
        
        # Balance between drama and sophistication
        balance = 1.0 - abs(drama - sophistication)
        score += balance * 0.3
        
        # Pattern complexity appeal
        pattern_count = len(design.pattern_pieces)
        if 3 <= pattern_count <= 8:  # Good pattern count
            score += 0.3
        else:
            score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_construction_quality(self, design: FashionDesign) -> float:
        """Evaluate construction feasibility and quality"""
        
        score = 0.0
        
        # Check pattern pieces
        for pattern in design.pattern_pieces:
            if 'vertices' in pattern and len(pattern['vertices']) >= 3:
                # Check for reasonable proportions
                vertices = np.array(pattern['vertices'])
                area = self._calculate_polygon_area(vertices)
                perimeter = self._calculate_polygon_perimeter(vertices)
                
                if 1000 < area < 500000 and 200 < perimeter < 3000:
                    score += 0.3
        
        # Check construction parameters
        params = design.construction_params
        if hasattr(params, 'ease_amounts'):
            reasonable_ease = all(10 <= ease <= 150 for ease in params.ease_amounts.values())
            if reasonable_ease:
                score += 0.4
        
        if hasattr(params, 'seam_allowances'):
            reasonable_seams = all(5 <= sa <= 25 for sa in params.seam_allowances.values())
            if reasonable_seams:
                score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate polygon area using shoelace formula"""
        if len(vertices) < 3:
            return 0.0
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] 
                            for i in range(-1, len(x)-1)))
    
    def _calculate_polygon_perimeter(self, vertices: np.ndarray) -> float:
        """Calculate polygon perimeter"""
        if len(vertices) < 2:
            return 0.0
        perimeter = 0.0
        for i in range(len(vertices)):
            next_i = (i + 1) % len(vertices)
            perimeter += np.linalg.norm(vertices[next_i] - vertices[i])
        return perimeter
    
    def run_beautiful_evolution(self, generations: int = 8, population_size: int = 12):
        """Run enhanced evolution focused on beautiful results"""
        
        # Create timestamped results folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = f"beautiful_fashion_evolution_{timestamp}"
        os.makedirs(results_folder, exist_ok=True)
        
        print(f"✨ Starting Beautiful Fashion Evolution")
        print(f"🎨 Generations: {generations}")
        print(f"👗 Population: {population_size}")
        print(f"📁 Results folder: {results_folder}")
        print(f"🎯 Focus: {self.garment_type.value}")
        
        # Initialize population
        population = self.evolution_engine.create_initial_population(
            population_size, self.garment_type.value
        )
        
        all_generations_data = []
        
        for generation in range(generations):
            print(f"\n🌟 Generation {generation + 1}/{generations}")
            print("=" * 50)
            
            # Create beautiful designs from genomes
            designs = []
            for i, genome in enumerate(population):
                print(f"   Creating beautiful design {i+1}/{len(population)}...")
                try:
                    design = self.create_enhanced_design_from_genome(genome)
                    design.generation = generation
                    designs.append(design)
                except Exception as e:
                    print(f"   ⚠️  Design creation failed: {e}")
                    # Create minimal fallback
                    fallback_design = FashionDesign(
                        genome=genome,
                        construction_params=ConstructionParameters.create_default(self.garment_type.value),
                        pattern_pieces=[],
                        fitness=random.uniform(0.1, 0.3),
                        generation=generation
                    )
                    designs.append(fallback_design)
            
            # Evaluate beauty and wearability
            print("   Evaluating design beauty...")
            for design in designs:
                beauty_score = self.evaluate_design_beauty(design)
                design.fitness = beauty_score
                design.genome.fitness = beauty_score
            
            # Display beautiful results
            self.display_beautiful_generation(designs, generation, results_folder)
            
            # Save generation data
            generation_data = [design.to_dict() for design in designs]
            all_generations_data.append(generation_data)
            
            # Print generation statistics
            fitnesses = [d.fitness for d in designs]
            print(f"   🏆 Best fitness: {max(fitnesses):.3f}")
            print(f"   📊 Average fitness: {np.mean(fitnesses):.3f}")
            print(f"   🎨 Best theme: {max(designs, key=lambda d: d.fitness).style_characteristics.get('theme', 'unknown')}")
            
            # Evolve to next generation
            if generation < generations - 1:
                population = self.evolution_engine.evolve_population(
                    [d.genome for d in designs], population_size
                )
        
        # Create final showcase
        print(f"\n🎉 Evolution Complete! Creating final showcase...")
        self.create_final_fashion_showcase(all_generations_data, results_folder)
        
        return designs
    
    def display_beautiful_generation(self, designs: List[FashionDesign], 
                                   generation: int, results_folder: str):
        """Display beautiful generation with enhanced visualization"""
        
        # Use enhanced renderer to display designs
        design_dicts = []
        for design in designs:
            design_dict = {
                'fitness': design.fitness,
                'generation': design.generation,
                'construction_params': design.construction_params,
                'style_characteristics': design.style_characteristics,
                'color_palette': design.color_palette,
                'design_notes': design.design_notes
            }
            design_dicts.append(design_dict)
        
        # Create beautiful visualization
        save_path = os.path.join(results_folder, f'generation_{generation + 1}_beautiful_dresses.png')
        
        try:
            self.renderer.render_design_comparison(design_dicts, self.body_model, save_path)
            print(f"   🖼️  Beautiful visualization saved to {save_path}")
        except Exception as e:
            print(f"   ⚠️  Visualization failed: {e}")
    
    def create_final_fashion_showcase(self, all_generations_data: List[List[Dict]], 
                                    results_folder: str):
        """Create stunning final fashion showcase"""
        
        showcase_path = os.path.join(results_folder, 'final_fashion_showcase.png')
        
        try:
            self.renderer.create_evolution_showcase(all_generations_data, self.body_model, showcase_path)
            print(f"   🎨 Final showcase created: {showcase_path}")
        except Exception as e:
            print(f"   ⚠️  Showcase creation failed: {e}")
        
        # Save detailed results
        final_data = {
            'evolution_summary': {
                'total_generations': len(all_generations_data),
                'garment_type': self.garment_type.value,
                'timestamp': datetime.datetime.now().isoformat()
            },
            'all_generations': all_generations_data
        }
        
        data_path = os.path.join(results_folder, 'complete_evolution_data.json')
        with open(data_path, 'w') as f:
            json.dump(final_data, f, indent=2, default=str)
        
        print(f"   💾 Complete data saved: {data_path}")

def main():
    """Main execution function for beautiful fashion evolution"""
    
    print("✨ Enhanced 3D Fashion CPPN Evolution System")
    print("=" * 60)
    print("🎨 Creating Beautiful Fashion Designs")
    print("=" * 60)
    
    # Choose garment type (default to dress for best visual results)
    garment_types = list(GarmentType)
    print("\nAvailable garment types:")
    for i, gt in enumerate(garment_types):
        print(f"  {i+1}. {gt.value}")
    
    try:
        choice = input("Choose garment type (1-5, default=2 for dress): ").strip()
        choice = int(choice) if choice else 2
        garment_type = garment_types[choice - 1]
    except (ValueError, IndexError):
        garment_type = GarmentType.DRESS
        print("Using default: dress")
    
    # Initialize enhanced system
    system = Enhanced3DFashionSystem(garment_type)
    
    # Get evolution parameters
    try:
        generations = int(input("Number of generations (default=6): ") or "6")
        population_size = int(input("Population size (default=9): ") or "9")
    except ValueError:
        generations = 6
        population_size = 9
    
    print(f"\n🚀 Starting evolution with {generations} generations, {population_size} designs per generation")
    
    # Run beautiful evolution
    final_designs = system.run_beautiful_evolution(generations, population_size)
    
    # Display final results
    best_design = max(final_designs, key=lambda d: d.fitness)
    print(f"\n🏆 EVOLUTION COMPLETE!")
    print(f"🎨 Best Design Fitness: {best_design.fitness:.3f}")
    print(f"👗 Best Design Theme: {best_design.style_characteristics.get('theme', 'unknown')}")
    print(f"📝 Design Notes: {'; '.join(best_design.design_notes)}")
    print(f"✨ Created {len(final_designs)} beautiful fashion designs!")

if __name__ == "__main__":
    main()