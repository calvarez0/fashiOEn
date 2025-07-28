# pattern_generator.py - Generates 2D patterns from construction parameters
"""
Generates actual 2D pattern pieces that can be cut from fabric and sewn together.
Uses professional pattern making techniques adapted for CPPN-driven design.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass
from body_model import HumanBodyModel
from garment_genome import ConstructionParameters

@dataclass
class PatternPiece:
    """Represents a single pattern piece"""
    name: str
    vertices: List[Tuple[float, float]]  # 2D coordinates in mm
    seam_allowances: Dict[str, float]    # Seam allowance for each edge
    grain_line: Tuple[Tuple[float, float], Tuple[float, float]]  # Grain line start/end
    notches: List[Tuple[float, float]]   # Notch positions for matching
    darts: List[Dict]                    # Dart information
    construction_notes: List[str]        # Notes for sewing
    
    def get_area(self) -> float:
        """Calculate pattern piece area using shoelace formula"""
        if len(self.vertices) < 3:
            return 0.0
        
        x = [v[0] for v in self.vertices]
        y = [v[1] for v in self.vertices]
        
        return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] 
                            for i in range(-1, len(x)-1)))
    
    def get_perimeter(self) -> float:
        """Calculate pattern piece perimeter"""
        if len(self.vertices) < 2:
            return 0.0
        
        perimeter = 0.0
        for i in range(len(self.vertices)):
            next_i = (i + 1) % len(self.vertices)
            dx = self.vertices[next_i][0] - self.vertices[i][0]
            dy = self.vertices[next_i][1] - self.vertices[i][1]
            perimeter += math.sqrt(dx*dx + dy*dy)
        
        return perimeter

class PatternGenerator:
    """Generates 2D patterns from 3D construction parameters"""
    
    def __init__(self, body_model: HumanBodyModel):
        self.body_model = body_model
        self.measurements = body_model.get_measurements()
        self.key_points = body_model.get_key_points()
    
    def generate_patterns(self, construction_params: ConstructionParameters, 
                         garment_type: str) -> List[Dict]:
        """Generate all pattern pieces for a garment"""
        
        if garment_type == "jacket":
            return self._generate_jacket_patterns(construction_params)
        elif garment_type == "dress":
            return self._generate_dress_patterns(construction_params)
        elif garment_type == "shirt":
            return self._generate_shirt_patterns(construction_params)
        else:
            return self._generate_basic_patterns(construction_params)
    
    def _generate_jacket_patterns(self, params: ConstructionParameters) -> List[Dict]:
        """Generate pattern pieces for a jacket"""
        patterns = []
        
        # Front pattern piece
        front_pattern = self._create_jacket_front(params)
        patterns.append(front_pattern.to_dict() if hasattr(front_pattern, 'to_dict') else 
                       self._pattern_piece_to_dict(front_pattern))
        
        # Back pattern piece
        back_pattern = self._create_jacket_back(params)
        patterns.append(self._pattern_piece_to_dict(back_pattern))
        
        # Sleeve pattern piece
        sleeve_pattern = self._create_jacket_sleeve(params)
        patterns.append(self._pattern_piece_to_dict(sleeve_pattern))
        
        # Lapel/collar pieces
        lapel_pattern = self._create_jacket_lapel(params)
        patterns.append(self._pattern_piece_to_dict(lapel_pattern))
        
        return patterns
    
    def _create_jacket_front(self, params: ConstructionParameters) -> PatternPiece:
        """Create jacket front pattern piece"""
        m = self.measurements
        
        # Base measurements
        chest_width = (m['chest_width'] + params.ease_amounts.get('chest', 100)) / 2
        waist_width = (m['waist_width'] + params.ease_amounts.get('waist', 80)) / 2
        hip_width = (m['hip_width'] + params.ease_amounts.get('hip', 90)) / 2
        
        # Heights
        total_length = m['shoulder_height'] - m['hem_height']
        chest_level = m['shoulder_height'] - m['chest_height']
        waist_level = m['shoulder_height'] - m['waist_height']
        hip_level = m['shoulder_height'] - m['hip_height']
        
        # Apply construction parameters
        waist_suppression = params.waist_suppression
        shoulder_slope = params.shoulder_slope
        lapel_width = params.lapel_width
        
        # Calculate key points
        vertices = []
        
        # Start at center front hem
        vertices.append((0, total_length))
        
        # Center front up to lapel break
        lapel_break_height = total_length * 0.3  # 30% up from hem
        vertices.append((0, lapel_break_height))
        
        # Lapel line
        vertices.append((lapel_width, lapel_break_height - 50))
        vertices.append((lapel_width, chest_level))
        
        # Shoulder point
        shoulder_x = m['shoulder_width'] / 2 + params.width_adjustments.get('shoulder', 0)
        shoulder_y = shoulder_slope * 10  # Convert slope to mm
        vertices.append((shoulder_x, shoulder_y))
        
        # Armhole curve - simplified as line segments
        armhole_depth = params.ease_amounts.get('armhole', 25)
        armhole_bottom = chest_level + armhole_depth
        
        vertices.append((shoulder_x, armhole_bottom))
        vertices.append((chest_width, armhole_bottom))
        
        # Side seam with waist suppression
        side_chest_x = chest_width
        side_waist_x = waist_width * (1 + waist_suppression)
        side_hip_x = hip_width
        
        # Apply seam curves if available
        if 'side_seam' in params.seam_curves:
            curve_points = params.seam_curves['side_seam']
            for t, curve_offset in curve_points:
                height = chest_level + t * (hip_level - chest_level)
                base_width = side_chest_x + t * (side_hip_x - side_chest_x)
                width = base_width + curve_offset
                vertices.append((width, height))
        else:
            # Standard side seam
            vertices.append((side_chest_x, chest_level))
            vertices.append((side_waist_x, waist_level))
            vertices.append((side_hip_x, hip_level))
        
        # Hip to hem
        vertices.append((side_hip_x, total_length))
        
        # Close the pattern
        vertices.append((0, total_length))
        
        # Create darts
        darts = []
        if 'front_waist' in params.dart_positions:
            dart_pos = params.dart_positions['front_waist']
            dart_intake = params.dart_intakes.get('front_waist', 20)
            
            dart_x = waist_width * dart_pos[0]
            dart_y = waist_level + dart_pos[1]
            
            darts.append({
                'type': 'waist_dart',
                'apex': (dart_x, dart_y - 50),  # 50mm above waist
                'legs': [
                    (dart_x - dart_intake/2, dart_y),
                    (dart_x + dart_intake/2, dart_y)
                ],
                'intake': dart_intake
            })
        
        # Seam allowances
        seam_allowances = {
            'center_front': 15,
            'shoulder': 12,
            'side_seam': 15,
            'armhole': 10,
            'hem': 40
        }
        seam_allowances.update(params.seam_allowances)
        
        # Grain line (parallel to center front)
        grain_line = ((50, 50), (50, total_length - 50))
        
        # Notches for matching
        notches = [
            (shoulder_x, shoulder_y),  # Shoulder notch
            (side_waist_x, waist_level),  # Side seam waist notch
        ]
        
        return PatternPiece(
            name="Jacket Front",
            vertices=vertices,
            seam_allowances=seam_allowances,
            grain_line=grain_line,
            notches=notches,
            darts=darts,
            construction_notes=[
                "Cut 2 pieces (1 left, 1 right)",
                "Interface lapel area",
                "Mark button positions",
                f"Lapel width: {lapel_width:.1f}mm"
            ]
        )
    
    def _create_jacket_back(self, params: ConstructionParameters) -> PatternPiece:
        """Create jacket back pattern piece"""
        m = self.measurements
        
        # Base measurements with ease
        chest_width = (m['chest_width'] + params.ease_amounts.get('chest', 100)) / 2
        waist_width = (m['waist_width'] + params.ease_amounts.get('waist', 80)) / 2
        hip_width = (m['hip_width'] + params.ease_amounts.get('hip', 90)) / 2
        
        # Heights
        total_length = m['shoulder_height'] - m['hem_height']
        chest_level = m['shoulder_height'] - m['chest_height']
        waist_level = m['shoulder_height'] - m['waist_height']
        hip_level = m['shoulder_height'] - m['hip_height']
        
        # Back is typically wider than front
        back_ease = 1.1
        chest_width *= back_ease
        waist_width *= back_ease
        hip_width *= back_ease
        
        vertices = []
        
        # Center back seam
        vertices.append((0, total_length))  # Hem
        vertices.append((0, 0))  # Neck
        
        # Shoulder line with slope
        shoulder_slope = params.shoulder_slope
        shoulder_x = m['shoulder_width'] / 2
        shoulder_y = -shoulder_slope * 10  # Opposite slope from front
        vertices.append((shoulder_x, shoulder_y))
        
        # Armhole
        armhole_depth = params.ease_amounts.get('armhole', 25)
        armhole_bottom = chest_level + armhole_depth
        vertices.append((shoulder_x, armhole_bottom))
        vertices.append((chest_width, armhole_bottom))
        
        # Side seam
        vertices.append((chest_width, chest_level))
        
        # Waist with suppression
        waist_suppression = params.waist_suppression
        side_waist_x = waist_width * (1 + waist_suppression * 0.7)  # Less suppression in back
        vertices.append((side_waist_x, waist_level))
        
        # Hip and hem
        vertices.append((hip_width, hip_level))
        vertices.append((hip_width, total_length))
        vertices.append((0, total_length))
        
        # Back darts (typically larger than front)
        darts = []
        if 'back_waist' in params.dart_positions:
            dart_pos = params.dart_positions['back_waist']
            dart_intake = params.dart_intakes.get('back_waist', 30)
            
            dart_x = waist_width * dart_pos[0]
            dart_y = waist_level + dart_pos[1]
            
            darts.append({
                'type': 'back_waist_dart',
                'apex': (dart_x, dart_y - 80),  # Higher apex for back
                'legs': [
                    (dart_x - dart_intake/2, dart_y),
                    (dart_x + dart_intake/2, dart_y)
                ],
                'intake': dart_intake
            })
        
        # Shoulder dart if needed
        if 'shoulder' in params.dart_positions and params.dart_intakes.get('shoulder', 0) > 5:
            shoulder_dart_intake = params.dart_intakes['shoulder']
            shoulder_dart_x = shoulder_x * 0.6
            
            darts.append({
                'type': 'shoulder_dart',
                'apex': (shoulder_dart_x, shoulder_y - 30),
                'legs': [
                    (shoulder_dart_x - shoulder_dart_intake/2, shoulder_y),
                    (shoulder_dart_x + shoulder_dart_intake/2, shoulder_y)
                ],
                'intake': shoulder_dart_intake
            })
        
        seam_allowances = {
            'center_back': 15,
            'shoulder': 12,
            'side_seam': 15,
            'armhole': 10,
            'hem': 40
        }
        seam_allowances.update(params.seam_allowances)
        
        grain_line = ((50, 50), (50, total_length - 50))
        
        notches = [
            (shoulder_x, shoulder_y),
            (side_waist_x, waist_level),
        ]
        
        return PatternPiece(
            name="Jacket Back",
            vertices=vertices,
            seam_allowances=seam_allowances,
            grain_line=grain_line,
            notches=notches,
            darts=darts,
            construction_notes=[
                "Cut 2 pieces (1 left, 1 right)",
                "May be cut on fold if symmetrical",
                "Back typically has larger darts"
            ]
        )
    
    def _create_jacket_sleeve(self, params: ConstructionParameters) -> PatternPiece:
        """Create jacket sleeve pattern piece"""
        m = self.measurements
        
        # Sleeve measurements
        sleeve_length = m['sleeve_length']
        armhole_circ = (m['shoulder_width'] + params.ease_amounts.get('armhole', 25)) * math.pi
        bicep_width = armhole_circ / 2  # Half circumference for flat pattern
        wrist_width = 120  # Standard wrist width
        
        # Sleeve cap height (critical for fit)
        cap_height = 120 + params.ease_amounts.get('sleeve_cap', 0)
        
        vertices = []
        
        # Start at wrist
        vertices.append((0, sleeve_length))
        vertices.append((wrist_width, sleeve_length))
        
        # Sleeve side seams
        vertices.append((bicep_width * 1.1, cap_height))  # Under arm
        
        # Sleeve cap curve (simplified as polygon)
        cap_points = 8
        for i in range(cap_points + 1):
            t = i / cap_points
            angle = math.pi * t
            
            x = bicep_width * (0.5 + 0.5 * math.cos(angle))
            y = cap_height * (1 - math.sin(angle))
            
            # Add curve variation from parameters
            if 'armhole' in params.seam_curves:
                curve_factor = 0.1  # Influence of curve parameters
                y += curve_factor * cap_height * math.sin(angle * 2)
            
            vertices.append((x, y))
        
        vertices.append((0, cap_height))
        vertices.append((0, sleeve_length))
        
        # Elbow dart for fitted sleeves
        darts = []
        elbow_level = sleeve_length * 0.6
        if params.ease_amounts.get('sleeve', 0) < 50:  # Fitted sleeve
            darts.append({
                'type': 'elbow_dart',
                'apex': (bicep_width * 0.8, elbow_level - 40),
                'legs': [
                    (bicep_width * 0.75, elbow_level),
                    (bicep_width * 0.85, elbow_level)
                ],
                'intake': 15
            })
        
        seam_allowances = {
            'sleeve_seam': 15,
            'armhole': 10,
            'hem': 25
        }
        
        grain_line = ((bicep_width/2, sleeve_length - 50), (bicep_width/2, 50))
        
        notches = [
            (bicep_width/2, 0),  # Top of cap
            (0, elbow_level),    # Elbow level
        ]
        
        return PatternPiece(
            name="Jacket Sleeve",
            vertices=vertices,
            seam_allowances=seam_allowances,
            grain_line=grain_line,
            notches=notches,
            darts=darts,
            construction_notes=[
                "Cut 2 pieces (1 left, 1 right)",
                f"Sleeve cap height: {cap_height:.1f}mm",
                "Ease sleeve cap into armhole",
                "Match notches to body armhole"
            ]
        )
    
    def _create_jacket_lapel(self, params: ConstructionParameters) -> PatternPiece:
        """Create jacket lapel/collar pattern piece"""
        
        lapel_width = params.lapel_width
        lapel_roll = params.lapel_roll_line
        neck_circumference = self.measurements['neck_width'] * 2
        
        # Simplified lapel shape
        vertices = [
            (0, 0),  # Collar point
            (lapel_width, 0),
            (lapel_width + 20, 30),  # Lapel point
            (lapel_width, 80),
            (lapel_roll, 100),
            (0, 80),
            (0, 0)
        ]
        
        seam_allowances = {
            'collar_seam': 10,
            'lapel_edge': 6,
            'center_back': 12
        }
        
        grain_line = ((lapel_width/2, 10), (lapel_width/2, 90))
        
        return PatternPiece(
            name="Jacket Lapel",
            vertices=vertices,
            seam_allowances=seam_allowances,
            grain_line=grain_line,
            notches=[],
            darts=[],
            construction_notes=[
                "Cut 2 pieces",
                "Interface entire piece",
                f"Roll line at {lapel_roll:.1f}mm"
            ]
        )
    
    def _generate_dress_patterns(self, params: ConstructionParameters) -> List[Dict]:
        """Generate pattern pieces for a dress"""
        patterns = []
        
        # Bodice front
        bodice_front = self._create_dress_bodice_front(params)
        patterns.append(self._pattern_piece_to_dict(bodice_front))
        
        # Bodice back
        bodice_back = self._create_dress_bodice_back(params)
        patterns.append(self._pattern_piece_to_dict(bodice_back))
        
        # Skirt front
        skirt_front = self._create_dress_skirt_front(params)
        patterns.append(self._pattern_piece_to_dict(skirt_front))
        
        # Skirt back
        skirt_back = self._create_dress_skirt_back(params)
        patterns.append(self._pattern_piece_to_dict(skirt_back))
        
        return patterns
    
    def _create_dress_bodice_front(self, params: ConstructionParameters) -> PatternPiece:
        """Create dress bodice front"""
        m = self.measurements
        
        bust_width = (m['bust_width'] + params.ease_amounts.get('bust', 80)) / 2
        waist_width = (m['waist_width'] + params.ease_amounts.get('waist', 60)) / 2
        
        bodice_length = m['shoulder_height'] - m['waist_height']
        bust_level = m['shoulder_height'] - m['bust_height']
        
        vertices = [
            (0, bodice_length),  # Center front waist
            (0, 0),              # Center front neck
            (m['neck_width']/2, 0),  # Side neck
            (m['shoulder_width']/2, 10),  # Shoulder
            (m['shoulder_width']/2, bust_level + 30),  # Armhole
            (bust_width, bust_level + 30),
            (bust_width, bust_level),
            (waist_width, bodice_length),
            (0, bodice_length)
        ]
        
        # Bust dart
        darts = []
        if 'bust' in params.dart_positions:
            bust_dart_pos = params.dart_positions['bust']
            bust_dart_intake = params.dart_intakes.get('bust', 25)
            
            dart_x = bust_width * bust_dart_pos[0]
            dart_y = bust_level + bust_dart_pos[1]
            
            darts.append({
                'type': 'bust_dart',
                'apex': (dart_x, dart_y),
                'legs': [
                    (dart_x - bust_dart_intake/2, bodice_length),
                    (dart_x + bust_dart_intake/2, bodice_length)
                ],
                'intake': bust_dart_intake
            })
        
        seam_allowances = {
            'center_front': 15,
            'shoulder': 12,
            'side_seam': 15,
            'armhole': 10,
            'waist': 15
        }
        
        return PatternPiece(
            name="Dress Bodice Front",
            vertices=vertices,
            seam_allowances=seam_allowances,
            grain_line=((50, 50), (50, bodice_length - 50)),
            notches=[(waist_width, bodice_length)],
            darts=darts,
            construction_notes=["Cut 2 pieces", "Bust dart critical for fit"]
        )
    
    def _create_dress_bodice_back(self, params: ConstructionParameters) -> PatternPiece:
        """Create dress bodice back"""
        m = self.measurements
        
        back_width = (m['chest_width'] + params.ease_amounts.get('bust', 80)) / 2 * 1.05
        waist_width = (m['waist_width'] + params.ease_amounts.get('waist', 60)) / 2 * 1.05
        
        bodice_length = m['shoulder_height'] - m['waist_height']
        chest_level = m['shoulder_height'] - m['chest_height']
        
        vertices = [
            (0, bodice_length),  # Center back waist
            (0, 0),              # Center back neck
            (m['neck_width']/2, 0),
            (m['shoulder_width']/2, 10),
            (m['shoulder_width']/2, chest_level + 30),
            (back_width, chest_level + 30),
            (back_width, chest_level),
            (waist_width, bodice_length),
            (0, bodice_length)
        ]
        
        # Back darts (usually waist and shoulder)
        darts = []
        if 'waist' in params.dart_positions:
            waist_dart_intake = params.dart_intakes.get('waist', 35)
            dart_x = waist_width * 0.5
            
            darts.append({
                'type': 'back_waist_dart',
                'apex': (dart_x, bodice_length - 100),
                'legs': [
                    (dart_x - waist_dart_intake/2, bodice_length),
                    (dart_x + waist_dart_intake/2, bodice_length)
                ],
                'intake': waist_dart_intake
            })
        
        seam_allowances = {
            'center_back': 15,
            'shoulder': 12,
            'side_seam': 15,
            'armhole': 10,
            'waist': 15
        }
        
        return PatternPiece(
            name="Dress Bodice Back",
            vertices=vertices,
            seam_allowances=seam_allowances,
            grain_line=((50, 50), (50, bodice_length - 50)),
            notches=[(waist_width, bodice_length)],
            darts=darts,
            construction_notes=["Cut 2 pieces", "Center back zipper opening"]
        )
    
    def _create_dress_skirt_front(self, params: ConstructionParameters) -> PatternPiece:
        """Create dress skirt front"""
        m = self.measurements
        
        waist_width = (m['waist_width'] + params.ease_amounts.get('waist', 60)) / 2
        hip_width = (m['hip_width'] + params.ease_amounts.get('hip', 100)) / 2
        hem_width = hip_width * 1.2  # A-line flare
        
        # Apply hem curve from parameters
        hem_curve = params.hem_curve
        
        skirt_length = m['waist_height'] - m['hem_height']
        hip_level = m['waist_height'] - m['hip_height']
        
        vertices = [
            (0, 0),  # Center front waist
            (waist_width, 0),
            (hip_width, hip_level),
            (hem_width, skirt_length + hem_curve),
            (0, skirt_length),
            (0, 0)
        ]
        
        seam_allowances = {
            'side_seam': 15,
            'waist': 15,
            'hem': 30
        }
        
        return PatternPiece(
            name="Dress Skirt Front",
            vertices=vertices,
            seam_allowances=seam_allowances,
            grain_line=((50, 50), (50, skirt_length - 50)),
            notches=[(hip_width, hip_level)],
            darts=[],
            construction_notes=["Cut 2 pieces", f"Hem curve: {hem_curve:.1f}mm"]
        )
    
    def _create_dress_skirt_back(self, params: ConstructionParameters) -> PatternPiece:
        """Create dress skirt back"""
        # Similar to front but potentially different curve
        skirt_front = self._create_dress_skirt_front(params)
        skirt_front.name = "Dress Skirt Back"
        skirt_front.construction_notes = ["Cut 2 pieces", "Center back seam"]
        return skirt_front
    
    def _generate_basic_patterns(self, params: ConstructionParameters) -> List[Dict]:
        """Generate basic pattern pieces for unknown garment types"""
        patterns = []
        
        # Create a simple front and back piece
        front = self._create_basic_front(params)
        patterns.append(self._pattern_piece_to_dict(front))
        
        back = self._create_basic_back(params)
        patterns.append(self._pattern_piece_to_dict(back))
        
        return patterns
    
    def _create_basic_front(self, params: ConstructionParameters) -> PatternPiece:
        """Create basic front pattern piece"""
        m = self.measurements
        
        width = m['chest_width'] / 2
        length = m['shoulder_height'] - m['hip_height']
        
        vertices = [
            (0, length),
            (0, 0),
            (width, 0),
            (width, length),
            (0, length)
        ]
        
        return PatternPiece(
            name="Basic Front",
            vertices=vertices,
            seam_allowances={'all': 15},
            grain_line=((width/2, 50), (width/2, length - 50)),
            notches=[],
            darts=[],
            construction_notes=["Basic rectangular pattern"]
        )
    
    def _create_basic_back(self, params: ConstructionParameters) -> PatternPiece:
        """Create basic back pattern piece"""
        front = self._create_basic_front(params)
        front.name = "Basic Back"
        return front
    
    def _pattern_piece_to_dict(self, pattern: PatternPiece) -> Dict:
        """Convert PatternPiece to dictionary for serialization"""
        return {
            'name': pattern.name,
            'vertices': pattern.vertices,
            'seam_allowances': pattern.seam_allowances,
            'grain_line': pattern.grain_line,
            'notches': pattern.notches,
            'darts': pattern.darts,
            'construction_notes': pattern.construction_notes,
            'area': pattern.get_area(),
            'perimeter': pattern.get_perimeter()
        }
    
    def visualize_patterns(self, patterns: List[Dict], save_path: Optional[str] = None):
        """Visualize all pattern pieces"""
        n_patterns = len(patterns)
        cols = min(3, n_patterns)
        rows = (n_patterns + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, pattern in enumerate(patterns):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue
            
            vertices = pattern['vertices']
            if len(vertices) > 2:
                # Close the polygon
                vertices_closed = vertices + [vertices[0]]
                x_coords = [v[0] for v in vertices_closed]
                y_coords = [v[1] for v in vertices_closed]
                
                # Plot pattern outline
                ax.plot(x_coords, y_coords, 'b-', linewidth=2, label='Pattern edge')
                ax.fill(x_coords, y_coords, alpha=0.3, color='lightblue')
                
                # Plot grain line
                if 'grain_line' in pattern:
                    grain = pattern['grain_line']
                    ax.plot([grain[0][0], grain[1][0]], 
                           [grain[0][1], grain[1][1]], 
                           'g--', linewidth=2, label='Grain line')
                
                # Plot notches
                if 'notches' in pattern:
                    for notch in pattern['notches']:
                        ax.plot(notch[0], notch[1], 'ro', markersize=8, label='Notch')
                
                # Plot darts
                if 'darts' in pattern:
                    for dart in pattern['darts']:
                        if 'apex' in dart and 'legs' in dart:
                            apex = dart['apex']
                            for leg in dart['legs']:
                                ax.plot([apex[0], leg[0]], [apex[1], leg[1]], 
                                       'r-', linewidth=1.5, alpha=0.7)
                            ax.plot(apex[0], apex[1], 'r^', markersize=6)
                
                # Annotations
                ax.set_title(f"{pattern['name']}\nArea: {pattern.get('area', 0):.0f}mm²")
                ax.set_xlabel('Width (mm)')
                ax.set_ylabel('Length (mm)')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                
                # Add construction notes
                if 'construction_notes' in pattern:
                    notes_text = '\n'.join(pattern['construction_notes'][:3])  # First 3 notes
                    ax.text(0.02, 0.98, notes_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=8, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Hide unused subplots
        for i in range(n_patterns, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Generated Pattern Pieces', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_patterns_dxf(self, patterns: List[Dict], filename: str):
        """Export patterns to DXF format for CAD software"""
        try:
            import ezdxf
        except ImportError:
            print("ezdxf package required for DXF export")
            return
        
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        for i, pattern in enumerate(patterns):
            # Create layer for this pattern
            layer_name = f"PATTERN_{i}_{pattern['name'].replace(' ', '_')}"
            doc.layers.new(name=layer_name, dxfattribs={'color': i + 1})
            
            # Draw pattern outline
            vertices = pattern['vertices']
            if len(vertices) > 2:
                points = [(v[0], v[1]) for v in vertices + [vertices[0]]]  # Close polygon
                msp.add_lwpolyline(points, dxfattribs={'layer': layer_name})
            
            # Add grain line
            if 'grain_line' in pattern:
                grain = pattern['grain_line']
                msp.add_line(grain[0], grain[1], 
                           dxfattribs={'layer': layer_name, 'linetype': 'DASHED'})
            
            # Add notches as circles
            if 'notches' in pattern:
                for notch in pattern['notches']:
                    msp.add_circle((notch[0], notch[1]), 2, 
                                 dxfattribs={'layer': layer_name})
            
            # Add text annotation
            if vertices:
                center_x = sum(v[0] for v in vertices) / len(vertices)
                center_y = sum(v[1] for v in vertices) / len(vertices)
                msp.add_text(pattern['name'], 
                           dxfattribs={'layer': layer_name, 'height': 10}).set_pos((center_x, center_y))
        
        doc.saveas(filename)
        print(f"Patterns exported to {filename}")
    
    def calculate_fabric_usage(self, patterns: List[Dict], fabric_width: float = 1500) -> Dict:
        """Calculate fabric usage and layout efficiency"""
        total_area = sum(pattern.get('area', 0) for pattern in patterns)
        
        # Simple rectangular bounding box calculation
        total_width = 0
        max_height = 0
        
        for pattern in patterns:
            vertices = pattern['vertices']
            if vertices:
                pattern_width = max(v[0] for v in vertices) - min(v[0] for v in vertices)
                pattern_height = max(v[1] for v in vertices) - min(v[1] for v in vertices)
                
                total_width += pattern_width + 50  # 50mm spacing
                max_height = max(max_height, pattern_height)
        
        # Calculate fabric usage
        if total_width <= fabric_width:
            fabric_length = max_height + 100  # 100mm margin
            layout_efficiency = total_area / (fabric_width * fabric_length)
        else:
            # Multiple rows needed
            rows = math.ceil(total_width / fabric_width)
            fabric_length = (max_height + 100) * rows
            layout_efficiency = total_area / (fabric_width * fabric_length)
        
        return {
            'total_pattern_area': total_area,
            'estimated_fabric_length': fabric_length,
            'fabric_width': fabric_width,
            'layout_efficiency': layout_efficiency,
            'fabric_waste_percentage': (1 - layout_efficiency) * 100,
            'estimated_rows': rows if 'rows' in locals() else 1
        }

# Example usage
if __name__ == "__main__":
    from body_model import HumanBodyModel
    from garment_genome import ConstructionParameters
    
    # Create body model
    body = HumanBodyModel(size='M')
    
    # Create pattern generator
    generator = PatternGenerator(body)
    
    # Create sample construction parameters
    params = ConstructionParameters.create_default("jacket")
    
    # Generate patterns
    patterns = generator.generate_patterns(params, "jacket")
    
    # Visualize patterns
    generator.visualize_patterns(patterns, "sample_jacket_patterns.png")
    
    # Calculate fabric usage
    fabric_usage = generator.calculate_fabric_usage(patterns)
    print(f"\nFabric Usage Analysis:")
    for key, value in fabric_usage.items():
        print(f"  {key}: {value}")
    
    # Export to DXF
    try:
        generator.export_patterns_dxf(patterns, "jacket_patterns.dxf")
    except Exception as e:
        print(f"DXF export failed: {e}")
    
    print(f"\nGenerated {len(patterns)} pattern pieces for jacket")