# garment_genome.py - CPPN genome for garment construction
"""
CPPN implementation specifically designed for garment construction parameters.
Outputs real tailoring measurements that can be used to create actual patterns.
"""

import numpy as np
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

class FashionActivationFunction(Enum):
    """Activation functions optimized for fashion construction"""
    SINE = "sine"                    # Natural curves, seam lines
    COSINE = "cosine"               # Complementary curves
    GAUSSIAN = "gaussian"           # Soft transitions, dart shapes
    SIGMOID = "sigmoid"             # Smooth ease transitions
    TANH = "tanh"                  # Symmetric shaping
    LINEAR = "linear"               # Straight seams, hems
    EASE_CURVE = "ease_curve"       # Body-fitting ease distribution
    DART_SHAPE = "dart_shape"       # Dart intake curves
    SEAM_CURVE = "seam_curve"       # Natural seam curvature
    GOLDEN_RATIO = "golden_ratio"   # Proportional harmony
    BODY_CURVE = "body_curve"       # Body-following curves
    FABRIC_TENSION = "fabric_tension" # Fabric behavior simulation

@dataclass
class GarmentGenome:
    """Genome representing the DNA of a garment's construction"""
    nodes: List[Dict] = field(default_factory=list)
    connections: List[Dict] = field(default_factory=list)
    garment_type: str = "jacket"
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'nodes': self.nodes,
            'connections': self.connections,
            'garment_type': self.garment_type,
            'fitness': self.fitness,
            'generation': self.generation,
            'parent_ids': self.parent_ids
        }

@dataclass
class ConstructionParameters:
    """Real garment construction parameters that can be used for pattern making"""
    
    # Ease amounts (how loose the garment is)
    ease_amounts: Dict[str, float] = field(default_factory=dict)
    
    # Dart placements and sizes
    dart_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    dart_intakes: Dict[str, float] = field(default_factory=dict)
    
    # Seam curves and allowances
    seam_curves: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    seam_allowances: Dict[str, float] = field(default_factory=dict)
    
    # Proportional adjustments
    length_adjustments: Dict[str, float] = field(default_factory=dict)
    width_adjustments: Dict[str, float] = field(default_factory=dict)
    
    # Style details
    lapel_width: float = 75.0  # mm
    lapel_roll_line: float = 25.0  # mm from edge
    button_stance: float = 0.0  # adjustment from standard
    hem_curve: float = 0.0  # curve amount
    
    # Advanced parameters
    asymmetry_factor: float = 0.0
    waist_suppression: float = 0.0
    shoulder_slope: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'ease_amounts': self.ease_amounts,
            'dart_positions': self.dart_positions,
            'dart_intakes': self.dart_intakes,
            'seam_curves': self.seam_curves,
            'seam_allowances': self.seam_allowances,
            'length_adjustments': self.length_adjustments,
            'width_adjustments': self.width_adjustments,
            'lapel_width': self.lapel_width,
            'lapel_roll_line': self.lapel_roll_line,
            'button_stance': self.button_stance,
            'hem_curve': self.hem_curve,
            'asymmetry_factor': self.asymmetry_factor,
            'waist_suppression': self.waist_suppression,
            'shoulder_slope': self.shoulder_slope
        }
    
    @classmethod
    def create_default(cls, garment_type: str) -> 'ConstructionParameters':
        """Create default construction parameters for a garment type"""
        if garment_type == "jacket":
            return cls(
                ease_amounts={
                    'chest': 100.0,  # mm
                    'waist': 80.0,
                    'hip': 90.0,
                    'armhole': 25.0
                },
                dart_positions={
                    'front_waist': (0.5, 0.0),  # Relative position
                    'back_waist': (0.4, 0.0),
                    'shoulder': (0.3, 0.0)
                },
                dart_intakes={
                    'front_waist': 20.0,  # mm
                    'back_waist': 30.0,
                    'shoulder': 15.0
                },
                seam_allowances={
                    'side_seam': 15.0,
                    'shoulder_seam': 12.0,
                    'armhole': 10.0,
                    'hem': 40.0
                },
                lapel_width=85.0,
                lapel_roll_line=30.0
            )
        elif garment_type == "dress":
            return cls(
                ease_amounts={
                    'bust': 80.0,
                    'waist': 60.0,
                    'hip': 100.0
                },
                dart_positions={
                    'bust': (0.4, 0.0),
                    'waist': (0.5, 0.0)
                },
                dart_intakes={
                    'bust': 25.0,
                    'waist': 35.0
                }
            )
        else:
            return cls()
    
    @classmethod
    def from_cppn_outputs(cls, cppn_outputs: Dict[str, Dict], garment_type: str) -> 'ConstructionParameters':
        """Create construction parameters from CPPN outputs"""
        params = cls.create_default(garment_type)
        
        # Extract and scale CPPN outputs to realistic construction values
        if garment_type == "jacket":
            params = cls._process_jacket_parameters(params, cppn_outputs)
        elif garment_type == "dress":
            params = cls._process_dress_parameters(params, cppn_outputs)
        
        return params
    
    @classmethod
    def _process_jacket_parameters(cls, params: 'ConstructionParameters', 
                                  outputs: Dict[str, Dict]) -> 'ConstructionParameters':
        """Process CPPN outputs for jacket construction"""
        
        # Extract key construction outputs
        if 'chest_point' in outputs:
            chest_out = outputs['chest_point']
            # Scale ease based on CPPN output (-1 to 1) to realistic range (50-150mm)
            params.ease_amounts['chest'] = 50 + (chest_out.get('ease', 0) + 1) * 50
        
        if 'waist_point' in outputs:
            waist_out = outputs['waist_point']
            params.ease_amounts['waist'] = 30 + (waist_out.get('ease', 0) + 1) * 50
            params.waist_suppression = waist_out.get('suppression', 0) * 0.3
        
        if 'shoulder_point' in outputs:
            shoulder_out = outputs['shoulder_point']
            params.shoulder_slope = shoulder_out.get('slope', 0) * 0.2
            params.dart_intakes['shoulder'] = 5 + (shoulder_out.get('dart', 0) + 1) * 15
        
        if 'lapel_point' in outputs:
            lapel_out = outputs['lapel_point']
            params.lapel_width = 60 + (lapel_out.get('width', 0) + 1) * 30
            params.lapel_roll_line = 20 + (lapel_out.get('roll', 0) + 1) * 20
        
        # Process asymmetry
        asymmetry_sum = sum(out.get('asymmetry', 0) for out in outputs.values())
        params.asymmetry_factor = np.clip(asymmetry_sum / len(outputs) * 0.2, -0.3, 0.3)
        
        # Process seam curves
        params.seam_curves = cls._generate_seam_curves(outputs)
        
        return params
    
    @classmethod
    def _process_dress_parameters(cls, params: 'ConstructionParameters', 
                                 outputs: Dict[str, Dict]) -> 'ConstructionParameters':
        """Process CPPN outputs for dress construction"""
        
        if 'bust_point' in outputs:
            bust_out = outputs['bust_point']
            params.ease_amounts['bust'] = 40 + (bust_out.get('ease', 0) + 1) * 40
            params.dart_intakes['bust'] = 10 + (bust_out.get('dart', 0) + 1) * 20
        
        if 'waist_point' in outputs:
            waist_out = outputs['waist_point']
            params.ease_amounts['waist'] = 20 + (waist_out.get('ease', 0) + 1) * 40
            params.dart_intakes['waist'] = 15 + (waist_out.get('dart', 0) + 1) * 25
        
        if 'hip_point' in outputs:
            hip_out = outputs['hip_point']
            params.ease_amounts['hip'] = 60 + (hip_out.get('ease', 0) + 1) * 60
        
        if 'hem_point' in outputs:
            hem_out = outputs['hem_point']
            params.hem_curve = hem_out.get('curve', 0) * 50  # mm
        
        return params
    
    @classmethod
    def _generate_seam_curves(cls, outputs: Dict[str, Dict]) -> Dict[str, List[Tuple[float, float]]]:
        """Generate seam curve control points from CPPN outputs"""
        curves = {}
        
        # Side seam curve
        if 'chest_point' in outputs and 'waist_point' in outputs and 'hip_point' in outputs:
            chest_curve = outputs['chest_point'].get('curve', 0)
            waist_curve = outputs['waist_point'].get('curve', 0)
            hip_curve = outputs['hip_point'].get('curve', 0)
            
            # Generate control points for side seam
            curves['side_seam'] = [
                (0.0, chest_curve * 10),    # Chest level
                (0.33, waist_curve * 15),   # Waist level
                (0.66, hip_curve * 10),     # Hip level
                (1.0, 0.0)                  # Hem level
            ]
        
        # Armhole curve
        if 'armhole_front' in outputs and 'armhole_back' in outputs:
            front_curve = outputs['armhole_front'].get('curve', 0)
            back_curve = outputs['armhole_back'].get('curve', 0)
            
            curves['armhole'] = [
                (0.0, front_curve * 8),     # Front armhole
                (0.25, front_curve * 12),   # Front pitch point
                (0.5, 0.0),                 # Shoulder point
                (0.75, back_curve * 10),    # Back pitch point
                (1.0, back_curve * 6)       # Back armhole
            ]
        
        return curves

class FashionActivations:
    """Fashion-specific activation functions for garment construction"""
    
    @staticmethod
    def sine(x: float) -> float:
        return math.sin(x)
    
    @staticmethod
    def cosine(x: float) -> float:
        return math.cos(x)
    
    @staticmethod
    def gaussian(x: float) -> float:
        return math.exp(-x * x / 2.0)
    
    @staticmethod
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))
    
    @staticmethod
    def tanh(x: float) -> float:
        return math.tanh(x)
    
    @staticmethod
    def linear(x: float) -> float:
        return x
    
    @staticmethod
    def ease_curve(x: float) -> float:
        """Smooth ease distribution for garment fit"""
        # Creates a curve that's minimal at body surface, increases smoothly
        return 1 - math.exp(-abs(x) * 2)
    
    @staticmethod
    def dart_shape(x: float) -> float:
        """Dart intake curve - sharp at point, smooth at base"""
        return math.exp(-x*x*4) * math.sin(x*3)
    
    @staticmethod
    def seam_curve(x: float) -> float:
        """Natural seam curvature following body lines"""
        return math.sin(x * math.pi) * math.exp(-abs(x))
    
    @staticmethod
    def golden_ratio(x: float) -> float:
        """Golden ratio proportioning"""
        phi = 1.618033988749
        return math.sin(x * phi) * math.exp(-abs(x) / phi)
    
    @staticmethod
    def body_curve(x: float) -> float:
        """Body-following curves for natural fit"""
        return math.tanh(x) * math.cos(x * 2) * 0.7
    
    @staticmethod
    def fabric_tension(x: float) -> float:
        """Simulates fabric tension and draping"""
        return math.sin(x * 0.5) * math.exp(-abs(x) * 0.3)

class GarmentCPPN:
    """CPPN specifically designed for garment construction parameters"""
    
    def __init__(self, genome: GarmentGenome, garment_type: str):
        self.genome = genome
        self.garment_type = garment_type
        self.activation_map = {
            FashionActivationFunction.SINE: FashionActivations.sine,
            FashionActivationFunction.COSINE: FashionActivations.cosine,
            FashionActivationFunction.GAUSSIAN: FashionActivations.gaussian,
            FashionActivationFunction.SIGMOID: FashionActivations.sigmoid,
            FashionActivationFunction.TANH: FashionActivations.tanh,
            FashionActivationFunction.LINEAR: FashionActivations.linear,
            FashionActivationFunction.EASE_CURVE: FashionActivations.ease_curve,
            FashionActivationFunction.DART_SHAPE: FashionActivations.dart_shape,
            FashionActivationFunction.SEAM_CURVE: FashionActivations.seam_curve,
            FashionActivationFunction.GOLDEN_RATIO: FashionActivations.golden_ratio,
            FashionActivationFunction.BODY_CURVE: FashionActivations.body_curve,
            FashionActivationFunction.FABRIC_TENSION: FashionActivations.fabric_tension,
        }
        self._build_network()
    
    def _build_network(self):
        """Build network from genome"""
        self.nodes = {}
        self.node_values = {}
        
        # Create nodes
        for node in self.genome.nodes:
            try:
                activation = FashionActivationFunction(node['activation'])
            except ValueError:
                activation = FashionActivationFunction.LINEAR
            
            self.nodes[node['id']] = {
                'type': node['type'],
                'activation': activation,
                'bias': node.get('bias', 0.0),
                'inputs': []
            }
        
        # Create connections
        for conn in self.genome.connections:
            if conn['enabled'] and conn['target'] in self.nodes:
                self.nodes[conn['target']]['inputs'].append({
                    'source': conn['source'],
                    'weight': conn['weight']
                })
    
    def evaluate(self, x: float, y: float, z: float) -> Dict[str, float]:
        """Evaluate CPPN at 3D coordinates to get construction parameters"""
        
        # Initialize input values
        self.node_values = {
            'x': x,
            'y': y, 
            'z': z,
            'x^2': x*x,
            'y^2': y*y,
            'z^2': z*z,
            'r': math.sqrt(x*x + y*y + z*z),
            'theta': math.atan2(y, x),
            'phi': math.atan2(math.sqrt(x*x + y*y), z),
            'xy': x*y,
            'xz': x*z,
            'yz': y*z
        }
        
        # Evaluate network
        evaluated = set(self.node_values.keys())
        
        # Keep evaluating until all nodes are processed
        max_iterations = 100
        iteration = 0
        
        while len(evaluated) < len(self.nodes) + len(self.node_values) and iteration < max_iterations:
            iteration += 1
            
            for node_id, node in self.nodes.items():
                if node_id not in evaluated:
                    # Check if all inputs are ready
                    inputs_ready = all(
                        inp['source'] in self.node_values
                        for inp in node['inputs']
                    )
                    
                    if inputs_ready:
                        # Calculate node value
                        net_input = node['bias']
                        for inp in node['inputs']:
                            source_val = self.node_values.get(inp['source'], 0.0)
                            net_input += source_val * inp['weight']
                        
                        # Apply activation function
                        activation_func = self.activation_map[node['activation']]
                        self.node_values[node_id] = activation_func(net_input)
                        evaluated.add(node_id)
        
        # Return construction-specific outputs
        return self._extract_construction_outputs()
    
    def _extract_construction_outputs(self) -> Dict[str, float]:
        """Extract construction parameters from node outputs"""
        outputs = {}
        
        # Map node outputs to construction parameters
        construction_mappings = {
            'ease': ['ease_output', 'ease', 'fit'],
            'dart': ['dart_output', 'dart', 'intake'],
            'curve': ['curve_output', 'curve', 'seam'],
            'suppression': ['suppression_output', 'suppression', 'waist'],
            'slope': ['slope_output', 'slope', 'shoulder'],
            'asymmetry': ['asymmetry_output', 'asymmetry', 'balance'],
            'width': ['width_output', 'width', 'lapel'],
            'roll': ['roll_output', 'roll', 'lapel_roll']
        }
        
        for param, node_names in construction_mappings.items():
            for node_name in node_names:
                if node_name in self.node_values:
                    outputs[param] = self.node_values[node_name]
                    break
            
            # If no specific node found, try to find any output node
            if param not in outputs:
                output_nodes = [nid for nid, node in self.nodes.items() 
                              if node['type'] == 'output' and nid in self.node_values]
                if output_nodes:
                    # Use hash of parameter name to consistently select node
                    node_idx = hash(param) % len(output_nodes)
                    outputs[param] = self.node_values[output_nodes[node_idx]]
        
        # Ensure all outputs are in reasonable range
        for key in outputs:
            outputs[key] = np.clip(outputs[key], -1.0, 1.0)
        
        return outputs

class GarmentGenomeFactory:
    """Factory for creating and evolving garment genomes"""
    
    @staticmethod
    def create_construction_genome(garment_type: str) -> GarmentGenome:
        """Create a genome optimized for garment construction"""
        genome = GarmentGenome(garment_type=garment_type)
        
        # Create output nodes for construction parameters
        if garment_type == "jacket":
            output_nodes = [
                {'id': 'ease_output', 'type': 'output', 'activation': 'ease_curve', 'bias': 0.0},
                {'id': 'dart_output', 'type': 'output', 'activation': 'dart_shape', 'bias': 0.0},
                {'id': 'curve_output', 'type': 'output', 'activation': 'seam_curve', 'bias': 0.0},
                {'id': 'suppression_output', 'type': 'output', 'activation': 'body_curve', 'bias': 0.0},
                {'id': 'slope_output', 'type': 'output', 'activation': 'linear', 'bias': 0.0},
                {'id': 'asymmetry_output', 'type': 'output', 'activation': 'sine', 'bias': 0.0},
                {'id': 'lapel_width_output', 'type': 'output', 'activation': 'golden_ratio', 'bias': 0.0},
                {'id': 'lapel_roll_output', 'type': 'output', 'activation': 'sigmoid', 'bias': 0.0}
            ]
        elif garment_type == "dress":
            output_nodes = [
                {'id': 'ease_output', 'type': 'output', 'activation': 'ease_curve', 'bias': 0.0},
                {'id': 'dart_output', 'type': 'output', 'activation': 'dart_shape', 'bias': 0.0},
                {'id': 'curve_output', 'type': 'output', 'activation': 'seam_curve', 'bias': 0.0},
                {'id': 'drape_output', 'type': 'output', 'activation': 'fabric_tension', 'bias': 0.0},
                {'id': 'flare_output', 'type': 'output', 'activation': 'golden_ratio', 'bias': 0.0},
                {'id': 'hem_output', 'type': 'output', 'activation': 'sine', 'bias': 0.0}
            ]
        else:
            # Generic outputs
            output_nodes = [
                {'id': 'ease_output', 'type': 'output', 'activation': 'ease_curve', 'bias': 0.0},
                {'id': 'curve_output', 'type': 'output', 'activation': 'seam_curve', 'bias': 0.0},
                {'id': 'proportion_output', 'type': 'output', 'activation': 'golden_ratio', 'bias': 0.0}
            ]
        
        # Create hidden nodes for complexity
        hidden_nodes = []
        num_hidden = random.randint(3, 8)
        
        for i in range(num_hidden):
            activation = random.choice(list(FashionActivationFunction))
            hidden_nodes.append({
                'id': f'h{i}',
                'type': 'hidden',
                'activation': activation.value,
                'bias': random.uniform(-1, 1)
            })
        
        genome.nodes = output_nodes + hidden_nodes
        
        # Create connections
        genome.connections = GarmentGenomeFactory._create_construction_connections(
            output_nodes, hidden_nodes
        )
        
        return genome
    
    @staticmethod
    def _create_construction_connections(output_nodes: List[Dict], 
                                       hidden_nodes: List[Dict]) -> List[Dict]:
        """Create connections optimized for construction parameter generation"""
        connections = []
        
        # Input sources
        input_sources = ['x', 'y', 'z', 'r', 'theta', 'phi', 'xy', 'xz', 'yz']
        all_sources = input_sources + [node['id'] for node in hidden_nodes]
        
        # Connect inputs to hidden nodes
        for hidden in hidden_nodes:
            num_connections = random.randint(2, 4)
            sources = random.sample(input_sources, min(num_connections, len(input_sources)))
            
            for source in sources:
                connections.append({
                    'source': source,
                    'target': hidden['id'],
                    'weight': random.uniform(-2, 2),
                    'enabled': True
                })
        
        # Connect to outputs (ensure each output gets connections)
        for output in output_nodes:
            num_connections = random.randint(2, 5)
            sources = random.sample(all_sources, min(num_connections, len(all_sources)))
            
            for source in sources:
                connections.append({
                    'source': source,
                    'target': output['id'],
                    'weight': random.uniform(-1.5, 1.5),
                    'enabled': True
                })
        
        return connections
    
    @staticmethod
    def mutate_construction_genome(genome: GarmentGenome, 
                                  mutation_rate: float = 0.15) -> GarmentGenome:
        """Mutate genome with construction-specific considerations"""
        new_genome = GarmentGenome(
            nodes=[node.copy() for node in genome.nodes],
            connections=[conn.copy() for conn in genome.connections],
            garment_type=genome.garment_type,
            generation=genome.generation + 1,
            parent_ids=[f"gen{genome.generation}"]
        )
        
        # Weight mutations (most important for construction)
        for conn in new_genome.connections:
            if random.random() < mutation_rate:
                if random.random() < 0.8:  # Small perturbation
                    conn['weight'] += random.uniform(-0.3, 0.3)
                else:  # Large change
                    conn['weight'] = random.uniform(-2, 2)
                
                # Keep weights reasonable for construction
                conn['weight'] = np.clip(conn['weight'], -3, 3)
        
        # Bias mutations
        for node in new_genome.nodes:
            if random.random() < mutation_rate * 0.7:
                node['bias'] += random.uniform(-0.2, 0.2)
                node['bias'] = np.clip(node['bias'], -2, 2)
        
        # Activation function mutations (careful with construction functions)
        if random.random() < mutation_rate * 0.4:
            mutable_nodes = [n for n in new_genome.nodes 
                           if n['type'] in ['hidden', 'output']]
            if mutable_nodes:
                node_to_mutate = random.choice(mutable_nodes)
                
                # Prefer construction-relevant activations
                if node_to_mutate['type'] == 'output':
                    construction_activations = [
                        'ease_curve', 'dart_shape', 'seam_curve', 
                        'golden_ratio', 'body_curve', 'fabric_tension'
                    ]
                    node_to_mutate['activation'] = random.choice(construction_activations)
                else:
                    node_to_mutate['activation'] = random.choice(list(FashionActivationFunction)).value
        
        # Structural mutations (less frequent)
        if random.random() < mutation_rate * 0.2:
            GarmentGenomeFactory._add_construction_connection(new_genome)
        
        if random.random() < mutation_rate * 0.1:
            GarmentGenomeFactory._add_construction_node(new_genome)
        
        return new_genome
    
    @staticmethod
    def _add_construction_connection(genome: GarmentGenome):
        """Add connection with construction considerations"""
        input_sources = ['x', 'y', 'z', 'r', 'theta', 'phi', 'xy', 'xz', 'yz']
        all_sources = input_sources + [n['id'] for n in genome.nodes if n['type'] != 'output']
        targets = [n['id'] for n in genome.nodes if n['type'] in ['hidden', 'output']]
        
        if all_sources and targets:
            source = random.choice(all_sources)
            target = random.choice(targets)
            
            # Avoid duplicates
            existing = [(c['source'], c['target']) for c in genome.connections]
            if (source, target) not in existing:
                genome.connections.append({
                    'source': source,
                    'target': target,
                    'weight': random.uniform(-1.5, 1.5),
                    'enabled': True
                })
    
    @staticmethod
    def _add_construction_node(genome: GarmentGenome):
        """Add node optimized for construction"""
        active_connections = [c for c in genome.connections if c['enabled']]
        if active_connections:
            # Split existing connection
            conn = random.choice(active_connections)
            
            # Create new hidden node
            new_node_id = f"h{len([n for n in genome.nodes if n['type'] == 'hidden']) + 1}"
            
            # Choose construction-relevant activation
            construction_activations = [
                'ease_curve', 'dart_shape', 'seam_curve', 'body_curve',
                'golden_ratio', 'fabric_tension', 'gaussian', 'sigmoid'
            ]
            
            new_node = {
                'id': new_node_id,
                'type': 'hidden',
                'activation': random.choice(construction_activations),
                'bias': random.uniform(-0.5, 0.5)
            }
            genome.nodes.append(new_node)
            
            # Disable old connection
            conn['enabled'] = False
            
            # Add new connections
            genome.connections.extend([
                {
                    'source': conn['source'],
                    'target': new_node_id,
                    'weight': 1.0,
                    'enabled': True
                },
                {
                    'source': new_node_id,
                    'target': conn['target'],
                    'weight': conn['weight'],
                    'enabled': True
                }
            ])
    
    @staticmethod
    def crossover_construction_genomes(parent1: GarmentGenome, 
                                     parent2: GarmentGenome) -> GarmentGenome:
        """Crossover optimized for construction genomes"""
        # Ensure same garment type
        if parent1.garment_type != parent2.garment_type:
            return GarmentGenomeFactory.mutate_construction_genome(parent1)
        
        child = GarmentGenome(
            garment_type=parent1.garment_type,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[f"gen{parent1.generation}", f"gen{parent2.generation}"]
        )
        
        # Inherit nodes from fitter parent
        fitter_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
        child.nodes = [node.copy() for node in fitter_parent.nodes]
        
        # Crossover connections intelligently
        p1_connections = {(c['source'], c['target']): c for c in parent1.connections}
        p2_connections = {(c['source'], c['target']): c for c in parent2.connections}
        
        all_connections = set(p1_connections.keys()) | set(p2_connections.keys())
        
        for conn_key in all_connections:
            if conn_key in p1_connections and conn_key in p2_connections:
                # Both parents have this connection - blend weights
                p1_conn = p1_connections[conn_key]
                p2_conn = p2_connections[conn_key]
                
                new_conn = p1_conn.copy()
                # Weighted average based on fitness
                total_fitness = parent1.fitness + parent2.fitness + 0.01  # Avoid division by zero
                w1 = parent1.fitness / total_fitness
                w2 = parent2.fitness / total_fitness
                
                new_conn['weight'] = w1 * p1_conn['weight'] + w2 * p2_conn['weight']
                child.connections.append(new_conn)
            
            elif conn_key in p1_connections:
                child.connections.append(p1_connections[conn_key].copy())
            else:
                child.connections.append(p2_connections[conn_key].copy())
        
        return child