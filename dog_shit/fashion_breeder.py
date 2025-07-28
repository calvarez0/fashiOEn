import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Callable, Tuple, Dict, Optional
import random
import math
from enum import Enum
from abc import ABC, abstractmethod
import pickle
import json
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.patches as patches
from matplotlib.path import Path
import seaborn as sns

class ActivationFunction(Enum):
    """Fashion-informed activation functions"""
    SINE = "sine"              # Natural waves, fabric flow
    COSINE = "cosine"          # Complementary waves
    GAUSSIAN = "gaussian"      # Soft transitions, natural curves
    SIGMOID = "sigmoid"        # Smooth transitions
    TANH = "tanh"             # Symmetric transitions
    LINEAR = "linear"         # Sharp lines, structural elements
    ABS = "abs"               # Sharp contrasts, angular cuts
    SQUARE = "square"         # Bold statements, geometric fashion
    FABRIC_DRAPE = "fabric_drape"    # Physics-informed fabric fall
    GOLDEN_RATIO = "golden_ratio"    # Proportional harmony
    ASYMMETRY = "asymmetry"          # Controlled imbalance
    TEXTURE_WAVE = "texture_wave"    # Surface pattern generation
    SILHOUETTE = "silhouette"        # Body-aware shaping

@dataclass
class FashionGenome:
    """Represents the genetic code of a fashion design"""
    nodes: List[Dict] = field(default_factory=list)
    connections: List[Dict] = field(default_factory=list)
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'nodes': self.nodes,
            'connections': self.connections,
            'fitness': self.fitness,
            'generation': self.generation,
            'parent_ids': self.parent_ids
        }

class FashionActivations:
    """Collection of fashion-specific activation functions"""
    
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
    def abs_func(x: float) -> float:
        return abs(x)
    
    @staticmethod
    def square(x: float) -> float:
        return x * x if x >= 0 else -(x * x)
    
    @staticmethod
    def fabric_drape(x: float) -> float:
        """Physics-informed fabric draping function"""
        # Simulates natural catenary curve of hanging fabric
        return math.cosh(x * 2) / math.cosh(2) - 1
    
    @staticmethod
    def golden_ratio(x: float) -> float:
        """Golden ratio-based proportioning"""
        phi = 1.618033988749
        return math.sin(x * phi) * math.exp(-abs(x) / phi)
    
    @staticmethod
    def asymmetry(x: float) -> float:
        """Controlled asymmetry function"""
        return math.sin(x) + 0.3 * math.sin(3 * x) + 0.1 * math.sin(7 * x)
    
    @staticmethod
    def texture_wave(x: float) -> float:
        """Multi-frequency texture generation"""
        return (math.sin(x * 5) * 0.5 + 
                math.sin(x * 13) * 0.3 + 
                math.sin(x * 29) * 0.2)
    
    @staticmethod
    def silhouette(x: float) -> float:
        """Body-aware silhouette shaping"""
        # Creates natural body-following curves
        return math.exp(-x*x/4) * math.sin(x*2) + 0.1*x

class FashionCPPN:
    """Compositional Pattern Producing Network for Fashion Design"""
    
    def __init__(self, genome: FashionGenome):
        self.genome = genome
        self.activation_map = {
            ActivationFunction.SINE: FashionActivations.sine,
            ActivationFunction.COSINE: FashionActivations.cosine,
            ActivationFunction.GAUSSIAN: FashionActivations.gaussian,
            ActivationFunction.SIGMOID: FashionActivations.sigmoid,
            ActivationFunction.TANH: FashionActivations.tanh,
            ActivationFunction.LINEAR: FashionActivations.linear,
            ActivationFunction.ABS: FashionActivations.abs_func,
            ActivationFunction.SQUARE: FashionActivations.square,
            ActivationFunction.FABRIC_DRAPE: FashionActivations.fabric_drape,
            ActivationFunction.GOLDEN_RATIO: FashionActivations.golden_ratio,
            ActivationFunction.ASYMMETRY: FashionActivations.asymmetry,
            ActivationFunction.TEXTURE_WAVE: FashionActivations.texture_wave,
            ActivationFunction.SILHOUETTE: FashionActivations.silhouette,
        }
        self._build_network()
    
    def _build_network(self):
        """Build the network from genome"""
        self.nodes = {}
        self.node_values = {}
        
        # Create nodes
        for node in self.genome.nodes:
            self.nodes[node['id']] = {
                'type': node['type'],
                'activation': ActivationFunction(node['activation']),
                'bias': node.get('bias', 0.0),
                'inputs': []
            }
        
        # Create connections
        for conn in self.genome.connections:
            if conn['enabled']:
                target_id = conn['target']
                if target_id in self.nodes:
                    self.nodes[target_id]['inputs'].append({
                        'source': conn['source'],
                        'weight': conn['weight']
                    })
    
    def evaluate(self, x: float, y: float, d: float = 0.0) -> Dict[str, float]:
        """Evaluate network at coordinates (x, y, d=bias)"""
        # Reset node values
        self.node_values = {
            'x': x,
            'y': y,
            'd': d,  # bias/distance input
            'x^2': x*x,
            'y^2': y*y,
            'xy': x*y,
            'r': math.sqrt(x*x + y*y),  # radius
            'theta': math.atan2(y, x),   # angle
        }
        
        # Evaluate nodes in topological order
        evaluated = set(['x', 'y', 'd', 'x^2', 'y^2', 'xy', 'r', 'theta'])
        
        while len(evaluated) < len(self.nodes) + 8:  # +8 for input nodes
            for node_id, node in self.nodes.items():
                if node_id not in evaluated:
                    # Check if all inputs are ready
                    inputs_ready = all(
                        inp['source'] in evaluated or inp['source'] in self.node_values
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
        
        # Return output values
        outputs = {}
        for node_id, node in self.nodes.items():
            if node['type'] == 'output':
                outputs[node_id] = self.node_values.get(node_id, 0.0)
        
        return outputs

class FashionGenomeFactory:
    """Factory for creating and mutating fashion genomes"""
    
    @staticmethod
    def create_minimal_genome() -> FashionGenome:
        """Create a minimal viable fashion genome"""
        genome = FashionGenome()
        
        # Output nodes for different fashion aspects
        output_nodes = [
            {'id': 'silhouette', 'type': 'output', 'activation': 'silhouette', 'bias': 0.0},
            {'id': 'texture', 'type': 'output', 'activation': 'texture_wave', 'bias': 0.0},
            {'id': 'color_h', 'type': 'output', 'activation': 'sine', 'bias': 0.0},
            {'id': 'color_s', 'type': 'output', 'activation': 'sigmoid', 'bias': 0.0},
            {'id': 'color_l', 'type': 'output', 'activation': 'gaussian', 'bias': 0.0},
            {'id': 'asymmetry', 'type': 'output', 'activation': 'asymmetry', 'bias': 0.0},
            {'id': 'structure', 'type': 'output', 'activation': 'fabric_drape', 'bias': 0.0},
        ]
        
        # Hidden nodes for complexity
        hidden_nodes = [
            {'id': 'h1', 'type': 'hidden', 'activation': 'golden_ratio', 'bias': random.uniform(-1, 1)},
            {'id': 'h2', 'type': 'hidden', 'activation': 'sine', 'bias': random.uniform(-1, 1)},
            {'id': 'h3', 'type': 'hidden', 'activation': 'gaussian', 'bias': random.uniform(-1, 1)},
        ]
        
        genome.nodes = output_nodes + hidden_nodes
        
        # Create random connections
        input_sources = ['x', 'y', 'd', 'r', 'theta', 'xy']
        all_sources = input_sources + [node['id'] for node in hidden_nodes]
        
        connections = []
        
        # Connect inputs to hidden nodes
        for hidden in hidden_nodes:
            num_connections = random.randint(1, 3)
            sources = random.sample(input_sources, num_connections)
            for source in sources:
                connections.append({
                    'source': source,
                    'target': hidden['id'],
                    'weight': random.uniform(-2, 2),
                    'enabled': True
                })
        
        # Connect to outputs
        for output in output_nodes:
            num_connections = random.randint(1, 4)
            sources = random.sample(all_sources, min(num_connections, len(all_sources)))
            for source in sources:
                connections.append({
                    'source': source,
                    'target': output['id'],
                    'weight': random.uniform(-2, 2),
                    'enabled': True
                })
        
        genome.connections = connections
        return genome
    
    @staticmethod
    def mutate_genome(genome: FashionGenome, mutation_rate: float = 0.1) -> FashionGenome:
        """Mutate a genome following NEAT-style evolution"""
        new_genome = FashionGenome(
            nodes=genome.nodes.copy(),
            connections=genome.connections.copy(),
            generation=genome.generation + 1,
            parent_ids=[f"gen{genome.generation}"]
        )
        
        # Weight mutations (most common)
        for conn in new_genome.connections:
            if random.random() < mutation_rate:
                if random.random() < 0.9:  # Perturb existing weight
                    conn['weight'] += random.uniform(-0.5, 0.5)
                else:  # Replace with new weight
                    conn['weight'] = random.uniform(-2, 2)
        
        # Bias mutations
        for node in new_genome.nodes:
            if random.random() < mutation_rate * 0.5:
                node['bias'] += random.uniform(-0.3, 0.3)
        
        # Activation function mutations
        if random.random() < mutation_rate * 0.3:
            node_to_mutate = random.choice([n for n in new_genome.nodes if n['type'] != 'input'])
            node_to_mutate['activation'] = random.choice(list(ActivationFunction)).value
        
        # Structural mutations (less common)
        if random.random() < mutation_rate * 0.2:
            FashionGenomeFactory._add_connection(new_genome)
        
        if random.random() < mutation_rate * 0.1:
            FashionGenomeFactory._add_node(new_genome)
        
        return new_genome
    
    @staticmethod
    def _add_connection(genome: FashionGenome):
        """Add a new connection"""
        input_sources = ['x', 'y', 'd', 'r', 'theta', 'xy']
        all_sources = input_sources + [n['id'] for n in genome.nodes if n['type'] != 'output']
        targets = [n['id'] for n in genome.nodes if n['type'] != 'input']
        
        if all_sources and targets:
            source = random.choice(all_sources)
            target = random.choice(targets)
            
            # Avoid duplicate connections
            existing = [(c['source'], c['target']) for c in genome.connections]
            if (source, target) not in existing:
                genome.connections.append({
                    'source': source,
                    'target': target,
                    'weight': random.uniform(-2, 2),
                    'enabled': True
                })
    
    @staticmethod
    def _add_node(genome: FashionGenome):
        """Add a new node by splitting an existing connection"""
        active_connections = [c for c in genome.connections if c['enabled']]
        if active_connections:
            # Choose random connection to split
            conn = random.choice(active_connections)
            
            # Create new node
            new_node_id = f"h{len([n for n in genome.nodes if n['type'] == 'hidden']) + 1}"
            new_node = {
                'id': new_node_id,
                'type': 'hidden',
                'activation': random.choice(list(ActivationFunction)).value,
                'bias': 0.0
            }
            genome.nodes.append(new_node)
            
            # Disable old connection
            conn['enabled'] = False
            
            # Add two new connections
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
    def crossover(parent1: FashionGenome, parent2: FashionGenome) -> FashionGenome:
        """Create offspring through crossover"""
        child = FashionGenome(
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[f"gen{parent1.generation}", f"gen{parent2.generation}"]
        )
        
        # Inherit nodes (take union, prefer fitter parent for conflicts)
        fitter_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
        
        child.nodes = fitter_parent.nodes.copy()
        
        # Crossover connections
        p1_connections = {(c['source'], c['target']): c for c in parent1.connections}
        p2_connections = {(c['source'], c['target']): c for c in parent2.connections}
        
        all_connections = set(p1_connections.keys()) | set(p2_connections.keys())
        
        for conn_key in all_connections:
            if conn_key in p1_connections and conn_key in p2_connections:
                # Both parents have this connection - randomly choose
                chosen_conn = random.choice([p1_connections[conn_key], p2_connections[conn_key]])
                child.connections.append(chosen_conn.copy())
            elif conn_key in p1_connections:
                child.connections.append(p1_connections[conn_key].copy())
            else:
                child.connections.append(p2_connections[conn_key].copy())
        
        return child

class FashionRenderer:
    """Renders fashion designs from CPPN outputs"""
    
    def __init__(self, width: int = 400, height: int = 600):
        self.width = width
        self.height = height
    
    def render_garment(self, cppn: FashionCPPN) -> np.ndarray:
        """Render a complete garment using CPPN"""
        # Create coordinate grid
        x_coords = np.linspace(-1, 1, self.width)
        y_coords = np.linspace(-1, 1, self.height)
        
        # Initialize output arrays
        silhouette = np.zeros((self.height, self.width))
        texture = np.zeros((self.height, self.width))
        hue = np.zeros((self.height, self.width))
        saturation = np.zeros((self.height, self.width))
        lightness = np.zeros((self.height, self.width))
        asymmetry = np.zeros((self.height, self.width))
        structure = np.zeros((self.height, self.width))
        
        # Evaluate CPPN at each coordinate
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                outputs = cppn.evaluate(x, y, 0.0)
                
                silhouette[i, j] = outputs.get('silhouette', 0.0)
                texture[i, j] = outputs.get('texture', 0.0)
                hue[i, j] = outputs.get('color_h', 0.0)
                saturation[i, j] = outputs.get('color_s', 0.0)
                lightness[i, j] = outputs.get('color_l', 0.0)
                asymmetry[i, j] = outputs.get('asymmetry', 0.0)
                structure[i, j] = outputs.get('structure', 0.0)
        
        # Normalize outputs
        silhouette = self._normalize(silhouette)
        texture = self._normalize(texture)
        hue = (self._normalize(hue) * 360).astype(np.uint16)  # 0-360 degrees
        saturation = np.clip(self._normalize(saturation), 0, 1)
        lightness = np.clip(self._normalize(lightness) * 0.5 + 0.3, 0.1, 0.9)  # Keep reasonable range
        
        # Create garment mask using silhouette and structure
        garment_mask = self._create_garment_mask(silhouette, structure, asymmetry)
        
        # Generate final image
        image = self._compose_garment(garment_mask, texture, hue, saturation, lightness)
        
        return image
    
    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range"""
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            return (arr - arr_min) / (arr_max - arr_min)
        return np.zeros_like(arr)
    
    def _create_garment_mask(self, silhouette: np.ndarray, structure: np.ndarray, asymmetry: np.ndarray) -> np.ndarray:
        """Create realistic garment silhouette"""
        # Combine silhouette with structure for realistic garment shape
        combined = silhouette + 0.3 * structure + 0.1 * asymmetry
        
        # Create body-aware mask
        height, width = combined.shape
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Normalize coordinates to [-1, 1]
        x_norm = (x_coords - width/2) / (width/2)
        y_norm = (y_coords - height/2) / (height/2)
        
        # Create basic jacket silhouette
        # Shoulders (wider at top)
        shoulder_width = 0.7 + 0.2 * np.sin(y_norm * np.pi)
        shoulder_mask = np.abs(x_norm) < shoulder_width * (1 - 0.3 * np.maximum(0, y_norm))
        
        # Torso tapering
        torso_width = 0.6 - 0.1 * np.maximum(0, y_norm - 0.2)
        torso_mask = np.abs(x_norm) < torso_width
        
        # Combine with CPPN output
        cppn_influence = 0.3
        final_mask = np.where(y_norm < 0, shoulder_mask, torso_mask)
        final_mask = final_mask & (combined > (0.5 - cppn_influence * combined))
        
        return final_mask.astype(np.float32)
    
    def _compose_garment(self, mask: np.ndarray, texture: np.ndarray, 
                        hue: np.ndarray, saturation: np.ndarray, lightness: np.ndarray) -> np.ndarray:
        """Compose final garment image"""
        height, width = mask.shape
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply colors only where garment exists
        for i in range(height):
            for j in range(width):
                if mask[i, j] > 0.1:  # Garment area
                    # Convert HSL to RGB
                    h = hue[i, j] / 360.0
                    s = saturation[i, j] * mask[i, j]  # Fade saturation at edges
                    l = lightness[i, j] + 0.1 * texture[i, j]  # Texture affects brightness
                    
                    rgb = self._hsl_to_rgb(h, s, l)
                    image[i, j] = [int(c * 255) for c in rgb]
                else:
                    # Background
                    image[i, j] = [240, 240, 245]  # Light gray background
        
        return image
    
    def _hsl_to_rgb(self, h: float, s: float, l: float) -> Tuple[float, float, float]:
        """Convert HSL to RGB"""
        def hue_to_rgb(p: float, q: float, t: float) -> float:
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
        
        if s == 0:
            r = g = b = l  # Achromatic
        else:
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)
        
        return (r, g, b)

class FashionEvolver:
    """Main evolution engine for fashion design"""
    
    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.population: List[FashionGenome] = []
        self.generation = 0
        self.renderer = FashionRenderer()
        self.history: List[Dict] = []
    
    def initialize_population(self):
        """Create initial random population"""
        self.population = []
        for i in range(self.population_size):
            genome = FashionGenomeFactory.create_minimal_genome()
            genome.generation = self.generation
            self.population.append(genome)
        
        print(f"Initialized population of {self.population_size} fashion genomes")
    
    def render_population(self) -> List[np.ndarray]:
        """Render all designs in current population"""
        images = []
        for i, genome in enumerate(self.population):
            print(f"Rendering design {i+1}/{len(self.population)}...")
            cppn = FashionCPPN(genome)
            image = self.renderer.render_garment(cppn)
            images.append(image)
        return images
    
    def select_parents(self, num_parents: int = 8) -> List[FashionGenome]:
        """Select parents for reproduction using tournament selection"""
        parents = []
        tournament_size = 3
        
        for _ in range(num_parents):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def evolve_generation(self, selected_indices: List[int]):
        """Evolve to next generation based on user selections"""
        # Update fitness based on user selections
        for i, genome in enumerate(self.population):
            if i in selected_indices:
                genome.fitness = 1.0 + random.uniform(0, 0.5)  # Selected designs get high fitness
            else:
                genome.fitness = random.uniform(0, 0.3)  # Non-selected get low fitness
        
        # Record generation stats
        avg_fitness = sum(g.fitness for g in self.population) / len(self.population)
        max_fitness = max(g.fitness for g in self.population)
        self.history.append({
            'generation': self.generation,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'selected_count': len(selected_indices)
        })
        
        print(f"Generation {self.generation}: avg_fitness={avg_fitness:.3f}, max_fitness={max_fitness:.3f}")
        
        # Create next generation
        new_population = []
        
        # Elitism: keep best designs
        elite_count = max(2, len(selected_indices))
        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:elite_count]
        new_population.extend(elite)
        
        # Fill rest with offspring
        parents = self.select_parents(num_parents=min(8, len(selected_indices) * 2))
        
        while len(new_population) < self.population_size:
            if len(parents) >= 2:
                parent1, parent2 = random.sample(parents, 2)
                child = FashionGenomeFactory.crossover(parent1, parent2)
                child = FashionGenomeFactory.mutate_genome(child, mutation_rate=0.15)
                new_population.append(child)
            else:
                # Mutation only if not enough parents
                parent = random.choice(self.population)
                child = FashionGenomeFactory.mutate_genome(parent, mutation_rate=0.3)
                new_population.append(child)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        print(f"Evolved to generation {self.generation}")
    
    def save_population(self, filename: str):
        """Save current population to file"""
        data = {
            'generation': self.generation,
            'history': self.history,
            'population': [genome.to_dict() for genome in self.population]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved population to {filename}")
    
    def display_generation(self, images: List[np.ndarray], save_path: Optional[str] = None):
        """Display current generation in a grid"""
        n_images = len(images)
        cols = 5
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        if rows == 1:
            axes = [axes]
        if cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, image in enumerate(images):
            row, col = i // cols, i % cols
            axes[row][col].imshow(image)
            axes[row][col].set_title(f'Design {i}', fontsize=10)
            axes[row][col].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, rows * cols):
            row, col = i // cols, i % cols
            axes[row][col].axis('off')
        
        plt.suptitle(f'Fashion Evolution - Generation {self.generation}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

def run_interactive_evolution():
    """Run interactive fashion evolution"""
    print("🎨 Fashion CPPN Evolution System")
    print("=" * 50)
    print("Creating neural networks that understand fashion...")
    
    # Initialize evolution system
    evolver = FashionEvolver(population_size=16)
    evolver.initialize_population()
    
    # Main evolution loop
    for generation in range(10):  # Run for 10 generations
        print(f"\n👗 Generation {generation + 1}")
        print("-" * 30)
        
        # Render current population
        images = evolver.render_population()
        
        # Display designs
        save_path = f"generation_{generation + 1}.png"
        evolver.display_generation(images, save_path)
        
        # Get user selections (simulated for now)
        print("\nWhich designs do you like? (Enter indices 0-15, separated by spaces)")
        print("For demo, auto-selecting some interesting designs...")
        
        # Auto-select based on complexity and diversity
        selected = []
        for i, genome in enumerate(evolver.population):
            # Score based on network complexity and activation diversity
            complexity_score = len(genome.nodes) + len([c for c in genome.connections if c['enabled']])
            activation_diversity = len(set(node['activation'] for node in genome.nodes))
            
            if complexity_score > 12 and activation_diversity > 3:
                selected.append(i)
        
        # Ensure we have at least 3 selections
        if len(selected) < 3:
            selected = random.sample(range(len(evolver.population)), 3)
        
        print(f"Selected designs: {selected}")
        
        # Evolve to next generation
        if generation < 9:  # Don't evolve after last generation
            evolver.evolve_generation(selected)
    
    # Save final results
    evolver.save_population("final_fashion_population.json")
    print("\n🎉 Evolution complete! Final population saved.")
    
    # Display evolution history
    if evolver.history:
        plt.figure(figsize=(10, 6))
        generations = [h['generation'] for h in evolver.history]
        avg_fitness = [h['avg_fitness'] for h in evolver.history]
        max_fitness = [h['max_fitness'] for h in evolver.history]
        
        plt.plot(generations, avg_fitness, 'b-', label='Average Fitness', linewidth=2)
        plt.plot(generations, max_fitness, 'r-', label='Max Fitness', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fashion Evolution Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('evolution_progress.png', dpi=150, bbox_inches='tight')
        plt.show()

class FashionAnalyzer:
    """Analyze evolved fashion designs for patterns and insights"""
    
    @staticmethod
    def analyze_genome_complexity(genome: FashionGenome) -> Dict:
        """Analyze the complexity metrics of a genome"""
        node_count = len(genome.nodes)
        connection_count = len([c for c in genome.connections if c['enabled']])
        
        # Activation function diversity
        activations = [node['activation'] for node in genome.nodes]
        activation_diversity = len(set(activations))
        
        # Network depth (approximate)
        depth = FashionAnalyzer._estimate_network_depth(genome)
        
        # Fashion-specific activation usage
        fashion_activations = ['fabric_drape', 'golden_ratio', 'asymmetry', 'texture_wave', 'silhouette']
        fashion_usage = sum(1 for activation in activations if activation in fashion_activations)
        
        return {
            'node_count': node_count,
            'connection_count': connection_count,
            'activation_diversity': activation_diversity,
            'estimated_depth': depth,
            'fashion_activation_usage': fashion_usage,
            'complexity_score': node_count + connection_count + activation_diversity
        }
    
    @staticmethod
    def _estimate_network_depth(genome: FashionGenome) -> int:
        """Estimate the depth of the network"""
        # Simple estimation based on connection patterns
        # In a real implementation, this would do proper topological analysis
        connections = genome.connections
        if not connections:
            return 1
        
        # Count maximum chain length (simplified)
        node_connections = {}
        for conn in connections:
            if conn['enabled']:
                if conn['target'] not in node_connections:
                    node_connections[conn['target']] = []
                node_connections[conn['target']].append(conn['source'])
        
        # Rough depth estimation
        return min(len(node_connections), 5) + 1
    
    @staticmethod
    def compare_genomes(genome1: FashionGenome, genome2: FashionGenome) -> Dict:
        """Compare two genomes for similarity"""
        # Structural similarity
        nodes1 = set(node['id'] for node in genome1.nodes)
        nodes2 = set(node['id'] for node in genome2.nodes)
        node_overlap = len(nodes1 & nodes2) / max(len(nodes1), len(nodes2))
        
        # Connection similarity
        conns1 = set((c['source'], c['target']) for c in genome1.connections if c['enabled'])
        conns2 = set((c['source'], c['target']) for c in genome2.connections if c['enabled'])
        conn_overlap = len(conns1 & conns2) / max(len(conns1), len(conns2)) if max(len(conns1), len(conns2)) > 0 else 0
        
        # Activation function similarity
        acts1 = [node['activation'] for node in genome1.nodes]
        acts2 = [node['activation'] for node in genome2.nodes]
        act_similarity = len(set(acts1) & set(acts2)) / len(set(acts1) | set(acts2))
        
        return {
            'node_overlap': node_overlap,
            'connection_overlap': conn_overlap,
            'activation_similarity': act_similarity,
            'overall_similarity': (node_overlap + conn_overlap + act_similarity) / 3
        }

class AdvancedFashionRenderer:
    """Advanced renderer with more sophisticated fashion representation"""
    
    def __init__(self, width: int = 512, height: int = 768):
        self.width = width
        self.height = height
    
    def render_high_quality_garment(self, cppn: FashionCPPN, garment_type: str = "jacket") -> np.ndarray:
        """Render high-quality garment with fashion-specific details"""
        # Create high-resolution coordinate grid
        x_coords = np.linspace(-1.5, 1.5, self.width)
        y_coords = np.linspace(-2, 2, self.height)
        
        # Evaluate CPPN at each point
        outputs = self._evaluate_cppn_field(cppn, x_coords, y_coords)
        
        # Create garment-specific silhouette
        if garment_type == "jacket":
            mask = self._create_jacket_silhouette(outputs, x_coords, y_coords)
        elif garment_type == "dress":
            mask = self._create_dress_silhouette(outputs, x_coords, y_coords)
        else:
            mask = self._create_generic_silhouette(outputs, x_coords, y_coords)
        
        # Apply advanced rendering techniques
        image = self._render_with_materials(mask, outputs)
        
        # Add fashion details
        image = self._add_construction_details(image, mask, outputs)
        
        return image
    
    def _evaluate_cppn_field(self, cppn: FashionCPPN, x_coords: np.ndarray, y_coords: np.ndarray) -> Dict[str, np.ndarray]:
        """Evaluate CPPN across entire coordinate field"""
        outputs = {
            'silhouette': np.zeros((len(y_coords), len(x_coords))),
            'texture': np.zeros((len(y_coords), len(x_coords))),
            'color_h': np.zeros((len(y_coords), len(x_coords))),
            'color_s': np.zeros((len(y_coords), len(x_coords))),
            'color_l': np.zeros((len(y_coords), len(x_coords))),
            'asymmetry': np.zeros((len(y_coords), len(x_coords))),
            'structure': np.zeros((len(y_coords), len(x_coords))),
        }
        
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                result = cppn.evaluate(x, y, 0.0)
                for key in outputs:
                    outputs[key][i, j] = result.get(key, 0.0)
        
        return outputs
    
    def _create_jacket_silhouette(self, outputs: Dict[str, np.ndarray], 
                                 x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Create realistic jacket silhouette"""
        height, width = outputs['silhouette'].shape
        y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Basic jacket proportions
        shoulder_line = -0.8
        waist_line = 0.2
        hem_line = 1.2
        
        # Shoulder width varies with height
        shoulder_width = 0.8 * (1 - 0.3 * np.maximum(0, (y_grid - shoulder_line) / 0.5))
        
        # Waist suppression
        waist_factor = 1 - 0.2 * np.exp(-((y_grid - waist_line) / 0.3) ** 2)
        
        # Create base silhouette
        base_mask = (np.abs(x_grid) < shoulder_width * waist_factor) & (y_grid > shoulder_line) & (y_grid < hem_line)
        
        # Apply CPPN influence
        silhouette_influence = 0.4 * (outputs['silhouette'] - 0.5)
        asymmetry_influence = 0.2 * outputs['asymmetry']
        
        # Modify silhouette based on CPPN outputs
        dynamic_width = shoulder_width * (1 + silhouette_influence + asymmetry_influence)
        final_mask = (np.abs(x_grid) < dynamic_width) & (y_grid > shoulder_line) & (y_grid < hem_line)
        
        # Smooth edges
        from scipy import ndimage
        final_mask = ndimage.gaussian_filter(final_mask.astype(float), sigma=1.5) > 0.3
        
        return final_mask.astype(np.float32)
    
    def _create_dress_silhouette(self, outputs: Dict[str, np.ndarray], 
                                x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Create dress silhouette with flowing lines"""
        height, width = outputs['silhouette'].shape
        y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Dress proportions
        shoulder_line = -1.0
        waist_line = 0.0
        hem_line = 1.6
        
        # A-line shape
        top_width = 0.6
        hem_width = 1.2
        
        # Calculate width at each height
        height_ratio = (y_grid - shoulder_line) / (hem_line - shoulder_line)
        dress_width = top_width + (hem_width - top_width) * height_ratio ** 0.7
        
        # Apply fabric drape influence
        drape_influence = 0.3 * outputs['structure'] * height_ratio
        final_width = dress_width * (1 + drape_influence)
        
        # Create mask
        base_mask = (np.abs(x_grid) < final_width) & (y_grid > shoulder_line) & (y_grid < hem_line)
        
        return base_mask.astype(np.float32)
    
    def _create_generic_silhouette(self, outputs: Dict[str, np.ndarray], 
                                  x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Generic garment silhouette"""
        return (outputs['silhouette'] > 0.3).astype(np.float32)
    
    def _render_with_materials(self, mask: np.ndarray, outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Render with material properties"""
        height, width = mask.shape
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Normalize color outputs
        hue = self._normalize_to_range(outputs['color_h'], 0, 360)
        saturation = np.clip(self._normalize_to_range(outputs['color_s'], 0, 1), 0, 1)
        lightness = np.clip(self._normalize_to_range(outputs['color_l'], 0.2, 0.8), 0.1, 0.9)
        
        # Add texture influence
        texture_factor = 0.15 * self._normalize_to_range(outputs['texture'], -1, 1)
        lightness = np.clip(lightness + texture_factor, 0.1, 0.9)
        
        # Render only where garment exists
        for i in range(height):
            for j in range(width):
                if mask[i, j] > 0.1:
                    h = hue[i, j] / 360.0
                    s = saturation[i, j] * mask[i, j]  # Fade at edges
                    l = lightness[i, j]
                    
                    rgb = self._hsl_to_rgb(h, s, l)
                    image[i, j] = [int(c * 255) for c in rgb]
                else:
                    # Background
                    image[i, j] = [248, 249, 250]
        
        return image
    
    def _add_construction_details(self, image: np.ndarray, mask: np.ndarray, 
                                 outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Add fashion construction details like seams, buttons, etc."""
        height, width = image.shape[:2]
        
        # Add subtle seam lines based on structure output
        structure = outputs['structure']
        seam_threshold = 0.7
        
        # Vertical seams
        for j in range(1, width-1):
            seam_strength = np.abs(structure[:, j] - structure[:, j-1])
            seam_positions = (seam_strength > seam_threshold) & (mask[:, j] > 0.5)
            
            for i in range(height):
                if seam_positions[i]:
                    # Darken seam area
                    image[i, j] = np.clip(image[i, j] * 0.8, 0, 255).astype(np.uint8)
        
        # Add buttons based on asymmetry patterns
        asymmetry = outputs['asymmetry']
        button_candidates = []
        
        center_col = width // 2
        for i in range(height // 4, 3 * height // 4, height // 12):
            if mask[i, center_col] > 0.5 and asymmetry[i, center_col] > 0.3:
                button_candidates.append((i, center_col + 15))
        
        # Draw buttons
        for button_pos in button_candidates[:4]:  # Max 4 buttons
            y, x = button_pos
            if 0 <= y < height and 0 <= x < width:
                # Simple button representation
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if 0 <= y+dy < height and 0 <= x+dx < width:
                            if dy*dy + dx*dx <= 4:  # Circle
                                image[y+dy, x+dx] = [60, 60, 60]  # Dark button
        
        return image
    
    def _normalize_to_range(self, arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Normalize array to specific range"""
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            normalized = (arr - arr_min) / (arr_max - arr_min)
            return normalized * (max_val - min_val) + min_val
        return np.full_like(arr, (min_val + max_val) / 2)
    
    def _hsl_to_rgb(self, h: float, s: float, l: float) -> Tuple[float, float, float]:
        """Convert HSL to RGB"""
        def hue_to_rgb(p: float, q: float, t: float) -> float:
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
        
        if s == 0:
            r = g = b = l
        else:
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)
        
        return (r, g, b)

def advanced_fashion_demo():
    """Demonstrate advanced fashion evolution"""
    print("🔬 Advanced Fashion CPPN Demo")
    print("=" * 40)
    
    # Create a sophisticated genome
    genome = FashionGenomeFactory.create_minimal_genome()
    
    # Add more complexity
    for _ in range(3):
        genome = FashionGenomeFactory.mutate_genome(genome, mutation_rate=0.3)
    
    # Analyze the genome
    analyzer = FashionAnalyzer()
    complexity = analyzer.analyze_genome_complexity(genome)
    
    print("Genome Analysis:")
    for key, value in complexity.items():
        print(f"  {key}: {value}")
    
    # Render with advanced renderer
    print("\nRendering high-quality garments...")
    renderer = AdvancedFashionRenderer()
    cppn = FashionCPPN(genome)
    
    # Render different garment types
    jacket_image = renderer.render_high_quality_garment(cppn, "jacket")
    dress_image = renderer.render_high_quality_garment(cppn, "dress")
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    
    axes[0].imshow(jacket_image)
    axes[0].set_title("Neural Network Jacket Design", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(dress_image)
    axes[1].set_title("Neural Network Dress Design", fontsize=14)
    axes[1].axis('off')
    
    plt.suptitle("CPPN-Generated Fashion Designs", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('advanced_fashion_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✨ Advanced demo complete!")

if __name__ == "__main__":
    print("🎨 Fashion CPPN Evolution System")
    print("Choose demo mode:")
    print("1. Interactive Evolution (full system)")
    print("2. Advanced Rendering Demo")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_interactive_evolution()
    elif choice == "2":
        advanced_fashion_demo()
    else:
        print("Running advanced demo by default...")
        advanced_fashion_demo()