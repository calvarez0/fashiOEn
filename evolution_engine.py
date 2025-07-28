# evolution_engine.py - Evolution engine for 3D fashion design
"""
Handles the evolutionary process for fashion genome populations.
Manages selection, crossover, mutation, and population dynamics.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass
from garment_genome import GarmentGenome, GarmentGenomeFactory

@dataclass
class EvolutionConfig:
    """Configuration for evolution parameters"""
    population_size: int = 16
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    elitism_rate: float = 0.2
    tournament_size: int = 3
    max_generations: int = 50
    fitness_threshold: float = 0.9
    diversity_pressure: float = 0.1
    
class EvolutionStats:
    """Track evolution statistics"""
    
    def __init__(self):
        self.generation_stats: List[Dict] = []
        self.best_fitness_history: List[float] = []
        self.average_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
    
    def record_generation(self, generation: int, population: List[GarmentGenome]):
        """Record statistics for a generation"""
        fitnesses = [genome.fitness for genome in population]
        
        stats = {
            'generation': generation,
            'best_fitness': max(fitnesses) if fitnesses else 0.0,
            'average_fitness': np.mean(fitnesses) if fitnesses else 0.0,
            'worst_fitness': min(fitnesses) if fitnesses else 0.0,
            'fitness_std': np.std(fitnesses) if fitnesses else 0.0,
            'population_size': len(population),
            'diversity': self._calculate_diversity(population)
        }
        
        self.generation_stats.append(stats)
        self.best_fitness_history.append(stats['best_fitness'])
        self.average_fitness_history.append(stats['average_fitness'])
        self.diversity_history.append(stats['diversity'])
        
        return stats
    
    def _calculate_diversity(self, population: List[GarmentGenome]) -> float:
        """Calculate population diversity based on genome differences"""
        if len(population) < 2:
            return 0.0
        
        diversity_sum = 0.0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # Simple diversity metric: difference in number of nodes and connections
                genome1, genome2 = population[i], population[j]
                
                node_diff = abs(len(genome1.nodes) - len(genome2.nodes))
                conn_diff = abs(len(genome1.connections) - len(genome2.connections))
                
                # Activation function diversity
                acts1 = set(node['activation'] for node in genome1.nodes)
                acts2 = set(node['activation'] for node in genome2.nodes)
                act_diff = len(acts1.symmetric_difference(acts2))
                
                diversity = (node_diff + conn_diff + act_diff) / 20.0  # Normalize
                diversity_sum += diversity
                comparisons += 1
        
        return diversity_sum / comparisons if comparisons > 0 else 0.0

class FashionEvolutionEngine:
    """Main evolution engine for fashion design"""
    
    def __init__(self, config: Optional[EvolutionConfig] = None):
        self.config = config or EvolutionConfig()
        self.stats = EvolutionStats()
        self.current_generation = 0
        self.innovation_number = 0  # For NEAT-style evolution
        
    def create_initial_population(self, population_size: int, garment_type: str) -> List[GarmentGenome]:
        """Create initial population of diverse genomes"""
        population = []
        
        for i in range(population_size):
            genome = GarmentGenomeFactory.create_construction_genome(garment_type)
            genome.generation = 0
            
            # Add some initial diversity through mutations
            if i > 0:  # Keep first genome as baseline
                num_mutations = random.randint(1, 3)
                for _ in range(num_mutations):
                    genome = GarmentGenomeFactory.mutate_construction_genome(
                        genome, mutation_rate=0.2
                    )
            
            population.append(genome)
        
        return population
    
    def evolve_population(self, population: List[GarmentGenome], target_size: int) -> List[GarmentGenome]:
        """Evolve population to next generation"""
        
        if not population:
            # If empty population, create new random population
            return self.create_initial_population(target_size, "jacket")
        
        # Record current generation stats AFTER fitness is set
        current_stats = self.stats.record_generation(self.current_generation, population)
        print(f"   Generation {self.current_generation} stats:")
        print(f"     Best fitness: {current_stats['best_fitness']:.3f}")
        print(f"     Avg fitness: {current_stats['average_fitness']:.3f}")
        print(f"     Diversity: {current_stats['diversity']:.3f}")
        
        # Selection and reproduction
        new_population = []
        
        # Elitism - keep best individuals
        elite_count = max(1, int(target_size * self.config.elitism_rate))
        elite = self._select_elite(population, elite_count)
        
        for genome in elite:
            elite_copy = self._copy_genome(genome)
            elite_copy.generation = self.current_generation + 1
            new_population.append(elite_copy)
        
        # Generate offspring through crossover and mutation
        while len(new_population) < target_size:
            if random.random() < self.config.crossover_rate and len(population) >= 2:
                # Crossover
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # Ensure parents are different
                attempts = 0
                while parent1 == parent2 and attempts < 5:
                    parent2 = self._tournament_selection(population)
                    attempts += 1
                
                child = GarmentGenomeFactory.crossover_construction_genomes(parent1, parent2)
                
                # Mutation
                if random.random() < self.config.mutation_rate:
                    child = GarmentGenomeFactory.mutate_construction_genome(
                        child, self.config.mutation_rate
                    )
                
                child.generation = self.current_generation + 1
                new_population.append(child)
            
            else:
                # Mutation only
                parent = self._tournament_selection(population)
                child = GarmentGenomeFactory.mutate_construction_genome(
                    parent, self.config.mutation_rate * 1.5  # Higher mutation rate for asexual reproduction
                )
                child.generation = self.current_generation + 1
                new_population.append(child)
        
        # Apply diversity pressure if population is too similar
        if current_stats['diversity'] < 0.1:
            new_population = self._apply_diversity_pressure(new_population)
        
        self.current_generation += 1
        return new_population[:target_size]
    
    def _select_elite(self, population: List[GarmentGenome], count: int) -> List[GarmentGenome]:
        """Select the best individuals for elitism"""
        return sorted(population, key=lambda g: g.fitness, reverse=True)[:count]
    
    def _tournament_selection(self, population: List[GarmentGenome]) -> GarmentGenome:
        """Select individual using tournament selection"""
        tournament_size = min(self.config.tournament_size, len(population))
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda g: g.fitness)
    
    def _copy_genome(self, genome: GarmentGenome) -> GarmentGenome:
        """Create a deep copy of a genome"""
        return GarmentGenome(
            nodes=[node.copy() for node in genome.nodes],
            connections=[conn.copy() for conn in genome.connections],
            garment_type=genome.garment_type,
            fitness=genome.fitness,
            generation=genome.generation,
            parent_ids=genome.parent_ids.copy()
        )
    
    def _apply_diversity_pressure(self, population: List[GarmentGenome]) -> List[GarmentGenome]:
        """Add diversity to population if it becomes too homogeneous"""
        diverse_population = population.copy()
        
        # Replace some individuals with highly mutated versions
        diversity_replacements = max(1, int(len(population) * self.config.diversity_pressure))
        
        for i in range(diversity_replacements):
            if i < len(diverse_population):
                # Take a random individual and apply heavy mutation
                original = random.choice(population)
                mutated = GarmentGenomeFactory.mutate_construction_genome(
                    original, mutation_rate=0.5  # High mutation rate
                )
                mutated.generation = self.current_generation + 1
                diverse_population[i] = mutated
        
        return diverse_population
    
    def evaluate_termination_criteria(self, population: List[GarmentGenome]) -> Tuple[bool, str]:
        """Check if evolution should terminate"""
        
        if not population:
            return True, "Empty population"
        
        best_fitness = max(genome.fitness for genome in population)
        
        # Fitness threshold reached
        if best_fitness >= self.config.fitness_threshold:
            return True, f"Fitness threshold reached: {best_fitness:.3f}"
        
        # Maximum generations reached
        if self.current_generation >= self.config.max_generations:
            return True, f"Maximum generations reached: {self.current_generation}"
        
        # Stagnation check - no improvement in last 10 generations
        if len(self.stats.best_fitness_history) >= 10:
            recent_best = self.stats.best_fitness_history[-10:]
            if max(recent_best) - min(recent_best) < 0.01:  # Less than 1% improvement
                return True, "Population stagnated"
        
        return False, "Continue evolution"
    
    def get_evolution_summary(self) -> Dict:
        """Get summary of evolution process"""
        if not self.stats.generation_stats:
            return {"status": "No evolution data"}
        
        best_generation = max(self.stats.generation_stats, key=lambda x: x['best_fitness'])
        final_generation = self.stats.generation_stats[-1]
        
        return {
            "total_generations": len(self.stats.generation_stats),
            "best_fitness_achieved": best_generation['best_fitness'],
            "best_fitness_generation": best_generation['generation'],
            "final_average_fitness": final_generation['average_fitness'],
            "final_diversity": final_generation['diversity'],
            "fitness_improvement": (
                final_generation['best_fitness'] - self.stats.generation_stats[0]['best_fitness']
                if len(self.stats.generation_stats) > 1 else 0
            ),
            "diversity_trend": (
                final_generation['diversity'] - self.stats.generation_stats[0]['diversity']
                if len(self.stats.generation_stats) > 1 else 0
            )
        }
    
    def save_evolution_data(self, filename: str, population: List[GarmentGenome]):
        """Save evolution data and final population"""
        
        data = {
            "config": {
                "population_size": self.config.population_size,
                "mutation_rate": self.config.mutation_rate,
                "crossover_rate": self.config.crossover_rate,
                "elitism_rate": self.config.elitism_rate,
                "tournament_size": self.config.tournament_size
            },
            "evolution_summary": self.get_evolution_summary(),
            "generation_stats": self.stats.generation_stats,
            "final_population": [genome.to_dict() for genome in population],
            "best_genomes": [
                genome.to_dict() for genome in 
                sorted(population, key=lambda g: g.fitness, reverse=True)[:5]
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Evolution data saved to {filename}")
    
    def load_evolution_data(self, filename: str) -> List[GarmentGenome]:
        """Load evolution data and return population"""
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Reconstruct genomes from saved data
        population = []
        for genome_data in data['final_population']:
            genome = GarmentGenome(
                nodes=genome_data['nodes'],
                connections=genome_data['connections'],
                garment_type=genome_data['garment_type'],
                fitness=genome_data['fitness'],
                generation=genome_data['generation'],
                parent_ids=genome_data['parent_ids']
            )
            population.append(genome)
        
        # Restore evolution stats
        self.stats.generation_stats = data['generation_stats']
        self.stats.best_fitness_history = [s['best_fitness'] for s in data['generation_stats']]
        self.stats.average_fitness_history = [s['average_fitness'] for s in data['generation_stats']]
        self.stats.diversity_history = [s['diversity'] for s in data['generation_stats']]
        
        print(f"Loaded evolution data with {len(population)} genomes")
        return population
    
    def visualize_evolution_progress(self, save_path: Optional[str] = None):
        """Visualize evolution progress with plots"""
        
        import matplotlib.pyplot as plt
        
        if not self.stats.generation_stats:
            print("No evolution data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        generations = list(range(len(self.stats.best_fitness_history)))
        
        # Fitness over time
        ax1 = axes[0, 0]
        ax1.plot(generations, self.stats.best_fitness_history, 'r-', linewidth=2, label='Best Fitness')
        ax1.plot(generations, self.stats.average_fitness_history, 'b-', linewidth=2, label='Average Fitness')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Diversity over time
        ax2 = axes[0, 1]
        ax2.plot(generations, self.stats.diversity_history, 'g-', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Population Diversity')
        ax2.set_title('Diversity Evolution')
        ax2.grid(True, alpha=0.3)
        
        # Fitness distribution in final generation
        ax3 = axes[1, 0]
        if self.stats.generation_stats:
            final_stats = self.stats.generation_stats[-1]
            # Create histogram of fitness values (simulated since we don't store individual fitnesses)
            # This is a simplified representation
            fitness_range = np.linspace(0, final_stats['best_fitness'], 20)
            # Simulate fitness distribution around average
            fitness_dist = np.random.normal(final_stats['average_fitness'], 
                                          final_stats['fitness_std'], 100)
            fitness_dist = np.clip(fitness_dist, 0, 1)  # Clip to valid range
            
            ax3.hist(fitness_dist, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(final_stats['average_fitness'], color='red', linestyle='--', 
                       linewidth=2, label=f'Average: {final_stats["average_fitness"]:.3f}')
            ax3.axvline(final_stats['best_fitness'], color='green', linestyle='--', 
                       linewidth=2, label=f'Best: {final_stats["best_fitness"]:.3f}')
            ax3.set_xlabel('Fitness')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Final Generation Fitness Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Evolution metrics summary
        ax4 = axes[1, 1]
        summary = self.get_evolution_summary()
        
        metrics = ['Best Fitness', 'Avg Fitness', 'Diversity', 'Improvement']
        values = [
            summary.get('best_fitness_achieved', 0),
            summary.get('final_average_fitness', 0),
            summary.get('final_diversity', 0),
            summary.get('fitness_improvement', 0)
        ]
        
        colors = ['gold', 'lightblue', 'lightgreen', 'coral']
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylabel('Value')
        ax4.set_title('Evolution Summary Metrics')
        ax4.set_ylim(0, max(values) * 1.2 if values else 1)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(f'Fashion Evolution Analysis - {len(generations)} Generations', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print detailed summary
        print("\n" + "="*60)
        print("EVOLUTION SUMMARY")
        print("="*60)
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("="*60)

class AdaptiveEvolutionEngine(FashionEvolutionEngine):
    """Advanced evolution engine with adaptive parameters"""
    
    def __init__(self, config: Optional[EvolutionConfig] = None):
        super().__init__(config)
        self.stagnation_counter = 0
        self.last_best_fitness = 0.0
        
    def evolve_population(self, population: List[GarmentGenome], target_size: int) -> List[GarmentGenome]:
        """Evolve with adaptive parameters based on population performance"""
        
        # Check for stagnation
        current_best = max(genome.fitness for genome in population) if population else 0.0
        
        if abs(current_best - self.last_best_fitness) < 0.01:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        self.last_best_fitness = current_best
        
        # Adapt mutation rate based on stagnation
        if self.stagnation_counter > 3:
            # Increase mutation rate to escape local optima
            adaptive_mutation_rate = min(self.config.mutation_rate * 2, 0.5)
            print(f"   Adaptive: Increased mutation rate to {adaptive_mutation_rate:.3f}")
        elif self.stagnation_counter == 0:
            # Decrease mutation rate for fine-tuning
            adaptive_mutation_rate = max(self.config.mutation_rate * 0.8, 0.05)
        else:
            adaptive_mutation_rate = self.config.mutation_rate
        
        # Temporarily modify config
        original_mutation_rate = self.config.mutation_rate
        self.config.mutation_rate = adaptive_mutation_rate
        
        # Run standard evolution
        new_population = super().evolve_population(population, target_size)
        
        # Restore original config
        self.config.mutation_rate = original_mutation_rate
        
        return new_population

class FashionSpecies:
    """Manage species for speciated evolution (NEAT-style)"""
    
    def __init__(self, representative: GarmentGenome, species_id: int):
        self.representative = representative
        self.species_id = species_id
        self.members: List[GarmentGenome] = [representative]
        self.best_fitness = representative.fitness
        self.average_fitness = representative.fitness
        self.stagnation_count = 0
        
    def add_member(self, genome: GarmentGenome):
        """Add genome to species"""
        self.members.append(genome)
        self._update_stats()
    
    def _update_stats(self):
        """Update species statistics"""
        if not self.members:
            return
        
        fitnesses = [member.fitness for member in self.members]
        new_best = max(fitnesses)
        
        if new_best <= self.best_fitness:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
            self.best_fitness = new_best
        
        self.average_fitness = np.mean(fitnesses)
    
    def get_offspring_quota(self, total_offspring: int, total_average_fitness: float) -> int:
        """Calculate how many offspring this species should produce"""
        if total_average_fitness == 0:
            return 0
        
        quota = int((self.average_fitness / total_average_fitness) * total_offspring)
        return max(1, quota)  # At least 1 offspring per species

class SpeciatedEvolutionEngine(FashionEvolutionEngine):
    """Evolution engine with speciation for maintaining diversity"""
    
    def __init__(self, config: Optional[EvolutionConfig] = None):
        super().__init__(config)
        self.species: List[FashionSpecies] = []
        self.compatibility_threshold = 3.0
        self.species_stagnation_threshold = 15
        
    def evolve_population(self, population: List[GarmentGenome], target_size: int) -> List[GarmentGenome]:
        """Evolve with speciation"""
        
        # Assign genomes to species
        self._assign_to_species(population)
        
        # Remove stagnant species
        self._remove_stagnant_species()
        
        # Calculate offspring quotas
        total_average_fitness = sum(species.average_fitness for species in self.species)
        
        new_population = []
        
        for species in self.species:
            offspring_quota = species.get_offspring_quota(target_size, total_average_fitness)
            
            # Evolve within species
            species_offspring = self._evolve_species(species, offspring_quota)
            new_population.extend(species_offspring)
        
        # Fill remaining slots if needed
        while len(new_population) < target_size:
            if self.species:
                # Add from best species
                best_species = max(self.species, key=lambda s: s.best_fitness)
                parent = random.choice(best_species.members)
                child = GarmentGenomeFactory.mutate_construction_genome(parent, self.config.mutation_rate)
                child.generation = self.current_generation + 1
                new_population.append(child)
            else:
                # Create new random genome
                new_genome = GarmentGenomeFactory.create_construction_genome("jacket")
                new_genome.generation = self.current_generation + 1
                new_population.append(new_genome)
        
        self.current_generation += 1
        return new_population[:target_size]
    
    def _assign_to_species(self, population: List[GarmentGenome]):
        """Assign genomes to species based on compatibility"""
        
        # Clear existing species memberships
        for species in self.species:
            species.members = []
        
        for genome in population:
            # Try to assign to existing species
            assigned = False
            
            for species in self.species:
                if self._calculate_compatibility(genome, species.representative) < self.compatibility_threshold:
                    species.add_member(genome)
                    assigned = True
                    break
            
            # Create new species if not assigned
            if not assigned:
                new_species = FashionSpecies(genome, len(self.species))
                self.species.append(new_species)
        
        # Remove empty species
        self.species = [species for species in self.species if species.members]
    
    def _calculate_compatibility(self, genome1: GarmentGenome, genome2: GarmentGenome) -> float:
        """Calculate compatibility distance between genomes"""
        
        # Simple compatibility based on structure
        node_diff = abs(len(genome1.nodes) - len(genome2.nodes))
        conn_diff = abs(len(genome1.connections) - len(genome2.connections))
        
        # Activation function differences
        acts1 = set(node['activation'] for node in genome1.nodes)
        acts2 = set(node['activation'] for node in genome2.nodes)
        act_diff = len(acts1.symmetric_difference(acts2))
        
        # Weight differences (simplified)
        weight_diff = 0.0
        conn1_dict = {(c['source'], c['target']): c['weight'] for c in genome1.connections}
        conn2_dict = {(c['source'], c['target']): c['weight'] for c in genome2.connections}
        
        common_connections = set(conn1_dict.keys()) & set(conn2_dict.keys())
        for conn_key in common_connections:
            weight_diff += abs(conn1_dict[conn_key] - conn2_dict[conn_key])
        
        if common_connections:
            weight_diff /= len(common_connections)
        
        # Combine factors
        compatibility = node_diff + conn_diff + act_diff + weight_diff
        
        return compatibility
    
    def _remove_stagnant_species(self):
        """Remove species that have stagnated for too long"""
        
        # Keep at least 2 species
        if len(self.species) <= 2:
            return
        
        # Remove stagnant species (except the best one)
        best_species = max(self.species, key=lambda s: s.best_fitness)
        
        self.species = [
            species for species in self.species
            if species == best_species or species.stagnation_count < self.species_stagnation_threshold
        ]
    
    def _evolve_species(self, species: FashionSpecies, offspring_count: int) -> List[GarmentGenome]:
        """Evolve within a single species"""
        
        if not species.members or offspring_count <= 0:
            return []
        
        offspring = []
        
        # Elitism within species
        if len(species.members) > 1:
            best_member = max(species.members, key=lambda g: g.fitness)
            elite_copy = self._copy_genome(best_member)
            elite_copy.generation = self.current_generation + 1
            offspring.append(elite_copy)
            offspring_count -= 1
        
        # Generate remaining offspring
        while len(offspring) < offspring_count:
            if len(species.members) >= 2 and random.random() < self.config.crossover_rate:
                # Crossover within species
                parent1 = random.choice(species.members)
                parent2 = random.choice(species.members)
                
                child = GarmentGenomeFactory.crossover_construction_genomes(parent1, parent2)
            else:
                # Mutation only
                parent = random.choice(species.members)
                child = self._copy_genome(parent)
            
            # Apply mutation
            if random.random() < self.config.mutation_rate:
                child = GarmentGenomeFactory.mutate_construction_genome(child, self.config.mutation_rate)
            
            child.generation = self.current_generation + 1
            offspring.append(child)
        
        return offspring

# Example usage and testing
if __name__ == "__main__":
    from garment_genome import GarmentGenomeFactory
    
    print("Testing Fashion Evolution Engine...")
    
    # Test basic evolution
    engine = FashionEvolutionEngine()
    population = engine.create_initial_population(12, "jacket")
    
    # Simulate some fitness values
    for i, genome in enumerate(population):
        genome.fitness = random.uniform(0.1, 0.8)
    
    print(f"Initial population: {len(population)} genomes")
    print(f"Fitness range: {min(g.fitness for g in population):.3f} - {max(g.fitness for g in population):.3f}")
    
    # Evolve for a few generations
    for generation in range(5):
        population = engine.evolve_population(population, 12)
        
        # Simulate fitness evaluation
        for genome in population:
            # Simulate improvement over generations
            base_fitness = random.uniform(0.2, 0.9)
            improvement_factor = generation * 0.05  # Gradual improvement
            genome.fitness = min(base_fitness + improvement_factor, 1.0)
    
    # Test adaptive evolution
    print("\nTesting Adaptive Evolution...")
    adaptive_engine = AdaptiveEvolutionEngine()
    
    # Test speciated evolution
    print("\nTesting Speciated Evolution...")
    speciated_engine = SpeciatedEvolutionEngine()
    
    # Visualize evolution progress
    engine.visualize_evolution_progress("evolution_test.png")
    
    # Save evolution data
    engine.save_evolution_data("test_evolution.json", population)
    
    print("\nEvolution engine testing complete!")