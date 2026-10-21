
import numpy as np


class GeneticAlgorithm:
    """Fuzzy-guided genetic algorithm for portfolio optimization."""

    def __init__(
        self,
        n_assets=6,
        max_weight=0.40,
        population_size=100,
        mutation_rate=0.05,
        generations=30,
        fuzzy_weight=0.2,
        random_seed=42,
    ):
        if n_assets < 1:
            raise ValueError("n_assets must be positive.")
        if n_assets * max_weight < 1.0:
            raise ValueError("Maximum weight is too low to fully invest.")
        if not 0 < max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1].")
        if population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if generations < 1:
            raise ValueError("generations must be positive.")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1.")
        if not 0 <= fuzzy_weight <= 1:
            raise ValueError("fuzzy_weight must be between 0 and 1.")

        self.n_assets = n_assets
        self.max_weight = max_weight
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.fuzzy_weight = fuzzy_weight
        self.rng = np.random.default_rng(random_seed)

    def generate_weights(self):
        """Generate a valid long-only portfolio."""
        for _ in range(10000):
            weights = self.rng.dirichlet(np.ones(self.n_assets))
            if np.all(weights <= self.max_weight):
                return weights

        raise RuntimeError("Could not generate valid portfolio weights.")

    def create_population(self):
        return np.array([
            self.generate_weights()
            for _ in range(self.population_size)
        ])

    def validate_weights(self, weights):
        weights = np.asarray(weights, dtype=float)

        return (
            weights.shape == (self.n_assets,)
            and np.all(np.isfinite(weights))
            and np.all(weights >= 0)
            and np.all(weights <= self.max_weight + 1e-9)
            and np.isclose(weights.sum(), 1.0)
        )

    def calculate_fitness(
        self,
        weights,
        fuzzy_scores,
        expected_returns,
        covariance_matrix,
    ):
        """Combine fuzzy suitability, expected return, and volatility."""
        weights = np.asarray(weights, dtype=float)
        fuzzy_scores = np.asarray(fuzzy_scores, dtype=float)
        expected_returns = np.asarray(expected_returns, dtype=float)
        covariance_matrix = np.asarray(covariance_matrix, dtype=float)

        if not self.validate_weights(weights):
            return -np.inf

        if fuzzy_scores.shape != (self.n_assets,):
            raise ValueError("fuzzy_scores must match n_assets.")
        if expected_returns.shape != (self.n_assets,):
            raise ValueError("expected_returns must match n_assets.")
        if covariance_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError("Covariance matrix has an incorrect shape.")

        if not (
            np.all(np.isfinite(fuzzy_scores))
            and np.all(np.isfinite(expected_returns))
            and np.all(np.isfinite(covariance_matrix))
        ):
            raise ValueError("Fitness inputs must be finite.")

        fuzzy_scores = np.clip(fuzzy_scores, 0, 100) / 100.0
        fuzzy_component = float(weights @ fuzzy_scores)
        expected_return = float(weights @ expected_returns)

        variance = float(weights @ covariance_matrix @ weights)
        volatility = np.sqrt(max(variance, 0.0))

        financial_component = (
            0.5
            + np.arctan(100 * (expected_return - volatility)) / np.pi
        )

        return float(
            self.fuzzy_weight * fuzzy_component
            + (1 - self.fuzzy_weight) * financial_component
        )

    def select_parent(self, population, fitness_scores):
        """Select a parent using tournament selection."""
        candidates = self.rng.choice(
            len(population),
            size=min(3, len(population)),
            replace=False,
        )
        winner = candidates[np.argmax(fitness_scores[candidates])]
        return population[winner].copy()

    def repair_weights(self, weights):
        """Project candidate weights onto the capped simplex."""
        weights = np.asarray(weights, dtype=float)

        if (
            weights.shape != (self.n_assets,)
            or not np.all(np.isfinite(weights))
        ):
            return np.full(self.n_assets, 1.0 / self.n_assets)

        lower = float(np.min(weights) - self.max_weight)
        upper = float(np.max(weights))

        for _ in range(100):
            threshold = (lower + upper) / 2
            candidate = np.clip(
                weights - threshold, 0.0, self.max_weight
            )

            if candidate.sum() > 1.0:
                lower = threshold
            else:
                upper = threshold

        repaired = np.clip(
            weights - (lower + upper) / 2,
            0.0,
            self.max_weight,
        )

        difference = 1.0 - repaired.sum()

        if abs(difference) > 1e-10:
            if difference > 0:
                available = np.where(
                    repaired < self.max_weight - 1e-10
                )[0]
            else:
                available = np.where(repaired > 1e-10)[0]

            for index in available:
                if difference > 0:
                    adjustment = min(
                        difference, self.max_weight - repaired[index]
                    )
                else:
                    adjustment = max(difference, -repaired[index])

                repaired[index] += adjustment
                difference -= adjustment

                if abs(difference) <= 1e-10:
                    break

        if not self.validate_weights(repaired):
            raise RuntimeError("Could not repair portfolio weights.")

        return repaired

    def crossover(self, parent1, parent2):
        alpha = self.rng.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        return self.repair_weights(child)

    def mutate(self, weights):
        child = weights.copy()

        for index in range(self.n_assets):
            if self.rng.random() < self.mutation_rate:
                child[index] += self.rng.normal(0, 0.05)

        return self.repair_weights(child)

    def optimize(
        self,
        fuzzy_scores,
        expected_returns,
        covariance_matrix,
    ):
        """Return the best portfolio, its fitness, and generation history."""
        population = self.create_population()
        history = []

        for generation in range(self.generations):
            fitness_scores = np.array([
                self.calculate_fitness(
                    weights,
                    fuzzy_scores,
                    expected_returns,
                    covariance_matrix,
                )
                for weights in population
            ])

            best_index = int(np.argmax(fitness_scores))
            best_weights = population[best_index].copy()

            history.append({
                "generation": generation + 1,
                "best_fitness": float(fitness_scores[best_index]),
            })

            next_population = [best_weights]

            while len(next_population) < self.population_size:
                parent1 = self.select_parent(population, fitness_scores)
                parent2 = self.select_parent(population, fitness_scores)
                child = self.mutate(self.crossover(parent1, parent2))
                next_population.append(child)

            population = np.array(next_population)

        final_fitness = np.array([
            self.calculate_fitness(
                weights,
                fuzzy_scores,
                expected_returns,
                covariance_matrix,
            )
            for weights in population
        ])

        best_index = int(np.argmax(final_fitness))

        return {
            "weights": population[best_index].copy(),
            "fitness": float(final_fitness[best_index]),
            "history": history,
        }
