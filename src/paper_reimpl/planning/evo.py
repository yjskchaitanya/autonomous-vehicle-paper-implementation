from __future__ import annotations

import numpy as np


class EnergyValleyOptimizer:
    def __init__(self, population: int = 64, iterations: int = 15, threshold_scale: float = 1.0) -> None:
        self.population = population
        self.iterations = iterations
        self.threshold_scale = threshold_scale

    def optimize(self, quality_map: np.ndarray) -> dict[str, np.ndarray | float | tuple[int, int]]:
        flat = quality_map.astype(np.float32).reshape(-1)
        if flat.size == 0:
            raise ValueError("quality_map cannot be empty")

        population = np.random.uniform(0, flat.size - 1, size=(self.population,)).astype(np.float32)
        enrichment_threshold = flat.mean() * self.threshold_scale
        quality_levels = self._normalize(flat)
        best_idx = int(np.argmax(flat))
        best_quality = float(flat[best_idx])

        for _ in range(self.iterations):
            center = float(population.mean())
            for j in range(self.population):
                idx = int(np.clip(round(population[j]), 0, flat.size - 1))
                iq = float(flat[idx])
                ql = float(quality_levels[idx])
                quality_threshold = float(np.random.rand())

                if iq > enrichment_threshold:
                    d1, d2, d3, d4 = np.random.rand(4)
                    if ql > quality_threshold:
                        population[j] = population[j] + ((d1 * best_idx) - (d2 * center)) / max(ql, 1e-6)
                    else:
                        neighbor = population[np.random.randint(0, self.population)]
                        population[j] = population[j] + (d3 * best_idx) - (d4 * neighbor)
                else:
                    population[j] = population[j] + np.random.rand()

                population[j] = float(np.clip(population[j], 0, flat.size - 1))
                candidate_idx = int(round(population[j]))
                if flat[candidate_idx] > best_quality:
                    best_quality = float(flat[candidate_idx])
                    best_idx = candidate_idx

        row, col = np.unravel_index(best_idx, quality_map.shape)
        return {
            "best_score": best_quality,
            "best_cell": (int(row), int(col)),
            "quality_levels": quality_levels.reshape(quality_map.shape),
            "population": population.copy(),
        }

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        low = values.min()
        high = values.max()
        if np.isclose(high, low):
            return np.ones_like(values)
        return (values - low) / (high - low)
