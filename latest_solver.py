import numpy as np

class Optimizer:
    """Differential Evolution optimizer for 10D continuous black-box optimization.
    
    Strategy:
    1. Phase 1: Latin Hypercube sampling for initial population (12 points)
    2. Phase 2: DE/best/1/bin with dither and jitter for exploitation
    3. Greedy selection: replace worst population member if offspring is better
    
    Handles noisy observations by tracking best observed value.
    """
    
    def __init__(self, dim, lower, upper, budget, seed, rng):
        self.dim = dim
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.budget = budget
        self.seed = seed
        self.rng = rng
        
        # DE parameters
        self.pop_size = min(12, budget // 10)
        self.F = 0.7
        self.CR = 0.8
        
        self.best_x = None
        self.best_y = np.inf
        self.query_count = 0
        
        self.population = None
        self.fitness = None
        self.pop_filled = False
        
        self.X_list = []
        self.y_list = []
    
    def _latin_hypercube(self, n):
        points = np.zeros((n, self.dim))
        for d in range(self.dim):
            perm = self.rng.permutation(n)
            points[:, d] = (perm + self.rng.uniform(0, 1, n)) / n
        points = self.lower + points * (self.upper - self.lower)
        return points
    
    def ask(self, n):
        n = min(n, self.budget - self.query_count)
        if n <= 0:
            return np.empty((0, self.dim))
        
        if self.population is None:
            self.population = self._latin_hypercube(self.pop_size)
            self.fitness = np.full(self.pop_size, np.inf)
        
        points = []
        remaining = n
        
        while remaining > 0:
            if not self.pop_filled:
                idx = self.query_count
                batch_size = min(remaining, self.pop_size - idx)
                batch = self.population[idx:idx + batch_size]
                points.append(batch)
                self.query_count += batch_size
                remaining -= batch_size
                if self.query_count >= self.pop_size:
                    self.pop_filled = True
            else:
                batch_size = remaining
                offspring = []
                
                best_idx = int(np.argmin(self.fitness))
                best = self.population[best_idx]
                
                for _ in range(batch_size):
                    idxs = self.rng.choice(self.pop_size, 3, replace=False)
                    a, b, c = self.population[idxs]
                    
                    F = self.F * (0.5 + self.rng.uniform(0, 1) * 0.5)
                    mutant = a + F * (b - c)
                    
                    CR = self.CR * (0.8 + self.rng.uniform(0, 1) * 0.2)
                    trial = best.copy()
                    cross_points = self.rng.uniform(0, 1, self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[self.rng.integers(self.dim)] = True
                    trial[cross_points] = mutant[cross_points]
                    
                    trial = np.clip(trial, self.lower, self.upper)
                    offspring.append(trial)
                
                batch = np.array(offspring)
                points.append(batch)
                self.query_count += batch_size
                remaining -= batch_size
        
        if len(points) == 1:
            result = points[0]
        else:
            result = np.vstack(points)
        
        return np.asarray(result, dtype=np.float64)
    
    def tell(self, X, y, metadata=None):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).flatten()
        
        for i in range(len(y)):
            self.X_list.append(X[i])
            self.y_list.append(y[i])
            
            if y[i] < self.best_y:
                self.best_y = y[i]
                self.best_x = X[i].copy()
            
            if not self.pop_filled:
                idx = len(self.X_list) - 1
                if idx < self.pop_size:
                    self.fitness[idx] = y[i]
            else:
                worst_idx = int(np.argmax(self.fitness))
                if y[i] < self.fitness[worst_idx]:
                    self.fitness[worst_idx] = y[i]
                    self.population[worst_idx] = X[i].copy()
