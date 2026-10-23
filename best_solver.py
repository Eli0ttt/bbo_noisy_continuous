import numpy as np
import math

class Optimizer:
    """Noise-aware optimizer for continuous black-box functions.
    
    Uses Latin Hypercube Sampling for initial exploration, followed by
    adaptive local search with multiple restart strategies.
    Designed for 10D problems with budget 120 and noisy observations.
    """
    
    def __init__(self, dim, lower, upper, budget, seed, rng):
        self.dim = dim
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.budget = budget
        self.rng = rng
        self.seed = seed
        
        # Storage for evaluated points
        self.X = np.empty((0, dim))
        self.y = np.empty((0,))
        
        # Best solution tracking
        self.best_x = None
        self.best_y = np.inf
        self.n_eval = 0
        
        # Preferred batch size
        self.batch = 5
        
        # Initial design: Latin Hypercube Sampling
        self.n_init = min(20, budget // 2)
        self.phase = 'init'
        self.init_idx = 0
        self.init_points = self._latin_hypercube(self.n_init)
        
        # Domain info (scalar values for bounds)
        self.domain_span = float(np.mean(self.upper - self.lower))
        
        # Adaptive step size (scalar)
        self.step_size = 0.2 * self.domain_span
        
        # Stagnation tracking
        self.no_improve = 0
        self.last_best_y = np.inf
        
        # Population for diversity (keep best points)
        self.pop_X = np.empty((0, dim))
        self.pop_y = np.empty((0,))
        
        # Number of top points to consider for mutation
        self.n_top = min(10, self.n_init)
        
    def _latin_hypercube(self, n):
        """Generate Latin Hypercube Sampling points."""
        samples = np.zeros((n, self.dim))
        for j in range(self.dim):
            perm = self.rng.permutation(n)
            samples[:, j] = (perm + self.rng.uniform(0.0, 1.0, n)) / n
        return self.lower + (self.upper - self.lower) * samples
    
    def _get_top_indices(self, n_top):
        """Get indices of top n_top points by y value."""
        n_top = min(n_top, len(self.pop_y))
        if n_top == 0:
            return np.array([], dtype=int)
        return np.argsort(self.pop_y)[:n_top]
    
    def _generate_candidate(self):
        """Generate a single candidate point using multiple strategies."""
        progress = float(self.n_eval) / float(self.budget)
        
        # Adjust step size over time (scalar)
        radius = self.step_size * max(0.02, 1.0 - progress * 0.9)
        
        # Random strategy selection
        strategy = self.rng.uniform(0.0, 1.0)
        
        if strategy < 0.25:
            # Pure random exploration
            return self.rng.uniform(self.lower, self.upper)
        
        elif strategy < 0.50 and self.best_x is not None:
            # Local search around best point
            pt = self.best_x + self.rng.normal(0.0, radius, size=self.dim)
            return np.clip(pt, self.lower, self.upper)
        
        elif strategy < 0.70 and len(self.pop_y) >= 3:
            # Differential Evolution-style mutation
            top_idx = self._get_top_indices(self.n_top)
            if len(top_idx) >= 3:
                i, j, k = self.rng.choice(top_idx, size=3, replace=False)
                F = 0.3 + self.rng.uniform() * 0.7
                pt = self.pop_X[i] + F * (self.pop_X[j] - self.pop_X[k])
                # Crossover with best point
                if self.best_x is not None:
                    cr = 0.7
                    mask = self.rng.uniform(size=self.dim) < cr
                    pt = np.where(mask, pt, self.best_x)
                return np.clip(pt, self.lower, self.upper)
            else:
                return self.rng.uniform(self.lower, self.upper)
        
        elif strategy < 0.85 and len(self.pop_y) >= 2:
            # Crossover between two top points
            top_idx = self._get_top_indices(min(5, len(self.pop_y)))
            if len(top_idx) >= 2:
                i, j = self.rng.choice(top_idx, size=2, replace=False)
                parent1 = self.pop_X[i]
                parent2 = self.pop_X[j]
                mask = self.rng.uniform(size=self.dim) < 0.5
                pt = np.where(mask, parent1, parent2)
                pt += self.rng.normal(0.0, radius * 0.3, size=self.dim)
                return np.clip(pt, self.lower, self.upper)
            else:
                return self.rng.uniform(self.lower, self.upper)
        
        else:
            # Random perturbation of a random point from population
            if len(self.pop_y) > 0:
                idx = self.rng.integers(0, len(self.pop_y))
                pt = self.pop_X[idx] + self.rng.normal(0.0, radius, size=self.dim)
                return np.clip(pt, self.lower, self.upper)
            else:
                return self.rng.uniform(self.lower, self.upper)
    
    def ask(self, n):
        """Return n candidate points to evaluate."""
        n = int(n)
        
        if self.phase == 'init':
            available = min(n, self.n_init - self.init_idx)
            if available > 0:
                pts = self.init_points[self.init_idx:self.init_idx + available]
                self.init_idx += available
                if self.init_idx >= self.n_init:
                    self.phase = 'optimize'
                return pts.copy()
        
        # Generate n candidates using multiple strategies
        points = []
        for _ in range(n):
            points.append(self._generate_candidate())
        
        result = np.array(points)
        # Final safety clip (should already be in bounds)
        return np.clip(result, self.lower, self.upper)
    
    def tell(self, X, y, metadata=None):
        """Update optimizer with new observations."""
        X_arr = np.atleast_2d(np.asarray(X, dtype=float))
        y_arr = np.asarray(y, dtype=float).ravel()
        
        # Add to global storage
        self.X = np.vstack([self.X, X_arr])
        self.y = np.concatenate([self.y, y_arr])
        self.n_eval += len(y_arr)
        
        # Add to population
        self.pop_X = np.vstack([self.pop_X, X_arr])
        self.pop_y = np.concatenate([self.pop_y, y_arr])
        
        # Keep only best points in population (max 50)
        if len(self.pop_y) > 50:
            idx = np.argsort(self.pop_y)[:50]
            self.pop_X = self.pop_X[idx]
            self.pop_y = self.pop_y[idx]
        
        # Update best point
        min_idx = int(np.argmin(self.y))
        if self.y[min_idx] < self.best_y:
            self.best_y = float(self.y[min_idx])
            self.best_x = self.X[min_idx].copy()
        
        # Adapt step size based on improvement
        if self.n_eval > self.n_init:
            # Check if we improved in this batch
            if len(self.y) >= len(y_arr):
                new_best = float(np.min(self.y[-len(y_arr):]))
            else:
                new_best = self.best_y
            
            if new_best < self.last_best_y - 1e-10:
                # Improving: shrink step size for finer search
                self.step_size *= 0.9
                self.no_improve = 0
            else:
                # Not improving: expand step size and track stagnation
                self.step_size *= 1.05
                self.no_improve += 1
            
            self.last_best_y = min(self.last_best_y, self.best_y)
            
            # Restart if stuck for too long
            if self.no_improve >= 12 and self.n_eval < self.budget - 5:
                self.best_x = self.rng.uniform(self.lower, self.upper)
                self.best_y = np.inf
                self.step_size = 0.2 * self.domain_span
                self.no_improve = 0
                self.last_best_y = np.inf
            
            # Keep step size bounded (scalar)
            min_step = 0.001 * self.domain_span
            max_step = 0.4 * self.domain_span
            self.step_size = max(min_step, min(self.step_size, max_step))
