import numpy as np

class Optimizer:
    """Noise-aware continuous black-box optimizer for 10D problems with 120-query budget.
    
    Strategy:
    - Latin Hypercube initial design for space-filling coverage.
    - Adaptive local exploitation around best point with decaying scalar step size.
    - Occasional restart from archived top candidates for diversity.
    - Pure random exploration to maintain global coverage.
    - All vector operations are elementwise; step sizes are scalar floats.
    """
    
    def __init__(self, dim, lower, upper, budget, seed, rng):
        self.dim = dim
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.budget = int(budget)
        self.rng = rng
        
        # State tracking
        self.n_asked = 0
        self.best_x = None
        self.best_y = np.inf
        
        # Scalar step size (relative to mean domain range)
        domain_range = float(np.mean(self.upper - self.lower))
        self.step_size = 0.2 * domain_range
        self.step_min = 1e-4 * domain_range
        
        # Latin Hypercube initial design
        self.n_init = min(12, self.budget // 10)
        self.init_points = self._latin_hypercube(self.n_init)
        self.init_idx = 0
        
        # Archive of top candidates for restart diversity
        self.top_x = []
        self.top_y = []
        self.top_k = 5
        
        # Stagnation counter for step size adaptation
        self.stagnation = 0
        
        # Preferred batch size (verifier may cap)
        self.batch = 1
    
    def _latin_hypercube(self, n):
        """Generate n Latin Hypercube samples in [0,1]^dim, then scale to bounds."""
        points = np.empty((n, self.dim), dtype=np.float64)
        for d in range(self.dim):
            perm = self.rng.permutation(n)
            points[:, d] = (perm + self.rng.uniform(0, 1, n)) / n
        return self.lower + points * (self.upper - self.lower)
    
    def ask(self, n):
        """Return n candidate points within bounds. Shape (n, dim), finite, in-bounds."""
        n = int(n)
        if n <= 0:
            return np.empty((0, self.dim))
        
        points = []
        for _ in range(n):
            if self.init_idx < self.n_init:
                # Phase 1: LHS exploration
                p = self.init_points[self.init_idx].copy()
                self.init_idx += 1
            elif self.best_x is None:
                # No best yet: uniform random
                p = self.rng.uniform(self.lower, self.upper, size=self.dim)
            else:
                # Adaptive search
                r = self.rng.uniform()
                if r < 0.7:
                    # Local exploitation around best (elementwise vector addition)
                    progress = self.n_asked / self.budget
                    scale = self.step_size * max(0.01, 1.0 - 0.8 * progress)
                    pert = self.rng.normal(0, scale, size=self.dim)
                    p = self.best_x + pert
                elif r < 0.9 and len(self.top_x) > 1:
                    # Restart from a random archived top point
                    idx = int(self.rng.integers(len(self.top_x)))
                    scale = self.step_size * 0.5
                    pert = self.rng.normal(0, scale, size=self.dim)
                    p = self.top_x[idx] + pert
                else:
                    # Pure random exploration
                    p = self.rng.uniform(self.lower, self.upper, size=self.dim)
            
            # Clip to bounds and ensure finite values
            p = np.clip(p, self.lower, self.upper)
            p = np.nan_to_num(p, nan=self.lower, posinf=self.upper, neginf=self.lower)
            points.append(p)
            self.n_asked += 1
        
        return np.array(points[:n], dtype=np.float64)
    
    def tell(self, X, y, metadata=None):
        """Process observed evaluations. Accepts optional metadata (3-argument call)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()
        
        improved = False
        for i in range(len(y)):
            xi = X[i]
            yi = y[i]
            
            if not np.isfinite(yi):
                continue
            
            # Update best (scalar comparison)
            if yi < self.best_y:
                self.best_y = yi
                self.best_x = xi.copy()
                improved = True
            
            # Maintain top archive
            self.top_x.append(xi.copy())
            self.top_y.append(yi)
        
        # Keep only top-k in archive
        if len(self.top_x) > self.top_k:
            idx = np.argsort(self.top_y)
            self.top_x = [self.top_x[i] for i in idx[:self.top_k]]
            self.top_y = [self.top_y[i] for i in idx[:self.top_k]]
        
        # Adapt scalar step size based on improvement
        if improved:
            self.stagnation = 0
            domain_range = float(np.mean(self.upper - self.lower))
            self.step_size = min(self.step_size * 1.1, 0.5 * domain_range)
        else:
            self.stagnation += len(y)
            if self.stagnation > 5:
                self.step_size = max(self.step_size * 0.85, self.step_min)
                self.stagnation = 0
