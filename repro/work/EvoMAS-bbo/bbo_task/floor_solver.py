import numpy as np


class Optimizer:
    """(mu,lambda)-Evolution Strategy for 10D noisy optimization.

    Core mechanisms:
      - LHS initialization (10 evals)
      - Per-dim CSA with exponent capping (ratio clipped to [-1,1])
      - Crossover toward _gbx (45% blend, _cr 0.25->0.50 with restarts)
      - Per-generation stall detection (ys[0] via 3-gen window)
      - Hybrid refinement center (blend xmean <-> _gbx based on step size)
      - Per-dim refinement spread: step * 0.20 * (1-progress)
      - Dual xmean re-evaluation (at mild and strong convergence)
        for more reliable best-value anchor
    """

    batch = 1

    def __init__(self, dim, lower, upper, budget, seed, rng):
        self.dim = d = int(dim)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.budget = int(budget)
        self.rng = rng
        self._range = self.upper - self.lower
        self._rm = self._range.mean()

        # ------ ES population ------
        self.lam = 8                     # offspring per generation
        self.mu = 3                      # parents selected for recombination

        # power-law recombination weights
        w = np.log(self.mu + 0.5) - np.log(1 + np.arange(self.mu))
        self.weights = w / w.sum()

        # ------ State ------
        self.xmean = self.lower + self._range * self.rng.uniform(0, 1, d)
        self.step = np.full(d, 0.40 * self._rm)   # per-dim step sizes

        # cumulative step-size adaptation (CSA) path
        self._ps = np.zeros(d)
        self._tgt = np.sqrt(2.0 / np.pi)

        # ------ Best-point tracking ------
        self.best_x = self.xmean.copy()
        self.best_y = np.inf
        self._gbx = self.xmean.copy()
        self._gby = np.inf

        # ------ Counters ------
        self._used = 0
        self._gen = 0
        self._stalled = 0
        self._prev_best = np.inf
        self._restarts = 0
        self._max_restarts = 5
        self._cr = 0.25

        # Phase boundary — last 18 evals go to local refinement
        self._refine_start = max(budget - 18, 0)

        # Buffer for incremental batch updates
        self._bx = []
        self._by = []
        self._inited = False

        # ------ Generation-best history for stall detection ------
        self._gen_best_history = []

        # ------ Dual xmean re-evaluation ------
        # First re-eval: when step.mean() < 0.06rm and used >= 40 (mild convergence)
        self._xmean_reeval1_ready = False
        self._xmean_reeval1_done = False
        self._xmean_reeval1_active = False
        # Second re-eval: when step.mean() < 0.04rm and used >= 70 (strong convergence)
        self._xmean_reeval2_ready = False
        self._xmean_reeval2_done = False
        self._xmean_reeval2_active = False

    # ------------------------------------------------------------------
    #   ES internals
    # ------------------------------------------------------------------
    def _sample(self, n):
        """Sample n offspring from current distribution with crossover."""
        n = int(n)
        pts = self.xmean + self.step * self.rng.normal(0, 1, (n, self.dim))

        if self._gen > 1 and self._gby < np.inf:
            nc = max(1, int(n * self._cr))
            step_mean = float(self.step.mean())
            # Blend: stronger pull toward gbx when step sizes are large
            blend_self = 0.60 + 0.15 * np.clip(
                1.0 - step_mean / (0.05 * self._rm), 0.0, 1.0
            )
            blend_gbx = 1.0 - blend_self
            mask = self.rng.uniform(0, 1, (nc, self.dim)) < 0.5
            pts[:nc] = np.where(
                mask,
                blend_self * pts[:nc] + blend_gbx * self._gbx,
                pts[:nc]
            )

        return np.clip(pts, self.lower, self.upper)

    def _update(self, X, y):
        """One ES generation: rank, recombine, adapt step sizes."""
        n = len(y)
        if n < self.mu:
            return

        idx = np.argsort(y)
        Xs, ys = X[idx], y[idx]

        # Global best (conservative: only improve, never degrade)
        if ys[0] < self._gby:
            self._gby = ys[0]
            self._gbx = Xs[0].copy()
        if ys[0] < self.best_y:
            self.best_y = ys[0]
            self.best_x = Xs[0].copy()

        # Record generation-best for stall detection
        gen_best = ys[0]
        self._gen_best_history.append(gen_best)

        # Stall detection using generation-best history
        if self._gen > 0 and (self._gen % 2) == 0:
            if len(self._gen_best_history) >= 3:
                older = self._gen_best_history[-3]
                newer = self._gen_best_history[-1]
                if newer >= older - 1e-7:
                    self._stalled += 1
                else:
                    self._stalled = 0
                self._prev_best = newer

        # Restart on stall
        if self._stalled >= 2 and self._restarts < self._max_restarts:
            self._restart()
            return

        # Weighted recombination
        parents = Xs[:self.mu]
        new_mean = np.dot(self.weights, parents)

        # Cumulative step-size adaptation (per dimension), capped
        shift = new_mean - self.xmean
        _beta = 0.4
        self._ps = (1.0 - _beta) * self._ps + _beta * (shift / (self.step + 1e-10))

        pm = np.abs(self._ps)
        ratio = np.clip(pm / self._tgt - 1.0, -1.0, 1.0)
        self.step *= np.exp(0.25 * ratio)
        self.step = np.clip(self.step, 1e-4 * self._rm, 0.6 * self._rm)

        self.xmean = new_mean
        self._gen += 1

    def _restart(self):
        """Start a fresh ES from perturbed best point or random location."""
        if self.rng.uniform() < 0.6 and self._gby < np.inf:
            pert = 0.30 * self._rm * self.rng.normal(0, 1, self.dim)
            self.xmean = np.clip(self._gbx + pert, self.lower, self.upper)
        else:
            self.xmean = self.lower + self._range * self.rng.uniform(0, 1, self.dim)

        self.step[:] = 0.30 * self._rm
        self._ps[:] = 0.0
        self.best_y = np.inf
        self.best_x = self.xmean.copy()
        self._stalled = 0
        self._prev_best = np.inf
        self._gen_best_history = []
        self._restarts += 1
        self._cr = min(0.50, 0.25 + 0.05 * self._restarts)

        # Reset both re-evaluation flags so they can fire again
        self._xmean_reeval1_ready = False
        self._xmean_reeval1_done = False
        self._xmean_reeval1_active = False
        self._xmean_reeval2_ready = False
        self._xmean_reeval2_done = False
        self._xmean_reeval2_active = False

    # ------------------------------------------------------------------
    #   Initial design (Latin Hypercube)
    # ------------------------------------------------------------------
    def _lhs(self, n):
        n = int(n)
        p = np.empty((n, self.dim))
        for j in range(self.dim):
            pr = self.rng.permutation(n)
            p[:, j] = (pr + self.rng.uniform(0.0, 1.0, n)) / n
        return self.lower + self._range * p

    # ------------------------------------------------------------------
    #   Refinement center (hybrid: ES mean + global best)
    # ------------------------------------------------------------------
    def _refinement_center(self):
        """Return center for concentric refinement.

        Blends xmean (noise-averaged via recombination) with _gbx
        (single best observed) based on step-size convergence.
        """
        blend = float(np.clip(1.0 - self.step.mean() / (0.10 * self._rm), 0.0, 1.0))
        if blend <= 0.0 or self._gby >= np.inf:
            return self._gbx.copy()
        if blend >= 1.0:
            return self.xmean.copy()
        return (blend * self.xmean + (1.0 - blend) * self._gbx).copy()

    # ------------------------------------------------------------------
    #   Public interface
    # ------------------------------------------------------------------
    def ask(self, n):
        n = int(n)

        # Phase 0: Latin hypercube initial sweep
        if not self._inited:
            nd = min(n, self.lam - self._used)
            self._pd = self._lhs(nd)
            return self._pd

        # --- Re-evaluation 1: mild convergence (step.mean() < 0.06rm, used >= 40) ---
        if (not self._xmean_reeval1_done and not self._xmean_reeval1_ready
                and self._used >= 40 and self._used < self._refine_start
                and self.step.mean() < 0.06 * self._rm):
            self._xmean_reeval1_ready = True

        if self._xmean_reeval1_ready:
            self._xmean_reeval1_ready = False
            self._xmean_reeval1_done = True
            self._xmean_reeval1_active = True
            pts = np.empty((n, self.dim), dtype=float)
            pts[0] = np.clip(self.xmean, self.lower, self.upper)
            if n > 1:
                pts[1:] = self._sample(n - 1)
            self._pd = pts
            return pts

        # --- Re-evaluation 2: strong convergence (step.mean() < 0.04rm, used >= 70) ---
        if (not self._xmean_reeval2_done and not self._xmean_reeval2_ready
                and self._used >= 70 and self._used < self._refine_start
                and self.step.mean() < 0.04 * self._rm):
            self._xmean_reeval2_ready = True

        if self._xmean_reeval2_ready:
            self._xmean_reeval2_ready = False
            self._xmean_reeval2_done = True
            self._xmean_reeval2_active = True
            pts = np.empty((n, self.dim), dtype=float)
            pts[0] = np.clip(self.xmean, self.lower, self.upper)
            if n > 1:
                pts[1:] = self._sample(n - 1)
            self._pd = pts
            return pts

        # Phase 2: final concentric refinement
        if self._used >= self._refine_start:
            progress = (self._used - self._refine_start) / float(
                max(self.budget - self._refine_start, 1)
            )
            center = self._refinement_center()
            spread = np.maximum(
                self.step * 0.20 * (1.0 - progress),
                0.002 * self._rm
            )
            pts = center + spread * self.rng.normal(0, 1, (n, self.dim))
            np.clip(pts, self.lower, self.upper, out=pts)
            self._pd = pts
            return pts

        # Phase 1: standard ES generation
        pts = self._sample(n)
        self._pd = pts
        return pts

    def tell(self, X, y, metadata=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = int(len(y))

        # Count ALL evaluations used (verifier already counted them in budget)
        self._used += n

        # --- Handle re-evaluation 1 feedback ---
        if self._xmean_reeval1_active:
            self._xmean_reeval1_active = False
            # Improvement-only update: keep the better of stored and fresh
            if y[0] < self._gby:
                self._gby = y[0]
                self._gbx = X[0].copy()
            if y[0] < self.best_y:
                self.best_y = y[0]
                self.best_x = X[0].copy()
            if n > 1:
                # Strip the re-eval point; remaining points are regular ES samples
                X = X[1:]
                y = y[1:]
                n = len(y)
            else:
                return

        # --- Handle re-evaluation 2 feedback ---
        if self._xmean_reeval2_active:
            self._xmean_reeval2_active = False
            if y[0] < self._gby:
                self._gby = y[0]
                self._gbx = X[0].copy()
            if y[0] < self.best_y:
                self.best_y = y[0]
                self.best_x = X[0].copy()
            if n > 1:
                X = X[1:]
                y = y[1:]
                n = len(y)
            else:
                return

        # Update best immediately (conservative)
        for i in range(n):
            if y[i] < self._gby:
                self._gby = y[i]
                self._gbx = X[i].copy()
            if y[i] < self.best_y:
                self.best_y = y[i]
                self.best_x = X[i].copy()

        # ----- Initialisation -----
        if not self._inited:
            self._bx.append(X)
            self._by.append(y)
            if self._used >= self.lam:
                ax = np.concatenate(self._bx, axis=0)
                ay = np.concatenate(self._by, axis=0)
                bi = int(np.argmin(ay))
                self.xmean = ax[bi].copy()
                self.best_x = self.xmean.copy()
                self.best_y = ay[bi]
                self._gbx = self.best_x.copy()
                self._gby = self.best_y
                self.step[:] = 0.25 * self._rm
                self._inited = True
                self._bx = []
                self._by = []
            return

        # ----- ES update (only during Phase 1) -----
        if self._used < self._refine_start:
            self._bx.append(X)
            self._by.append(y)

            total_in_buffer = sum(len(b) for b in self._by)
            if total_in_buffer >= self.lam:
                ax = np.concatenate(self._bx, axis=0)
                ay = np.concatenate(self._by, axis=0)
                self._update(ax, ay)
                self._bx = []
                self._by = []
            elif self._used + n >= self._refine_start:
                # Flush whatever we have when approaching refinement boundary
                ax = np.concatenate(self._bx + [X], axis=0)
                ay = np.concatenate(self._by + [y], axis=0)
                if len(ax) >= self.mu:
                    self._update(ax, ay)
                self._bx = []
                self._by = []