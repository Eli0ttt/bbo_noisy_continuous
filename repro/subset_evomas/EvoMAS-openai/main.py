#!/usr/bin/env python3
"""
EvoMAS Main Entry Point - Evolutionary Synthesis of Multi-Agent Systems

Implements the complete EvoMAS pipeline:
1. Selection: Select k parent configurations from pool (π^S)
2. Evolution: Generate/Mutate/Crossover operations (π^M, π^C)
3. Evaluation: Execute MAS and compute reward R(C,q)
4. Pool Update: Add successful configurations to pool (C̄_{t+1})
5. Memory Update: Consolidate experiences (π^U)

Usage:
    python main.py --dataset workbench_email --max-steps 3
"""

import sys
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from src.meta_model.metamodel import MetaModel
from src.meta_model.selection import selection_operator, select_with_diversity
from src.meta_model.reward import compute_reward
from src.meta_model.pool import ConfigurationPool, add_to_pool_if_better
from src.meta_model.experience import consolidate_evolution_trace
from src.meta_model.mas_index import MASIndex
from src.dataset.load_dataset import load_dataset
from src.mas.interpreter import interpret_mas


def fmt_acc(v):
    """Format accuracy: 0-1 as percentage, 0-100 as score."""
    return f"{v:.2%}" if v <= 1.0 else f"{v:.1f}/100"


# ============================================================================
# EVOMAS CONFIGURATION PARAMETERS
# ============================================================================

# Meta-Model Settings
# Meta-model (planning / mutation / crossover / memory operators): Claude Sonnet 4.5.
META_MODEL_ID = "bedrock:global.anthropic.claude-sonnet-4-5-20250929-v1:0"
META_MODEL_TEMPERATURE = 0.7
META_MODEL_MAX_TOKENS = 8192

# Available models for MAS worker agents. The meta-model can select from this
# list when generating a new MAS config. Qwen3 235B leads as the default worker
# model; Claude Sonnet/Haiku remain available for agents that need alternative
# capabilities.
DEFAULT_MODEL_LIST = [
    "bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock:qwen.qwen3-235b-a22b-2507-v1:0",
    "bedrock:qwen.qwen3-coder-480b-a35b-v1:0",
]

# Selection Settings
NUM_PARENTS = 2  # Number of parent configurations to select (k)
USE_DIVERSITY = True  # Whether to use diversity-based selection
DIVERSITY_WEIGHT = 0.3  # Weight for diversity vs similarity (0-1)

# Evolution Settings
MAX_STEPS = 3  # Maximum evolutionary steps
NUM_EVAL_TASKS = 1  # Number of tasks for evaluation (1 = evolve on target query only)
MUTATION_PROB = 0.8  # Probability of mutation vs crossover

# Reward Function Settings
BETA = 1e-6  # Cost trade-off parameter (β ≥ 0); 10⁻⁶ balances Metrics∈[0,100] vs Cost∈[100K–32M]
COST_WEIGHT = "both"  # Cost metric: "tokens", "time", or "both"; Cost = tokens + 10³×time(s)

# Evaluation Settings
# LLM-as-judge is the default reward source for all datasets
# This avoids requiring environment setup (e.g., for SWE-bench test execution)
LLM_AS_JUDGE = "bedrock:global.anthropic.claude-sonnet-4-5-20250929-v1:0"  # Model ID for LLM-as-judge evaluation (None = use dataset evaluator)

# Pool Management Settings
IMPROVEMENT_THRESHOLD = 0.05  # Minimum reward improvement to add to pool (1%)
AUTO_POOL_DIR = True  # Automatically determine pool directory from dataset

# Dataset-to-Pool Mapping (used when AUTO_POOL_DIR=True)
DATASET_POOL_MAPPING = {
    # BBEH datasets
    "bbeh_": "mas_pools/bbeh",
    # WorkBench datasets
    "workbench_analytics": "mas_pools/workbench/analytics",
    "workbench_calendar": "mas_pools/workbench/calendar",
    "workbench_customer_relationship_manager": "mas_pools/workbench/customer_relationship_manager",
    "workbench_email": "mas_pools/workbench/email",
    "workbench_multi_domain": "mas_pools/workbench/multi_domain",
    "workbench_project_management": "mas_pools/workbench/project_management",
    # SWE-bench datasets
    "swebench": "mas_pools/swebench",
    "swe": "mas_pools/swebench",
}

# Output Settings
OUTPUT_DIR = "output"  # Base output directory
MEMORY_PATH = "src/meta_model/memory.json"  # Path to save/load memory

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pool_dir(dataset_name: str, manual_pool_dir: str = None) -> str:
    """
    Determine pool directory based on dataset name.

    Args:
        dataset_name: Name of the dataset
        manual_pool_dir: Manually specified pool directory (overrides auto)

    Returns:
        Pool directory path
    """
    if manual_pool_dir:
        return manual_pool_dir

    if not AUTO_POOL_DIR:
        return "mas_pools/bbeh"  # Default

    # Try exact match first
    if dataset_name in DATASET_POOL_MAPPING:
        return DATASET_POOL_MAPPING[dataset_name]

    # Try prefix match
    for prefix, pool_dir in DATASET_POOL_MAPPING.items():
        if dataset_name.startswith(prefix):
            return pool_dir

    # Default fallback
    logging.warning(f"No pool mapping found for {dataset_name}, using default: mas_pools/bbeh")
    return "mas_pools/bbeh"


def save_config(config_yaml: str, filepath: str):
    """Save configuration to file."""
    with open(filepath, 'w') as f:
        f.write(config_yaml)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MAIN EVOLUTION PIPELINE
# ============================================================================

def run_evolution_pipeline(
    dataset_name: str,
    pool_dir: str,
    num_eval_tasks: int = NUM_EVAL_TASKS,
    max_steps: int = MAX_STEPS,
    num_parents: int = NUM_PARENTS,
    seed: int = 42,
    output_dir: str = OUTPUT_DIR,
    meta_model_id: str = META_MODEL_ID,
    model_list: List[str] = None,
    llm_as_judge: Optional[str] = None,
    task_ids: Optional[List[Union[int, str]]] = None,
    memory_path: Optional[str] = None,
    memory_evolution: bool = True,
    batch_size: int = 1,
    workers: int = 1,
) -> Dict[str, Any]:
    """Run the EvoMAS evolutionary synthesis pipeline, with optional batching.

    Per the docstring algorithm, evolution is per-query: each task runs its
    own selection → generate → mutate/crossover → best → pool/memory update
    loop. `batch_size` controls how many tasks share a single evolution
    trajectory (scored jointly on aggregate reward) and `workers` controls
    how many such batches run in parallel.

    Args:
        dataset_name: Dataset to optimize for.
        pool_dir: Directory containing MAS pool.
        num_eval_tasks: Number of tasks for evaluation when task_ids is None.
            When task_ids is provided and batching applies, this is ignored
            in favor of the batch's own task_ids length.
        max_steps: Maximum evolutionary steps per batch.
        num_parents: Number of parent configurations to select per batch.
        seed: Random seed.
        output_dir: Output directory. When batching, each batch's outputs go
            under `<output_dir>/batch_NNN/`.
        meta_model_id: Model ID for meta-model evolutionary operators.
        model_list: List of available models for MAS agents.
        llm_as_judge: Model ID for LLM-as-judge evaluation (None = dataset evaluator).
        task_ids: Specific task IDs. If None or shorter than batch_size, the
            whole list runs as a single batch (no chunking). Otherwise the
            list is split into chunks of `batch_size` and each chunk runs its
            own evolution trajectory.
        memory_path: Path to the meta-model memory JSON file.
            - None (default): resolves to
              `dataset/<dataset_name>/memory_<YYYYMMDD_HHMMSS>.json`.
            - Explicit path: load if exists, save back (when memory_evolution=True).
            When workers > 1, each batch writes to a sibling file with a
            `_batchNNN` suffix to avoid concurrent-write races.
        memory_evolution: If True, persist memory updates; if False, read-only.
        batch_size: Tasks per evolution batch. Default 1 = per-query (algorithm
            as described in the docstring). Larger = batch mode: each evolution
            step scores one MAS on `batch_size` tasks jointly on aggregate reward.
        workers: Number of batches run in parallel (via ThreadPoolExecutor).
            Default 1 = serial. Use >1 when batches can run independently — note
            that pool mutations and memory writes across parallel batches are
            isolated per-batch for safety (pool_dir is still shared, but each
            batch's added config gets a unique timestamped filename).

    Returns:
        When a single batch runs, returns the per-batch result dict directly
        (keys: initial_accuracy, final_accuracy, final_official_accuracy,
        initial_reward, final_reward, improved, added_to_pool, num_operations,
        output_location).
        When multiple batches run, returns an aggregated dict with
        batch-size-weighted means for scalar metrics plus `batches` (list of
        per-batch dicts), `num_batches`, `num_failed_batches`.
    """
    # Basic validation that applies regardless of batching
    if model_list is None:
        model_list = DEFAULT_MODEL_LIST
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if num_parents < 1:
        raise ValueError(f"num_parents must be >= 1, got {num_parents}")
    if num_eval_tasks < 1:
        raise ValueError(f"num_eval_tasks must be >= 1, got {num_eval_tasks}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    # Resolve the default memory path once here so per-batch files (if any)
    # are derived consistently from the same base path. Each dataset family
    # has subset-specific folders already under dataset/; memory is nested
    # alongside that subset's data so co-located artifacts stay together.
    if memory_path is None:
        from datetime import datetime
        dsn = dataset_name.lower()
        if dsn.startswith("bbeh_"):
            # bbeh_<subset> → dataset/bbeh/benchmark_tasks/bbeh_<subset>/
            mem_dir = Path("dataset") / "bbeh" / "benchmark_tasks" / dataset_name
        elif dsn.startswith("workbench_"):
            # workbench_<domain> → dataset/workbench/<domain>/
            domain = dataset_name[len("workbench_"):]
            mem_dir = Path("dataset") / "workbench" / domain
        elif dsn.startswith("swe_bench_") or dsn.startswith("swebench"):
            # swe_bench_<variant> lives at dataset/swe_bench_<variant>/
            safe_ds = dataset_name.replace("/", "_").replace(" ", "_")
            mem_dir = Path("dataset") / safe_ds
        else:
            # Fallback: dataset/<dataset_name>/
            safe_ds = dataset_name.replace("/", "_").replace(" ", "_")
            mem_dir = Path("dataset") / safe_ds
        indicator = datetime.now().strftime("%Y%m%d_%H%M%S")
        mem_dir.mkdir(parents=True, exist_ok=True)
        memory_path = str(mem_dir / f"memory_{indicator}.json")

    # Fast path: no batching needed (task_ids too short or absent)
    if task_ids is None or len(task_ids) <= batch_size:
        return _run_single_batch(
            dataset_name=dataset_name,
            pool_dir=pool_dir,
            num_eval_tasks=num_eval_tasks,
            max_steps=max_steps,
            num_parents=num_parents,
            seed=seed,
            output_dir=output_dir,
            meta_model_id=meta_model_id,
            model_list=model_list,
            llm_as_judge=llm_as_judge,
            task_ids=task_ids,
            memory_path=memory_path,
            memory_evolution=memory_evolution,
        )

    # Chunk task_ids into batches
    batches = [task_ids[i:i + batch_size] for i in range(0, len(task_ids), batch_size)]
    n_batches = len(batches)
    logger.info(
        f"BATCHING: {len(task_ids)} tasks → {n_batches} batches of up to {batch_size} "
        f"(workers={workers}, memory_path={memory_path})"
    )

    def _per_batch_memory(base: Optional[str], batch_idx: int) -> Optional[str]:
        if base is None or workers <= 1:
            return base
        bp = Path(base)
        return str(bp.parent / f"{bp.stem}_batch{batch_idx:03d}{bp.suffix}")

    def run_one(batch_idx: int, batch_task_ids):
        batch_output_dir = str(Path(output_dir) / f"batch_{batch_idx:03d}")
        Path(batch_output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"[batch_{batch_idx:03d}] starting ({len(batch_task_ids)} tasks)")
        try:
            result = _run_single_batch(
                dataset_name=dataset_name,
                pool_dir=pool_dir,
                num_eval_tasks=len(batch_task_ids),
                max_steps=max_steps,
                num_parents=num_parents,
                seed=seed,
                output_dir=batch_output_dir,
                meta_model_id=meta_model_id,
                model_list=model_list,
                llm_as_judge=llm_as_judge,
                task_ids=batch_task_ids,
                memory_path=_per_batch_memory(memory_path, batch_idx),
                memory_evolution=memory_evolution,
            )
            result["batch_idx"] = batch_idx
            return result
        except Exception as e:
            logger.exception(f"[batch_{batch_idx:03d}] FAILED: {e}")
            return {"batch_idx": batch_idx, "error": str(e)}

    if workers <= 1:
        batch_results = [run_one(i, b) for i, b in enumerate(batches)]
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        batch_results = [None] * n_batches
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(run_one, i, b): i for i, b in enumerate(batches)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    batch_results[i] = fut.result()
                except Exception as e:
                    logger.exception(f"[batch_{i:03d}] executor raised: {e}")
                    batch_results[i] = {"batch_idx": i, "error": str(e)}

    return _aggregate_batch_results(batch_results, batches, output_dir)


def _aggregate_batch_results(results, batches, output_dir) -> Dict[str, Any]:
    """Combine per-batch results into a single summary dict."""
    def weighted_mean(key: str):
        total_w, total = 0.0, 0.0
        for r, b in zip(results, batches):
            if not isinstance(r, dict) or "error" in r:
                continue
            v = r.get(key)
            if v is None:
                continue
            w = float(len(b))
            total_w += w
            total += float(v) * w
        return (total / total_w) if total_w > 0 else None

    valid = [r for r in results if isinstance(r, dict) and "error" not in r]
    return {
        "initial_accuracy": weighted_mean("initial_accuracy"),
        "final_accuracy": weighted_mean("final_accuracy"),
        "final_official_accuracy": weighted_mean("final_official_accuracy"),
        "initial_reward": weighted_mean("initial_reward"),
        "final_reward": weighted_mean("final_reward"),
        "improved": any(r.get("improved") for r in valid),
        "added_to_pool": any(r.get("added_to_pool") for r in valid),
        "num_operations": sum((r.get("num_operations") or 0) for r in valid),
        "output_location": str(output_dir),
        "num_batches": len(batches),
        "num_failed_batches": sum(1 for r in results if isinstance(r, dict) and "error" in r),
        "batches": results,
    }


def _run_single_batch(
    dataset_name: str,
    pool_dir: str,
    num_eval_tasks: int = NUM_EVAL_TASKS,
    max_steps: int = MAX_STEPS,
    num_parents: int = NUM_PARENTS,
    seed: int = 42,
    output_dir: str = OUTPUT_DIR,
    meta_model_id: str = META_MODEL_ID,
    model_list: List[str] = None,
    llm_as_judge: Optional[str] = None,
    task_ids: Optional[List[Union[int, str]]] = None,
    memory_path: Optional[str] = None,
    memory_evolution: bool = True,
) -> Dict[str, Any]:
    """Execute one batch of evolutionary synthesis.

    This is the original per-batch body of the algorithm. It scores one MAS
    on all given task_ids jointly and uses aggregate reward to drive
    selection / mutate / crossover / pool+memory updates. Invoked directly by
    run_evolution_pipeline when batching is not needed, or once per chunk
    when batching is.
    """
    # Use default model list if not provided
    if model_list is None:
        model_list = DEFAULT_MODEL_LIST

    # Parameter validation
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if num_parents < 1:
        raise ValueError(f"num_parents must be >= 1, got {num_parents}")
    if num_eval_tasks < 1:
        raise ValueError(f"num_eval_tasks must be >= 1, got {num_eval_tasks}")

    pool_path = Path(pool_dir)
    if not pool_path.exists():
        raise ValueError(f"pool_dir does not exist: {pool_dir}")
    yaml_files = list(pool_path.glob("*.yaml")) + list(pool_path.glob("*.yml"))
    if not yaml_files:
        raise ValueError(f"pool_dir contains no YAML files: {pool_dir}")

    # Resolve per-run memory path default: dataset/<dataset_name>/memory_<timestamp>.json
    # so each run gets its own fresh memory, nested under the dataset's directory.
    # Callers can pass an explicit memory_path to load an existing memory file.
    if memory_path is None:
        from datetime import datetime
        safe_ds = dataset_name.replace("/", "_").replace(" ", "_")
        indicator = datetime.now().strftime("%Y%m%d_%H%M%S")
        mem_dir = Path("dataset") / safe_ds
        mem_dir.mkdir(parents=True, exist_ok=True)
        memory_path = str(mem_dir / f"memory_{indicator}.json")

    logger.info(f"Validated parameters: max_steps={max_steps}, num_parents={num_parents}, "
                f"num_eval_tasks={num_eval_tasks}, pool_dir={pool_dir} ({len(yaml_files)} configs)")
    logger.info(f"Memory: path={memory_path}, evolution={memory_evolution}")

    logger.info("=" * 80)
    logger.info("EVOMAS - EVOLUTIONARY SYNTHESIS OF MULTI-AGENT SYSTEMS")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Pool: {pool_dir}")
    logger.info(f"Evaluation tasks: {num_eval_tasks} (seed={seed})")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Parents to select: {num_parents}")
    logger.info(f"Meta-model: {meta_model_id}")
    logger.info(f"Available models for MAS agents: {len(model_list)}")
    logger.info(f"Evaluator: {llm_as_judge if llm_as_judge else 'Dataset-specific'}")
    logger.info(f"Reward β: {BETA}, Cost weight: {COST_WEIGHT}")
    logger.info(f"Improvement threshold: {IMPROVEMENT_THRESHOLD}")
    logger.info("=" * 80)

    # Initialize components
    meta_model = MetaModel(
        model_id=meta_model_id,
        temperature=META_MODEL_TEMPERATURE,
        max_tokens=META_MODEL_MAX_TOKENS,
        memory_path=memory_path,
        memory_evolution=memory_evolution,
        verbose=True,
    )

    pool = ConfigurationPool(pool_dir=pool_dir)
    logger.info(f"Loaded pool with {len(pool)} configurations")

    # Initialize MAS index (tracks structure, performance, solved tasks)
    mas_index = MASIndex(pool_dir)
    logger.info(f"MAS index: {len(mas_index)} configurations indexed")

    # Load dataset for task samples
    dataset = load_dataset(dataset_name, split="test")

    # SWE-bench uses string instance IDs (e.g. "astropy__astropy-12907");
    # BBEH/WorkBench use int indices. If callers pass int indices for an
    # SWE dataset, resolve them to string IDs here so downstream MasRunner
    # lookups (get_by_id) work.
    if task_ids is not None and "swe" in dataset_name.lower() and task_ids and isinstance(task_ids[0], int):
        resolved = []
        for i in task_ids:
            if 0 <= i < len(dataset):
                resolved.append(dataset[i].id)
        if resolved:
            logger.info(f"Resolved {len(task_ids)} SWE task indices -> string IDs (first 3: {resolved[:3]})")
            task_ids = resolved

    if task_ids is not None:
        # Support both int indices (BBEH/WorkBench) and string IDs (SWE-bench)
        task_samples = []
        for tid in task_ids[:3]:
            if isinstance(tid, str):
                t = dataset.get_by_id(tid)
                if t is not None:
                    task_samples.append(t.model_dump())
            else:
                task_samples.append(dataset[tid].model_dump())
    else:
        task_samples = [dataset[i].model_dump() for i in range(min(3, len(dataset)))]

    # Build task_query from the actual task content (not a dataset-specific
    # label) so memory entries carry semantic information that transfers
    # across domains. A BBEH reasoning memory is only useful to a WorkBench
    # selection if the meta-model can see what the original task was about.
    def _task_text(sample):
        return (sample.get("query") or sample.get("q") or "").strip()

    # Effective evaluation size for the memory prefix. When task_ids is
    # provided explicitly, use its length. Otherwise use num_eval_tasks
    # (the planned sample size), not len(dataset).
    effective_n = len(task_ids) if task_ids is not None else num_eval_tasks

    if task_samples:
        if effective_n == 1:
            single = _task_text(task_samples[0])
            tid_str = str(task_ids[0]) if task_ids else "0"
            task_query = (single or f"Task {tid_str} from {dataset_name}")[:1000]
        else:
            pieces = [_task_text(s)[:250] for s in task_samples[:3]]
            pieces = [p for p in pieces if p]
            if pieces:
                prefix = f"[{effective_n} tasks | dataset={dataset_name}] "
                task_query = prefix + " ||| ".join(pieces)
                if effective_n > len(pieces):
                    task_query += f" ||| ... +{effective_n - len(pieces)} more"
                task_query = task_query[:2000]
            else:
                task_query = f"Tasks from {dataset_name} dataset"
    else:
        task_query = f"Tasks from {dataset_name} dataset"

    # ========================================================================
    # STEP 1: SELECTION - π^S(q, D(C̄))
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: SELECTION (via meta-model)")
    logger.info("=" * 80)

    # Build task description from samples for meta-model context
    task_description = f"Dataset: {dataset_name}\n"
    for i, sample in enumerate(task_samples[:2]):
        q = sample.get("query", sample.get("q", ""))[:300]
        task_description += f"Example task {i+1}: {q}...\n"

    # Use meta-model for selection (informed by MAS index + memory)
    try:
        parent_paths = meta_model.select(
            task_query=task_query,
            task_description=task_description,
            pool_dir=pool_dir,
            k=num_parents,
            mas_index=mas_index,
        )
    except Exception as e:
        # Fallback to programmatic selection if meta-model fails
        logger.warning(f"Meta-model selection failed ({e}), falling back to diversity-based selection")
        parent_paths = select_with_diversity(
            query=task_query,
            pool_dir=pool_dir,
            k=num_parents,
            diversity_weight=DIVERSITY_WEIGHT
        )

    if not parent_paths:
        raise ValueError(f"No configurations found in pool: {pool_dir}")

    logger.info(f"Selected {len(parent_paths)} parent configurations:")
    for i, path in enumerate(parent_paths, 1):
        logger.info(f"   {i}. {Path(path).name}")

    # Load parent configs
    parent_configs = []
    for path in parent_paths:
        with open(path, 'r') as f:
            parent_configs.append(f.read())

    # ========================================================================
    # STEP 2: INITIALIZE EVOLUTION
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: INITIALIZE EVOLUTION")
    logger.info("=" * 80)

    # Use temp directory for intermediate configs
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix="evomas_"))
    logger.info(f"Temporary directory: {temp_dir}")

    # Evolution trace O_q
    evolution_operations = []

    # Generate and evaluate all k parent configurations
    temp_output = temp_dir / "temp_output"
    all_configs = []
    all_stats = []
    all_rewards = []

    for p_idx, p_path in enumerate(parent_paths):
        logger.info(f"\nGENERATE: Adapting parent {p_idx+1}/{len(parent_paths)} ({Path(p_path).name}) to tasks")
        adapted_config = meta_model.generate(
            mas_config_path=p_path,
            task_samples=task_samples,
            task_description=task_query,
            model_list=model_list
        )

        adapted_path = temp_dir / f"step0_parent{p_idx}.yaml"
        save_config(adapted_config, str(adapted_path))

        logger.info(f"Evaluating parent {p_idx+1}/{len(parent_paths)}...")
        adapted_result = interpret_mas(
            config_path=str(adapted_path),
            dataset_name=dataset_name,
            num_tasks=num_eval_tasks,
            task_ids=task_ids,
            save_outputs=False,
            output_dir=str(temp_output),
            verbose=False,
            llm_as_judge=llm_as_judge
        )

        adapted_stats = adapted_result.get('statistics', {})
        adapted_reward = compute_reward(adapted_stats, beta=BETA, cost_weight=COST_WEIGHT)

        logger.info(f"Parent {p_idx+1} Accuracy: {fmt_acc(adapted_stats.get('accuracy', 0))}")
        logger.info(f"Parent {p_idx+1} Reward: {adapted_reward:.4f}")

        all_configs.append(adapted_config)
        all_stats.append(adapted_stats)
        all_rewards.append(adapted_reward)

    # Track best configuration across all k parents
    best_idx = all_rewards.index(max(all_rewards))
    best_config = all_configs[best_idx]
    best_stats = all_stats[best_idx]
    best_reward = all_rewards[best_idx]

    logger.info(f"\nBest initial parent: {best_idx+1} with reward {best_reward:.4f} (accuracy {fmt_acc(best_stats.get('accuracy', 0))})")

    # ========================================================================
    # STEP 3: EVOLUTIONARY LOOP
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: EVOLUTIONARY LOOP")
    logger.info("=" * 80)

    for step in range(1, max_steps + 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"EVOLUTION STEP {step}/{max_steps}")
        logger.info(f"{'=' * 80}")

        # Decide: Mutate OR Crossover (not both)
        use_mutation = random.random() < MUTATION_PROB

        # Fall back to mutation if crossover chosen but <2 configs exist
        if not use_mutation and len(all_configs) < 2:
            logger.info("Crossover selected but <2 configs available, falling back to Mutate")
            use_mutation = True

        old_best_accuracy = best_stats.get('accuracy', 0)
        old_best_config = best_config

        if use_mutation:
            # ----------------------------------------------------------------
            # MUTATION - π^M(C_i, f(τ(C_i)), E)
            # ----------------------------------------------------------------
            logger.info(f"\nMUTATE: Refining configuration based on feedback")

            observations = (
                f"Current accuracy: {fmt_acc(best_stats.get('accuracy', 0))}\n"
                f"Current reward: {best_reward:.4f}\n"
                f"Total tokens: {best_stats.get('total_tokens', 0):,}\n"
            )

            offspring_config = meta_model.mutate(
                mas_config=best_config,
                execution_logs=best_stats,
                observations=observations,
                model_list=model_list
            )
            action_type = 'mutation'
            offspring_path = temp_dir / f"step{step}_mutated.yaml"
        else:
            # ----------------------------------------------------------------
            # CROSSOVER - π^C(C_i, C_j, f(τ(C_i)), f(τ(C_j)), E)
            # ----------------------------------------------------------------
            logger.info(f"\nCROSSOVER: Combining configurations")

            # Select two best configs
            sorted_indices = sorted(
                range(len(all_rewards)),
                key=lambda i: all_rewards[i],
                reverse=True
            )
            parent1_idx = sorted_indices[0]
            parent2_idx = sorted_indices[1]

            offspring_config = meta_model.crossover(
                mas_config_1=all_configs[parent1_idx],
                mas_config_2=all_configs[parent2_idx],
                logs_1=all_stats[parent1_idx],
                logs_2=all_stats[parent2_idx],
                model_list=model_list
            )
            action_type = 'crossover'
            offspring_path = temp_dir / f"step{step}_crossover.yaml"

        # Save and evaluate offspring
        save_config(offspring_config, str(offspring_path))

        logger.info(f"Evaluating {action_type} offspring...")
        offspring_result = interpret_mas(
            config_path=str(offspring_path),
            dataset_name=dataset_name,
            num_tasks=num_eval_tasks,
            task_ids=task_ids,
            save_outputs=False,
            output_dir=str(temp_output),
            verbose=False,
            llm_as_judge=llm_as_judge
        )

        offspring_stats = offspring_result.get('statistics', {})
        offspring_reward = compute_reward(offspring_stats, beta=BETA, cost_weight=COST_WEIGHT)

        logger.info(f"Offspring Accuracy: {fmt_acc(offspring_stats.get('accuracy', 0))}")
        logger.info(f"Offspring Reward: {offspring_reward:.4f}")

        all_configs.append(offspring_config)
        all_stats.append(offspring_stats)
        all_rewards.append(offspring_reward)

        # Record operation
        evolution_operations.append({
            'type': action_type,
            'step': step,
            'accuracy': offspring_stats.get('accuracy', 0),
            'reward': offspring_reward,
            'improved': offspring_reward > best_reward,
            'changes': f'{action_type.capitalize()} applied',
            'accuracy_change': offspring_stats.get('accuracy', 0) - best_stats.get('accuracy', 0)
        })

        # Update best if improved
        if offspring_reward > best_reward:
            best_config = offspring_config
            best_stats = offspring_stats
            best_reward = offspring_reward
            logger.info(f"New best reward: {best_reward:.4f}")

        # Memory update - record experience for cross-query transfer
        meta_model.update_memory(
            action_type=action_type,
            query=task_query,
            old_config=old_best_config,
            new_config=offspring_config,
            old_accuracy=old_best_accuracy,
            new_accuracy=offspring_stats.get('accuracy', 0)
        )

    # ========================================================================
    # STEP 4: SELECT BEST - argmax R(C, q)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: SELECT BEST CONFIGURATION")
    logger.info("=" * 80)

    best_idx = all_rewards.index(max(all_rewards))
    final_config = all_configs[best_idx]
    final_stats = all_stats[best_idx]
    final_reward = all_rewards[best_idx]

    logger.info(f"Best configuration found at index {best_idx}")
    logger.info(f"Final Accuracy: {fmt_acc(final_stats.get('accuracy', 0))}")
    logger.info(f"Final Reward: {final_reward:.4f}")

    # ========================================================================
    # STEP 5: CONSOLIDATE EXPERIENCE - π^U(O_q)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: CONSOLIDATE EXPERIENCE")
    logger.info("=" * 80)

    initial_accuracy = all_stats[0].get('accuracy', 0)
    final_accuracy = final_stats.get('accuracy', 0)

    consolidated_summary = consolidate_evolution_trace(
        operations=evolution_operations,
        query=task_query,
        initial_accuracy=initial_accuracy,
        final_accuracy=final_accuracy,
        best_config=final_config
    )

    logger.info("\n" + consolidated_summary)

    # ========================================================================
    # STEP 6: UPDATE POOL - C̄_{t+1} = C̄_t ∪ {C_q}
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: UPDATE CONFIGURATION POOL")
    logger.info("=" * 80)

    # Compute parent rewards for comparison
    parent_stats_list = []
    for parent_path in parent_paths:
        logger.info(f"Evaluating parent: {Path(parent_path).name}")
        parent_result = interpret_mas(
            config_path=parent_path,
            dataset_name=dataset_name,
            num_tasks=num_eval_tasks,
            task_ids=task_ids,
            save_outputs=False,
            output_dir=str(temp_output),
            verbose=False,
            llm_as_judge=llm_as_judge
        )
        parent_stats_list.append(parent_result.get('statistics', {}))

    # Check if should add to pool
    added_path = add_to_pool_if_better(
        pool=pool,
        config_yaml=final_config,
        new_metrics=final_stats,
        parent_metrics=parent_stats_list,
        task_query=task_query,
        beta=BETA,
        cost_weight=COST_WEIGHT,
        threshold=IMPROVEMENT_THRESHOLD
    )

    if added_path:
        logger.info(f"Configuration added to pool: {Path(added_path).name}")
        # Index the new config
        mas_index.add_config(added_path)
    else:
        logger.info("Configuration not added to pool (insufficient improvement)")

    # Update MAS index with query results for the winning config
    # Record which config won this query (for future selection)
    best_config_name = Path(parent_paths[best_idx]).stem if best_idx < len(parent_paths) else "evolved"
    if added_path:
        best_config_name = Path(added_path).stem
    query_snippet = task_query[:150]
    task_id_str = str(task_ids[0]) if task_ids else str(seed)
    mas_index.record_query_result(
        config_name=best_config_name,
        task_id=task_id_str,
        query_snippet=query_snippet,
        accuracy=final_stats.get('accuracy', 0),
        tokens=final_stats.get('total_tokens', 0),
        time_seconds=final_stats.get('total_time', 0),
        is_winner=True
    )

    # ========================================================================
    # STEP 7: FINAL EVALUATION & SAVE
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: FINAL EVALUATION & SAVE RESULTS")
    logger.info("=" * 80)

    # Save best config for final evaluation
    # Name it "evomas.yaml" so outputs go to output/{dataset}/evomas/
    final_config_path = temp_dir / "evomas.yaml"
    save_config(final_config, str(final_config_path))

    # Run final evaluation with save_outputs=True
    # Use dataset-specific evaluator (ground-truth) for final accuracy/solved-rate,
    # not the LLM-as-judge used during evolution for reward signals.
    logger.info("Running final evaluation (dataset evaluator)...")
    final_result = interpret_mas(
        config_path=str(final_config_path),
        dataset_name=dataset_name,
        num_tasks=num_eval_tasks,
        task_ids=task_ids,
        save_outputs=True,
        output_dir=output_dir,
        verbose=True,
        llm_as_judge=None,
        skip_evaluation=False
    )

    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info("Cleaned up temporary files")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    # Pull the OFFICIAL ground-truth accuracy from STEP 7's final eval
    final_official_accuracy = final_result.get("statistics", {}).get("accuracy")

    logger.info("\n" + "=" * 80)
    logger.info("EVOLUTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Initial Accuracy (judge):    {fmt_acc(initial_accuracy)}")
    logger.info(f"Final Accuracy (judge):      {fmt_acc(final_accuracy)}")
    if final_official_accuracy is not None:
        logger.info(f"Final Accuracy (OFFICIAL):   {fmt_acc(final_official_accuracy)}")
    logger.info(f"Initial Reward: {all_rewards[0]:.4f}")
    logger.info(f"Final Reward:   {final_reward:.4f}")

    if final_accuracy > initial_accuracy:
        improvement = final_accuracy - initial_accuracy
        logger.info(f"IMPROVED by {'+' if improvement > 0 else ''}{improvement:.1f}{'pp' if initial_accuracy > 1.0 else '%'}!")
    elif final_accuracy == initial_accuracy:
        logger.info(f"No change in accuracy")
    else:
        logger.info(f"Accuracy decreased")

    output_location = f"{output_dir}/{dataset_name}/evomas/"
    logger.info(f"Results saved to: {output_location}")
    logger.info("=" * 80)

    return {
        "initial_accuracy": initial_accuracy,
        "final_accuracy": final_accuracy,
        "final_official_accuracy": final_official_accuracy,
        "initial_reward": all_rewards[0],
        "final_reward": final_reward,
        "improved": final_accuracy > initial_accuracy,
        "output_location": output_location,
        "num_operations": len(evolution_operations),
        "added_to_pool": added_path is not None
    }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EvoMAS - Evolutionary Synthesis of Multi-Agent Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evolution on workbench email dataset
  python main.py --dataset workbench_email

  # Run with more steps and evaluation tasks
  python main.py --dataset workbench_email --max-steps 5 --num-eval-tasks 20

  # Run on BBEH dataset with custom pool
  python main.py --dataset bbeh_word_sorting --pool-dir mas_pools/bbeh

  # Run with specific meta-model and model list
  python main.py --dataset workbench_email \\
    --meta-model-id bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0 \\
    --model-list bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0 \\
                  bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0

  # Run with LLM-as-judge evaluation (instead of dataset-specific)
  python main.py --dataset workbench_email \\
    --llm-as-judge bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0

  # Run on SWE-bench dataset
  python main.py --dataset swebench_lite --num-eval-tasks 5
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., workbench_email, bbeh_word_sorting, swebench_lite)"
    )

    parser.add_argument(
        "--pool-dir",
        type=str,
        default=None,
        help="Directory containing MAS pool (auto-detected if not specified)"
    )

    parser.add_argument(
        "--num-eval-tasks",
        type=int,
        default=NUM_EVAL_TASKS,
        help=f"Number of tasks for evaluation (default: {NUM_EVAL_TASKS})"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Maximum evolutionary steps (default: {MAX_STEPS})"
    )

    parser.add_argument(
        "--num-parents",
        type=int,
        default=NUM_PARENTS,
        help=f"Number of parent configurations to select (default: {NUM_PARENTS})"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for task selection (default: 42)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )

    parser.add_argument(
        "--meta-model-id",
        type=str,
        default=META_MODEL_ID,
        help=f"Model ID for meta-model evolutionary operators (default: {META_MODEL_ID})"
    )

    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="List of available models for MAS agents (space-separated)"
    )

    parser.add_argument(
        "--llm-as-judge",
        type=str,
        default=LLM_AS_JUDGE,
        help=f"Model ID for LLM-as-judge evaluation (default: {LLM_AS_JUDGE}). Set to 'none' to use dataset-specific evaluator."
    )

    def _task_id(value: str):
        """Accept either an integer index (BBEH/WorkBench) or a string
        instance ID (SWE-bench, e.g. 'astropy__astropy-12907')."""
        try:
            return int(value)
        except ValueError:
            return value

    parser.add_argument(
        "--task-ids",
        type=_task_id,
        nargs="+",
        default=None,
        help="Specific task IDs to evolve on. Accepts int indices "
             "(BBEH/WorkBench) or string instance IDs (SWE-bench, "
             "e.g. 'astropy__astropy-12907'). Overrides --num-eval-tasks."
    )

    parser.add_argument(
        "--memory-path",
        type=str,
        default=None,
        help="Meta-model memory JSON path. Default = dataset/<dataset>/memory_<timestamp>.json (fresh per run). Pass an existing path to continue a prior memory."
    )

    parser.add_argument(
        "--memory-evolution",
        type=str,
        choices=["true", "false"],
        default="true",
        help="If 'true' (default), the run updates + saves memory. If 'false', memory is read-only (useful for clean A/B comparisons)."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Tasks per evolution batch. Default 1 = per-query (paper algorithm). Set to len(task_ids) for single-batch batch mode."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of batches run in parallel inside the pipeline. Default 1 = serial. Increase cautiously (watch API rate limits)."
    )

    args = parser.parse_args()

    # Determine pool directory
    pool_dir = get_pool_dir(args.dataset, args.pool_dir)

    # Validate pool directory exists
    pool_path = Path(pool_dir)
    if not pool_path.exists():
        logger.error(f"Pool directory not found: {pool_dir}")
        logger.error(f"Please create the pool directory or specify a valid --pool-dir")
        return 1

    try:
        # Run evolution pipeline
        results = run_evolution_pipeline(
            dataset_name=args.dataset,
            pool_dir=pool_dir,
            num_eval_tasks=args.num_eval_tasks,
            max_steps=args.max_steps,
            num_parents=args.num_parents,
            seed=args.seed,
            output_dir=args.output_dir,
            meta_model_id=args.meta_model_id,
            model_list=args.model_list if args.model_list else None,
            llm_as_judge=args.llm_as_judge if args.llm_as_judge and args.llm_as_judge.lower() != 'none' else None,
            task_ids=args.task_ids,
            memory_path=args.memory_path,
            memory_evolution=(args.memory_evolution.lower() == "true"),
            batch_size=args.batch_size,
            workers=args.workers,
        )

        # Print summary
        print("\n" + "=" * 80)
        print("EVOMAS EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Dataset: {args.dataset}")
        print(f"Initial Accuracy: {fmt_acc(results['initial_accuracy'])}")
        print(f"Final Accuracy:   {fmt_acc(results['final_accuracy'])}")
        print(f"Initial Reward:   {results['initial_reward']:.4f}")
        print(f"Final Reward:     {results['final_reward']:.4f}")
        print(f"Improvement: {'Yes ' if results['improved'] else 'No'}")
        print(f"Added to Pool: {'Yes ' if results['added_to_pool'] else 'No'}")
        print(f"Evolution Operations: {results['num_operations']}")
        print(f"Results saved to: {results['output_location']}")
        print("=" * 80)

        return 0

    except KeyboardInterrupt:
        logger.warning("\nEvolution interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
