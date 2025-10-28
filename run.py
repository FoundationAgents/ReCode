import argparse
import asyncio
import importlib
import time
import shutil
from pathlib import Path
from typing import Type, Dict, Any, List, Optional
import inspect

import logging
from datetime import datetime

# Suppress common warnings - must be at the very beginning
import warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

# Suppress all categories of warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

# Make sure these are applied globally
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Specific warning suppressions
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Module-specific suppressions
warnings.filterwarnings("ignore", module="gym")
warnings.filterwarnings("ignore", module="gym.*")
warnings.filterwarnings("ignore", module="faiss")
warnings.filterwarnings("ignore", module="faiss.*") 
warnings.filterwarnings("ignore", module="setuptools")
warnings.filterwarnings("ignore", module="setuptools.*")
warnings.filterwarnings("ignore", module="typer")
warnings.filterwarnings("ignore", module="typer.*")
warnings.filterwarnings("ignore", module="spacy")
warnings.filterwarnings("ignore", module="spacy.*")
warnings.filterwarnings("ignore", module="click")
warnings.filterwarnings("ignore", module="click.*")

from tqdm import tqdm

from base.agent import Agent
from base.environment import Env
from utils.logger import SimpleLogger
from utils.errors import StepLimitError

# Allow short aliases like `-a human` and `-e alfworld`
AGENT_ALIASES = {
    "recode": "agents.recode.agent.ReCodeAgent",
}

ENV_ALIASES = {
    "alfworld": "envs.alfworld.env.AlfworldEnv",
    "webshop": "envs.webshop.env.WebShopEnv",
    "sciworld": "envs.sciworld.env.SciWorldEnv",
}

def resolve_class_identifier(identifier: str, aliases: Dict[str, str], kind: str) -> str:
    """Resolve a possibly-short alias (e.g., 'human') to a full dotted class path.

    If `identifier` already looks like a dotted path, return it unchanged.
    Otherwise, look up a lowercase alias in `aliases`.
    """
    if not identifier:
        raise ValueError(f"Empty {kind} identifier")
    if "." in identifier:
        return identifier
    key = identifier.strip().lower()
    if key in aliases:
        return aliases[key]
    available = ", ".join(sorted(aliases.keys()))
    raise ValueError(f"Unknown {kind} alias '{identifier}'. Available: {available}")

def _default_run_id(agent_path: str, env_path: str) -> str:
    """Generate default run_id = <timestamp>_<AgentCls>_<EnvCls>."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_cls_name = agent_path.split(".")[-1]
    env_cls_name = env_path.split(".")[-1]
    return f"{ts}_{agent_cls_name}_{env_cls_name}"

def create_instance(cls: Type, running_config: Optional[Dict[str, Any]], logger: Optional[SimpleLogger]):
    """Instantiate a class, injecting logger and config-defined constructor kwargs."""
    sig = inspect.signature(cls)
    kwargs: Dict[str, Any] = {}

    if logger is not None and "logger" in sig.parameters:
        kwargs["logger"] = logger

    for k in running_config or {}:
        if k in sig.parameters and k not in kwargs:
            kwargs[k] = running_config[k]

    if "task_type" in sig.parameters and running_config and "task_types" in running_config:
        task_types = running_config.get("task_types", [])
        if isinstance(task_types, list) and task_types:
            kwargs["task_type"] = task_types[0].upper()
        elif isinstance(task_types, str):
            kwargs["task_type"] = task_types.upper()

    try:
        return cls(**kwargs)  # type: ignore[arg-type]
    except TypeError:
        return cls()


def load_class(path: str) -> Type:
    """Import a class given a dotted path "package.module.Class" only."""
    try:
        module_path, class_name = path.rsplit(".", 1)
    except ValueError:
        raise ValueError(f"Invalid class path '{path}'. Expected format: package.module.Class")

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if not isinstance(cls, type):
        raise AttributeError(f"'{path}' does not resolve to a class")
    return cls


def _safe_report(obj: Any) -> Dict[str, Any]:
    """Call obj.report() if available and return a dict; otherwise return {}."""
    try:
        if hasattr(obj, "report") and callable(getattr(obj, "report")):
            data = getattr(obj, "report")() or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _assemble_result(
    agent: Agent,
    env: Env,
    instance_id: Optional[int],
    duration: float,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the unified result dict from agent/env reports plus local info."""
    agent_report = _safe_report(agent)
    # print(f"[HERE] {agent_report}")
    env_report = _safe_report(env)
    # Ensure task_type present if env exposes it
    if hasattr(env, "task_type") and "task_type" not in env_report:
        try:
            env_report["task_type"] = getattr(env, "task_type")
        except Exception:
            pass
    local_info: Dict[str, Any] = {
        "instance_id": instance_id,
        "time": duration,
    }
    if error is not None:
        local_info["error"] = error
    return {**agent_report, **env_report, **local_info}

async def run_single_instance(
    agent: Agent,
    env: Env,
    config: Dict[str, Any],
    logger: SimpleLogger,
    instance_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one episode and collect result dict (async)."""

    # Determine per-instance time limit (seconds). Default to 900s if unspecified.
    try:
        max_duration_cfg = config.get("max_duration", 900)
        time_limit_secs = float(max_duration_cfg if max_duration_cfg is not None else 900)
        if time_limit_secs <= 0:
            time_limit_secs = 900.0
    except Exception:
        time_limit_secs = 900.0

    init_info = env.reset(config, str(instance_id) if instance_id is not None else None)
    observations = init_info["observations"]
    agent.reset(config, init_info)
    logger.info(f"[Instance {instance_id}] Environment reset. Starting episode.")

    start_time = time.time()

    async def episode_runner() -> Dict[str, Any]:
        nonlocal observations
        try:
            while not env.is_done():
                actions = await agent.act(observations)
                observations = await env.run(actions)

            success = env.is_success()
            duration_local = time.time() - start_time
            final_steps_local = env._step_count

            logger.info(
                f"{env.id}-Finished: {'SUCCESS' if success else 'FAILURE'} "
                f"({final_steps_local} steps, {duration_local:.4f}s)"
            )

            return _assemble_result(agent, env, instance_id, duration_local)

        except StepLimitError as e:
            duration_local = time.time() - start_time
            final_steps_local = env._step_count
            logger.warning(f"[Instance {instance_id}] {e} ({final_steps_local} steps, {duration_local:.4f}s)")

            return _assemble_result(agent, env, instance_id, duration_local, error=str(e))

        except Exception as e:
            duration_local = time.time() - start_time
            try:
                final_steps_local = env.get_step_count()
            except Exception:
                final_steps_local = getattr(env, "_step_count", 0)
            logger.error(f"{env.id}-ERROR: {e} ({final_steps_local} steps, {duration_local:.4f}s)")

            return _assemble_result(agent, env, instance_id, duration_local, error=str(e))

    try:
        result = await asyncio.wait_for(episode_runner(), timeout=time_limit_secs)
        return result
    except asyncio.TimeoutError:
        duration = time.time() - start_time
        try:
            final_steps = env.get_step_count()
        except Exception:
            final_steps = getattr(env, "_step_count", 0)
        logger.warning(
            f"[Instance {instance_id}] TIMEOUT after {int(time_limit_secs)}s "
            f"({final_steps} steps, {duration:.4f}s)"
        )
        res = _assemble_result(
            agent, env, instance_id, duration, error=f"Timeout after {int(time_limit_secs)}s"
        )
        # Explicitly mark as failure to ensure correct final statistics
        res["success"] = False
        return res


async def run_concurrent_instances(
    agent_cls: Type[Agent],
    env_cls: Type[Env],
    num_instances: int,
    max_concurrent: int = 10,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[SimpleLogger] = None,
) -> List[Dict[str, Any]]:
    """Run many environment instances concurrently with a live progress UI.

    If `rich` is available, use a richer UI with per-instance spinners; otherwise
    fallback to tqdm-based overall bar plus lightweight per-instance lines.
    """

    config = config or {}
    # Determine base start id from running_config
    try:
        base_start_id = int(config.get("start_id", 0) or 0)
    except (TypeError, ValueError):
        base_start_id = 0

    sem = asyncio.Semaphore(max_concurrent)

    # Decide whether to use any progress UI. Allow config to forcibly disable it
    # (e.g., for HumanAgent which reads from stdin and conflicts with live updating UIs).
    disable_rich_ui = False
    try:
        # Accept multiple possible keys to disable rich UI
        for key in ("disable_rich_ui", "no_rich", "disable_rich"):
            v = config.get(key)
            if isinstance(v, str):
                v_norm = v.strip().lower()
                if v_norm in ("1", "true", "yes", "y", "on"):  # treat truthy strings as True
                    disable_rich_ui = True
                    break
            elif v:
                disable_rich_ui = True
                break
    except Exception:
        disable_rich_ui = False

    # Also disable UI if HumanAgent is used to avoid interfering with stdin
    try:
        if getattr(agent_cls, "__name__", "") == "HumanAgent":
            disable_rich_ui = True
    except Exception:
        pass

    use_rich = False
    if not disable_rich_ui:
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn
            from rich.console import Group
            from rich.live import Live
            from rich.text import Text
            use_rich = True
        except Exception:
            use_rich = False

    # Common runner utilities -------------------------------------------------
    def make_instance_logger(effective_id: int):
        instance_logger = None
        if logger is not None:
            import logging
            instance_logger_name = f"instance_{effective_id}_{logger.run_id}"
            instance_logger_obj = logging.getLogger(instance_logger_name)
            instance_logger_obj.setLevel(logging.INFO)
            instance_logger_obj.handlers.clear()
            instance_log_file = Path(logger.get_log_dir()) / f"instance_{effective_id}.log"
            file_handler = logging.FileHandler(instance_log_file, mode="w", encoding="utf-8")
            from utils.logger import MultiLineFormatter
            file_handler.setFormatter(MultiLineFormatter('%(asctime)s - %(levelname)s - %(message)s'))
            instance_logger_obj.addHandler(file_handler)

            class InstanceLogger:
                def __init__(self, logger_obj, main_logger):
                    self.logger = logger_obj
                    self.main_logger = main_logger
                    self.run_id = main_logger.run_id
                def info(self, message):
                    self.logger.info(message)
                def warning(self, message):
                    self.logger.warning(message)
                def error(self, message):
                    self.logger.error(message)
                def get_log_dir(self):
                    return self.main_logger.get_log_dir()
                def get_base_dir(self):
                    return self.main_logger.get_base_dir()

            instance_logger = InstanceLogger(instance_logger_obj, logger)
        return instance_logger or logger

    # No-UI branch (for HumanAgent or when explicitly disabled) --------------
    if disable_rich_ui:
        results: List[Dict[str, Any]] = []

        for instance_id in range(num_instances):
            effective_id = base_start_id + instance_id
            plogger = make_instance_logger(effective_id)
            agent = create_instance(agent_cls, config, plogger)
            env = create_instance(env_cls, config, plogger)
            res = await run_single_instance(agent, env, config, plogger, effective_id)
            results.append(res)

        return results

    # Rich UI branch ----------------------------------------------------------
    if use_rich:
        results: List[Dict[str, Any]] = []

        overall_progress = Progress(
            TextColumn("[bold]Overall[/bold]"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            refresh_per_second=8,
        )

        instances_progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}[/bold]"),
            TextColumn("{task.fields[status]}", style="dim"),
            refresh_per_second=8,
        )

        instance_tasks: Dict[int, int] = {}
        finished_names: List[str] = []

        async def runner(instance_id: int):
            async with sem:
                effective_id = base_start_id + instance_id
                plogger = make_instance_logger(effective_id)
                agent = create_instance(agent_cls, config, plogger)
                env = create_instance(env_cls, config, plogger)

                # Add a per-instance spinner task
                task_id = instances_progress.add_task(f"instance {effective_id}", status="running")
                instance_tasks[effective_id] = task_id
                try:
                    res = await run_single_instance(agent, env, config, plogger, effective_id)
                    overall_progress.update(overall_task, advance=1)
                    # Remove finished task from running list and update Done line
                    try:
                        # Prefer hiding the task to avoid accumulating many visible lines
                        instances_progress.update(task_id, status="done", visible=True)
                        # Immediately hide the finished task for a clean UI
                        instances_progress.update(task_id, visible=False)
                    except Exception:
                        pass
                    try:
                        # Best-effort removal (not strictly required if hidden)
                        instances_progress.remove_task(task_id)
                    except Exception:
                        pass
                    finished_names.append(f"instance {effective_id}")
                    try:
                        done_renderable = Text("✔ Done: ", style="green")
                        if finished_names:
                            done_renderable.append(", ".join(finished_names))
                        live.update(Group(overall_progress, instances_progress, done_renderable))
                    except Exception:
                        pass
                    return res
                finally:
                    # Keep finished tasks displayed; just cleanup mapping
                    instance_tasks.pop(effective_id, None)
                    # Ensure any lingering task is hidden in case of earlier failure
                    try:
                        tid = instance_tasks.get(effective_id)
                        if tid is not None:
                            instances_progress.update(tid, visible=False)
                    except Exception:
                        pass

        with Live(Group(overall_progress, instances_progress, Text("✔ Done: ", style="green")), refresh_per_second=8, transient=False) as live:
            overall_task = overall_progress.add_task("Instances", total=num_instances)
            tasks = [asyncio.create_task(runner(i)) for i in range(num_instances)]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, res in enumerate(raw_results):
            if isinstance(res, Exception):
                if logger:
                    logger.error(f"Instance {idx} raised exception: {res}")
                else:
                    print(f"Instance {idx} raised exception: {res}")
                results.append({"instance_id": idx, "success": False, "error": str(res)})
            else:
                results.append(res)  # type: ignore[arg-type]

        return results

    # Fallback tqdm branch ----------------------------------------------------
    # Main overall progress bar (position 0)
    progress_bar = tqdm(total=num_instances, desc="Instances", leave=True)

    # Allocate fixed display slots for per-instance lightweight spinners
    slot_queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(max_concurrent):
        slot_queue.put_nowait(i)

    slot_bars = [
        tqdm(
            total=1,
            position=1 + i,
            leave=True,
            bar_format="{desc} {postfix}",
            dynamic_ncols=True,
        )
        for i in range(max_concurrent)
    ]
    for i, bar in enumerate(slot_bars):
        bar.set_description_str("[instance -]")
        bar.set_postfix_str("")

    active_slots: Dict[int, Dict[str, Any]] = {}
    stop_spinners = asyncio.Event()

    async def spinner_updater():
        spinner_chars = ["|", "/", "-", "\\"]
        idx = 0
        try:
            while not stop_spinners.is_set():
                for slot, meta in list(active_slots.items()):
                    bar = slot_bars[slot]
                    inst_id = meta.get("id")
                    bar.set_description_str(f"[instance {inst_id}]")
                    bar.set_postfix_str(f"running {spinner_chars[idx % len(spinner_chars)]}")
                    bar.refresh()
                idx += 1
                await asyncio.sleep(0.1)
        finally:
            for slot, meta in list(active_slots.items()):
                bar = slot_bars[slot]
                inst_id = meta.get("id")
                bar.set_description_str(f"[instance {inst_id}]")
                bar.set_postfix_str("done")
                bar.refresh()

    async def runner(instance_id: int):
        async with sem:
            effective_id = base_start_id + instance_id
            slot = await slot_queue.get()
            active_slots[slot] = {"id": effective_id}

            plogger = make_instance_logger(effective_id)
            agent = create_instance(agent_cls, config, plogger)
            env = create_instance(env_cls, config, plogger)
            try:
                result = await run_single_instance(agent, env, config, plogger, effective_id)
                progress_bar.update(1)
                return result
            finally:
                try:
                    bar = slot_bars[slot]
                    bar.set_description_str(f"[instance {effective_id}]")
                    bar.set_postfix_str("done")
                    bar.refresh()
                except Exception:
                    pass
                active_slots.pop(slot, None)
                slot_queue.put_nowait(slot)

    spinner_task = asyncio.create_task(spinner_updater())
    tasks = [asyncio.create_task(runner(i)) for i in range(num_instances)]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    stop_spinners.set()
    try:
        await spinner_task
    except Exception:
        pass
    progress_bar.close()
    for bar in slot_bars:
        try:
            bar.close()
        except Exception:
            pass

    results: List[Dict[str, Any]] = []
    for idx, res in enumerate(raw_results):
        if isinstance(res, Exception):
            if logger:
                logger.error(f"Instance {idx} raised exception: {res}")
            else:
                print(f"Instance {idx} raised exception: {res}")
            results.append({"instance_id": idx, "success": False, "error": str(res)})
        else:
            results.append(res)  # type: ignore[arg-type]

    return results


def write_summary(results: List[Dict[str, Any]], output_file: Path):
    """Write results summary to `output_file`, creating parent dirs."""

    total = len(results)
    successes = sum(1 for r in results if r.get("success"))
    # Per-task-type aggregation (only if task_type present)
    by_task: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if "task_type" not in r or r.get("task_type") is None:
            continue
        task_type = str(r.get("task_type"))
        bucket = by_task.setdefault(task_type, {
            "total_instances": 0,
            "successful_instances": 0,
            "total_time": 0.0,
            "total_steps": 0,
            "total_cost": 0.0,
            "total_reward": 0.0,
        })
        bucket["total_instances"] += 1
        if r.get("success"):
            bucket["successful_instances"] += 1
        bucket["total_time"] += float(r.get("time", 0.0))
        bucket["total_steps"] += int(r.get("steps", 0) or 0)
        bucket["total_cost"] += float(r.get("cost", 0.0))
        bucket["total_reward"] += float(r.get("reward", 0.0))
    # Compute averages per task
    for t, b in by_task.items():
        ti = b["total_instances"] or 1
        b["success_rate"] = b["successful_instances"] / ti
        b["avg_time_per_instance"] = b["total_time"] / ti
        b["avg_steps_per_instance"] = b["total_steps"] / ti
        b["avg_cost_per_instance"] = b["total_cost"] / ti

    # Dynamically aggregate all numeric-like metrics (totals and averages)
    # Exclude only 'success' to avoid double counting in metrics,
    # and exclude non-meaningful fields like instance_id
    numeric_keys = set()
    for r in results:
        for k, v in r.items():
            if k in ("instance_id", "success"):
                continue
            if isinstance(v, (int, float, bool)):
                numeric_keys.add(k)

    metrics_total: Dict[str, float] = {}
    metrics_avg: Dict[str, float] = {}
    for k in sorted(numeric_keys):
        s = 0.0
        for r in results:
            try:
                val = r.get(k, 0)
                s += float(val or 0)
            except Exception:
                continue
        metrics_total[k] = s
        metrics_avg[k] = (s / total) if total > 0 else 0.0

    # Per-task dynamic metrics
    by_task_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    if by_task:
        for task_type in by_task.keys():
            totals: Dict[str, float] = {}
            avgs: Dict[str, float] = {}
            bucket_results = [r for r in results if str(r.get("task_type")) == task_type]
            bucket_n = len(bucket_results) or 1
            for k in sorted(numeric_keys):
                s = 0.0
                for r in bucket_results:
                    try:
                        val = r.get(k, 0)
                        s += float(val or 0)
                    except Exception:
                        continue
                totals[k] = s
                avgs[k] = s / bucket_n
            by_task_metrics[task_type] = {"metrics_total": totals, "metrics_avg": avgs}

    summary = {
        "summary": {
            "total_instances": total,
            "successful_instances": successes,
            "success_rate": successes / total if total > 0 else 0,
            "metrics_total": metrics_total,
            "metrics_avg": metrics_avg,
        },
        "instances": results,
    }
    if by_task:
        # merge base by_task stats with dynamic metrics
        merged_by_task: Dict[str, Any] = {}
        for t, base_stats in by_task.items():
            merged = dict(base_stats)
            if t in by_task_metrics:
                merged.update(by_task_metrics[t])
            merged_by_task[t] = merged
        summary["by_task_type"] = merged_by_task

    output_file.parent.mkdir(parents=True, exist_ok=True)
    import json
    output_file.write_text(json.dumps(summary, indent=2))

    print("\n📊 Summary:")
    rate_pct = (successes / total * 100.0) if total > 0 else 0.0
    print(f"   Success: {successes}/{total} ({rate_pct:.4f}%)")
    print(f"   Results saved to: {output_file}")
    # Print per-task breakdown if any
    if by_task:
        print("   By task_type:")
        for t, b in by_task.items():
            rpct = b["success_rate"] * 100.0
            print(f"     - {t}: {b['successful_instances']}/{b['total_instances']} ({rpct:.4f}%)")
    # Print standard metrics if present
    standard_keys_order = ["time", "steps", "cost", "reward"]
    std_present = [k for k in standard_keys_order if k in metrics_total]
    if std_present:
        print("   Metrics (totals/avg):")
        for k in std_present:
            total_v = metrics_total[k]
            avg_v = metrics_avg[k]
            try:
                print(f"     - {k}: total={total_v:.4f}, avg={avg_v:.4f}")
            except Exception:
                print(f"     - {k}: total={total_v}, avg={avg_v}")
    # Print any additional numeric metrics not already shown (excluding 'success')
    excluded_keys = set(std_present)
    extra_keys = [k for k in metrics_total.keys() if k not in excluded_keys]
    if extra_keys:
        print("   Extra metrics (totals/avg):")
        for k in extra_keys:
            total_v = metrics_total[k]
            avg_v = metrics_avg[k]
            try:
                print(f"     - {k}: total={total_v:.4f}, avg={avg_v:.4f}")
            except Exception:
                print(f"     - {k}: total={total_v}, avg={avg_v}")


def main():
    parser = argparse.ArgumentParser(
        description="Run an agent in an environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-a", "--agent",
        type=str,
        default="agents.recode.agent.ReCodeAgent",
        help="Agent class path or alias. Examples: agents.recode.agent.ReCodeAgent | aliases: human, recode, react, codeact, adaplanner",
    )
    parser.add_argument(
        "-e", "--env",
        type=str,
        default="envs.alfworld.env.AlfworldEnv",
        help="Environment class path or alias. Examples: envs.alfworld.env.AlfworldEnv | aliases: alfworld, webshop, sciworld, travelplanner",
    )
    parser.add_argument(
        "-n", "--instances",
        type=int,
        default=1,
        help="Number of instances to run",
    )
    parser.add_argument(
        "-c", "--concurrent",
        type=int,
        default=1,
        help="Maximum concurrent instances",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="results.json",
        help="Results JSON filename (will be saved in logs/<log_dir>/)",
    )
    parser.add_argument(
        "-C", "--config",
        type=str,
        default=None,
        help="YAML config file path. Values here override CLI flags.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (e.g., train/valid/test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed forwarded to environments",
    )
    parser.add_argument(
        "-p", "--profile",
        type=str,
        default=None,
        help="LLM profile name forwarded to the agent",
    )
    parser.add_argument(
        "-l", "--log-dir",
        type=str,
        default=None,
        help="Custom log directory name (otherwise autogenerated)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth for agent execution",
    )

    args = parser.parse_args()

    try:
        # Load YAML config (overrides CLI)
        import yaml
        yaml_cfg = {}
        if args.config:
            try:
                with open(args.config) as f:
                    yaml_cfg = yaml.safe_load(f) or {}
            except FileNotFoundError:
                print(f"⚠️  Config file not found: {args.config}. Using CLI values only.")
                yaml_cfg = {}

        # Compose final config: CLI base, YAML overrides
        cli_cfg: Dict[str, Any] = {
            "agent": args.agent,
            "env": args.env,
            "instances": args.instances,
            "concurrent": args.concurrent,
            "output": args.output,
            "log_dir": args.log_dir,
            "split": args.split,
            "seed": args.seed,
            "profile": args.profile,
            "max_depth": args.max_depth,
        }
        config: Dict[str, Any] = {**cli_cfg, **yaml_cfg}

        agent_path: str = config.get("agent", args.agent)
        env_path: str = config.get("env", args.env)
        instances: int = int(config.get("instances", args.instances) or 1)
        concurrent: int = int(config.get("concurrent", args.concurrent) or 1)
        output_name: str = str(config.get("output", args.output))

        # Resolve short aliases if provided
        agent_path = resolve_class_identifier(agent_path, AGENT_ALIASES, "agent")
        env_path = resolve_class_identifier(env_path, ENV_ALIASES, "env")

        agent_cls = load_class(agent_path)
        env_cls = load_class(env_path)

        # Use class names for default run_id for readability
        run_id = config.get("log_dir") or _default_run_id(agent_cls.__name__, env_cls.__name__)
        # Clear existing log directory if present
        existing_base_dir = Path("logs") / run_id
        if existing_base_dir.exists():
            try:
                shutil.rmtree(existing_base_dir)
            except Exception as e:
                print(f"⚠️  Failed to clear existing log directory: {existing_base_dir} ({e})")
        logger = SimpleLogger(run_id=run_id)

        # Special handling for HumanAgent: disable Rich UI and force concurrency to 1
        is_human_agent = (getattr(agent_cls, "__name__", "") == "HumanAgent") or agent_path.endswith(".HumanAgent")
        if is_human_agent:
            if concurrent != 1:
                logger.info(f"Human agent detected. Forcing max concurrent to 1 (was {concurrent}).")
            concurrent = 1
            config["concurrent"] = 1
            config["disable_rich_ui"] = True

        logger.info(f"🤖 Agent: {agent_path}")
        logger.info(f"🌍 Environment: {env_path}")
        logger.info(f"📊 Instances: {instances} (max {concurrent} concurrent)")
        logger.info("-" * 50)

        results = asyncio.run(
            run_concurrent_instances(agent_cls, env_cls, instances, concurrent, config, logger)
        )

        output_file = logger.get_base_dir() / output_name
        write_summary(results, output_file)

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        # import traceback
        # traceback.print_exc()
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 