import contextlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

import yaml
from base.environment import Env
from utils.errors import StepLimitError

class SciWorldEnv(Env):
    """
    An environment wrapper for ScienceWorld to conform to the base `Env` interface.
    """
    env_name = "sciworld"
    # Shared cache per data_root_dir to avoid repeated file reads across instances
    _shared_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        config_path: str = "envs/sciworld/base_config.yaml",
        simplification: str = "easy",
        logger: Optional[Any] = None,
    ) -> None:
        self.simplification = simplification
        self.logger = logger
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}

        self.data_root_dir = Path(self.config['data_root_dir'])
        root_key = str(self.data_root_dir.resolve())
        cache = SciWorldEnv._shared_cache.get(root_key)
        if cache is None:
            cache = {
                "taskname2id": json.load(open(self.data_root_dir / "taskname2id.json")),
                "max_steps": json.load(open(self.data_root_dir / "max_steps.json")),
                "indices_by_split": {},
            }
            SciWorldEnv._shared_cache[root_key] = cache
        self.taskname2id = cache["taskname2id"]
        self.max_steps_dict = cache["max_steps"]

    def _initialize(self) -> None:
        """Initialize the ScienceWorld environment."""
        if self.logger:
            self.logger.info("Initializing ScienceWorld environment")
        try:
            import scienceworld
        except ImportError as e:
            raise ImportError(
                "The 'scienceworld' library is required to use SciWorldEnv. "
                "Please install it with 'pip install scienceworld'."
            ) from e

        # Suppress verbose output from the ScienceWorld library
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            self.env = scienceworld.ScienceWorldEnv(
                serverPath=None,
                envStepLimit=np.inf
            )

        if self.logger:
            self.logger.info("ScienceWorld environment initialized successfully")

    def _load_indices(self, split: str, seed: int) -> List[int]:
        """Load the indices for the given split (cached across instances)."""
        root_key = str(self.data_root_dir.resolve())
        cache = SciWorldEnv._shared_cache.get(root_key)
        if cache is None:
            cache = {
                "taskname2id": getattr(self, "taskname2id", {}),
                "max_steps": getattr(self, "max_steps_dict", {}),
                "indices_by_split": {},
            }
            SciWorldEnv._shared_cache[root_key] = cache
        indices_cache = cache.setdefault("indices_by_split", {})
        if split not in indices_cache:
            if split == "validation":
                split = "valid"
            with open(self.data_root_dir / f"{split}_indices.json", "r") as f:
                indices_cache[split] = json.load(f)

        random.seed(seed)
        random.shuffle(indices_cache[split])
        return indices_cache[split]

    def reset(self, running_config: dict, id: Optional[str] = None) -> dict:
        """
        Reset environment to its initial state.
        The `id` can be used to set the variation for the task.
        """
        self._initialize()
        if self.env is None:
            raise RuntimeError("Environment could not be initialized.")
        
        self.split = running_config.get("split", "train")
        seed = running_config.get("seed", 42)
        self.indices = self._load_indices(self.split, seed)

        id_int: Optional[int] = None
        if id is not None:
            try:
                id_int = int(id)
            except ValueError:
                raise ValueError(f"Task ID '{id}' is not a valid integer.")
            if not 0 <= id_int < len(self.indices):
                raise ValueError(
                    f"Task ID {id_int} is out of valid range (0-{len(self.indices) - 1})."
                )
            self.task_name, self.variation = self.indices[id_int]

        self.id = f"{self.taskname2id[self.task_name]}_{self.variation}"
        self.max_steps = self.max_steps_dict[self.task_name]

        self.env.load(self.task_name, self.variation, self.simplification, generateGoldPath=False)
        obs, info = self.env.reset()

        self._step_count = 0
        self._done = False
        self._success = False
        self._reward = 0.0
        task_description = info.get("taskDesc", "No task description found.")
        observation = f"{task_description}\n{obs}"

        return {"observations": [observation], "env_name": self.env_name, "env": self}

    async def _run(self, single_action: str) -> str:
        """Execute one action against the underlying ScienceWorld env."""

        if not single_action:
            return ""

        # Early exit if already terminated
        if self._done:
            return "The environment has already terminated."

        if single_action.strip().lower() == "[finish]":
            self._done = True
            return "You have finished the task." if self._success else "Task failed."

        # Increment step counter before executing
        self._step_count += 1
        if self.max_steps and self._step_count > self.max_steps:
            self._done = True
            raise StepLimitError(f"Step limit of {self.max_steps} exceeded.")

        try:
            obs, _, done, info = self.env.step(single_action)
            self.logger.info(f"[Score] {info['score']}")
            # self.logger.info(f"[Info]: {info}")
            self._reward = info['score'] if info['score'] is not None and info['score'] > self._reward else self._reward
            self._done = done
            if info['score'] > 0 and done:
                self._success = True
            return obs
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error executing action '{single_action}': {e}")
            self._done = True
            self._success = False
            raise

    def is_done(self) -> bool:
        """Check if the environment is done."""
        return self._done

    # Provide current step count for external callers (e.g., run.py)
    def get_step_count(self) -> int:
        return self._step_count

    def is_success(self) -> bool:
        """Check if the task was successfully completed."""
        return self._success

    def report(self):
        return {
            "success": self._success,
            "step": self._step_count,
            "reward": self._reward,
            "task_type": self.task_name,
        }

    async def close(self) -> None:
        """Close the ScienceWorld environment and clean up resources."""
        if self.logger:
            self.logger.info("Closing ScienceWorld environment")
        
        try:
            # Clean up the ScienceWorld environment if it exists
            if hasattr(self, 'env') and self.env is not None:
                # ScienceWorld doesn't have an explicit close method, but we can clean up references
                self.env = None
                
            # Reset state variables
            self._step_count = 0
            self._done = False
            self._success = False
            self._reward = 0.0
            
            if self.logger:
                self.logger.info("ScienceWorld environment closed successfully")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error closing ScienceWorld environment: {e}")
            raise