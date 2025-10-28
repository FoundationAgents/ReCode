import contextlib
import glob
import os
import re
from typing import Any, Dict, List, Optional, Union, Tuple

import yaml
from alfworld.agents.environment import get_environment

from base.environment import Env
from utils.errors import StepLimitError

import random

# Provide a sensible default if the user has not set $ALFWORLD_DATA
DEFAULT_ALFWORLD_DATA = os.path.expanduser("~/.cache/alfworld")
if "ALFWORLD_DATA" not in os.environ:
    os.environ["ALFWORLD_DATA"] = DEFAULT_ALFWORLD_DATA

prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

DEFAULT_MAX_STEPS = 50

class AlfworldEnv(Env):
    """A fully-featured ALFWorld environment that conforms to the base Env interface."""
    env_name = "alfworld"
    _cached_game_files: Dict[Tuple[str, str, Optional[Tuple[str, ...]]], List[str]] = {}

    def __init__(
        self,
        base_config_path: str = "envs/alfworld/base_config.yaml",
        # split: str = "train",
        specific_game_file: Optional[str] = None,
        task_types: Optional[List[str]] = None,
        logger: Optional[Any] = None,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS,
    ) -> None:
        self.base_config_path = base_config_path
        # self.split = split
        self.specific_game_file = specific_game_file
        self.logger = logger  # Accepts any logger with an `info` method
        self.task_types: Optional[List[str]] = [t.lower() for t in task_types] if task_types else None

        self.max_steps: Optional[int] = max_steps
        self._step_count: int = 0

        self.env: Optional[Any] = None  # Underlying ALFWorld env
        self.game_files: Optional[List[str]] = None
        self.game_name: str = "unknown_game"
        self._done: bool = False
        self._success: bool = False

    def _get_game_files(self, seed: int = 42) -> List[str]:
        """Get a sorted list of all game files for the current split."""
        if self.game_files is not None:
            return self.game_files

        cache_key: Tuple[str, str, Optional[Tuple[str, ...]]] = (
            os.path.abspath(self.base_config_path),
            self.split,
            tuple(sorted(self.task_types)) if self.task_types else None,
        )
        if cache_key in AlfworldEnv._cached_game_files:
            self.game_files = AlfworldEnv._cached_game_files[cache_key]
            return self.game_files

        with open(self.base_config_path) as reader:
            config = yaml.safe_load(reader)

        if self.split == "test":
            data_path_key = "eval_ood_data_path"
        elif self.split == "valid":
            data_path_key = "eval_id_data_path"
        else:
            data_path_key = "data_path"

        data_path = config["dataset"].get(data_path_key)
        if data_path:
            data_path = os.path.expandvars(data_path)

        if not data_path or not os.path.isdir(data_path):
            raise FileNotFoundError(f"Data path for split '{self.split}' not found or is not a valid directory: {data_path}")
        
        search_path = os.path.join(data_path, "**", "traj_data.json")
        game_files = glob.glob(search_path, recursive=True)

        if self.task_types:
            def _extract_mapped_task_type(path: str) -> Optional[str]:
                try:
                    parts = os.path.normpath(path).split(os.sep)
                    task_dir = parts[-3].lower()
                except Exception:
                    return None

                for k, v in prefixes.items():
                    if task_dir.startswith(k):
                        return v
                return task_dir

            filtered: List[str] = []
            for gf in game_files:
                mapped = _extract_mapped_task_type(gf)
                if mapped and mapped in self.task_types:
                    filtered.append(gf)

            game_files = filtered

        if self.logger and not game_files:
            self.logger.warning(
                f"No game files found for split '{self.split}' at path '{search_path}'"
                + (f" after applying task_type filter {self.task_types}" if self.task_types else "")
            )

        filtered_with_pddl: List[str] = []
        missing_pddl_count = 0
        for traj_path in game_files:
            pddl_path = traj_path.replace("traj_data.json", "game.tw-pddl")
            if os.path.exists(pddl_path):
                filtered_with_pddl.append(traj_path)
            else:
                missing_pddl_count += 1

        game_files = filtered_with_pddl

        if self.logger and missing_pddl_count:
            self.logger.info(
                f"Skipped {missing_pddl_count} game(s) without corresponding game.tw-pddl files."
            )

        random.seed(seed)
        random.shuffle(game_files) 

        self.game_files = game_files
        AlfworldEnv._cached_game_files[cache_key] = game_files
        return self.game_files

    def _normalize_split_for_alfworld(self, split: str) -> str:
        """Map user/config split names to ALFWorld's expected names."""
        s = (split or "train").lower()
        if s in {"valid", "valid_seen", "eval_id", "eval_in_distribution"}:
            return "eval_in_distribution"
        if s in {"test", "valid_unseen", "eval_ood", "eval_out_of_distribution"}:
            return "eval_out_of_distribution"
        return "train"

    def _initialize(self) -> None:
        """Initialize the ALFWorld environment, optionally targeting a specific game file."""
        normalized_split = self._normalize_split_for_alfworld(self.split)

        if self.logger:
            self.logger.info(f"Initializing ALFWorld environment with split: {normalized_split}")
            if self.specific_game_file:
                self.logger.info(f"Target game file: {self.specific_game_file}")

        with open(self.base_config_path) as reader:
            config = yaml.safe_load(reader)

        env_type = config["env"]["type"]
        env_class = get_environment(env_type)

        if self.specific_game_file:
            self._configure_for_specific_game(config, self.specific_game_file)
            # Provide external game files list to avoid ALFWorld scanning on init
            pddl_game_file = self.specific_game_file.replace("traj_data.json", "game.tw-pddl")
            config.setdefault("env", {})
            config["env"]["external_game_files"] = [pddl_game_file]
        elif self.task_types:
            # Precompute filtered PDDL files and pass to ALFWorld to skip scanning
            filtered_traj_files = self._get_game_files()
            filtered_pddl_files = [f.replace("traj_data.json", "game.tw-pddl") for f in filtered_traj_files]
            config.setdefault("env", {})
            config["env"]["external_game_files"] = filtered_pddl_files

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(
            devnull
        ), contextlib.redirect_stderr(devnull):
            alfworld_env = env_class(config, train_eval=normalized_split)

            # Backward-compatibility: explicitly set game_files on the ALFWorld env instance
            # so that even if the package doesn't support external_game_files, we still avoid rescans
            if self.specific_game_file:
                pddl_game_file = self.specific_game_file.replace("traj_data.json", "game.tw-pddl")
                alfworld_env.game_files = [pddl_game_file]
                alfworld_env.num_games = 1
            elif self.task_types:
                filtered_traj_files = self._get_game_files()
                filtered_pddl_files = [f.replace("traj_data.json", "game.tw-pddl") for f in filtered_traj_files]
                alfworld_env.game_files = filtered_pddl_files
                alfworld_env.num_games = len(filtered_pddl_files)
                if self.logger:
                    self.logger.info(
                        f"Task-type filter active. Loaded {len(filtered_pddl_files)} games for types {self.task_types}."
                    )

            self.env = alfworld_env.init_env(batch_size=1)

        if self.logger:
            self.logger.info("ALFWorld environment initialized successfully")

    def _configure_for_specific_game(self, config: Dict[str, Any], game_file: str) -> None:
        """Modify the config to load a specific game file."""
        if not os.path.exists(game_file):
            raise FileNotFoundError(f"Specific game file not found: {game_file}")

        if self.split == "eval_out_of_distribution":
            data_path_key = "eval_ood_data_path"
        elif self.split == "eval_in_distribution":
            data_path_key = "eval_id_data_path"
        else:
            data_path_key = "data_path"

        split_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(game_file)))
        config["dataset"][data_path_key] = split_root_dir

        num_games_key = data_path_key.replace("data_path", "num_games").replace("eval_id", "num_eval").replace("eval_ood", "num_eval")
        if num_games_key in config["dataset"]:
            del config["dataset"][num_games_key]

        pddl_game_file = game_file.replace("traj_data.json", "game.tw-pddl")
        if not os.path.exists(pddl_game_file):
            if self.logger:
                self.logger.warning(
                    f"PDDL file not found for {game_file}. Enabling regen_game_files so it will be generated."
                )

            if "env" not in config:
                config["env"] = {}

            config["env"]["regen_game_files"] = True

    def reset(self, running_config: dict, id: Optional[str] = None) -> dict:
        """Reset environment to initial state and return the first observation."""
        if self.logger:
            self.logger.info("Resetting ALFWorld environment")
        
        seed = running_config.get("seed", 42) if running_config else 42
        task_type_filter = running_config.get("task_type", None) if running_config else None
        self.split = running_config.get("split", "train") if running_config else "train"
        self.id = id
        id_int: Optional[int] = None

        if id is not None:
            try:
                id_int = int(id)
            except ValueError:
                raise ValueError(f"Task ID '{id}' is not a valid integer.")
            if self.game_files is None:
                self.game_files = self._get_game_files(seed)
            if not 0 <= id_int < len(self.game_files):
                raise ValueError(
                    f"Task ID {id_int} is out of valid range (0-{len(self.game_files) - 1})."
                )
            if task_type_filter:
                game_files = []
                for game_file in self.game_files:
                    task_type = game_file.split("/")[-3]
                    if task_type in task_type_filter:
                        game_files.append(game_file)
                self.game_files = game_files

            self.specific_game_file = self.game_files[id_int]
            self.task_type = self.specific_game_file.split("/")[-3]
            for k, v in prefixes.items():
                if self.task_type.startswith(k):
                    self.task_type = v
                    break
            
            if self.logger:
                self.logger.info(f"Task type: {self.task_type}")
            self.env = None  # Force re-initialization for the specific game
            if self.logger:
                self.logger.info(f"Set to run specific game file for ID {id}: {self.specific_game_file}")

        if self.env is None:
            self._initialize()

        if self.env is None:
            raise ValueError("Environment could not be initialized.")

        ob_raw, info_raw = self.env.reset()
        # Reset step counter and status flags on env reset
        self._step_count = 0
        self._done = False
        self._success = False

        # Extract game name for logging/debugging
        self.game_name = "unknown_game"
        if "extra.gamefile" in info_raw and info_raw["extra.gamefile"]:
            try:
                self.game_name = "/".join(info_raw["extra.gamefile"][0].split("/")[-3:-1])
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Could not parse game name from info: {e}")

        # Process observation for the agent
        obs = "\n".join(ob_raw[0].split("\n\n")[1:])
        # self.logger.info(f"[Observation ENV] {obs}")
        # Return unified reset format
        return {"observations": [obs], "task_type": self.task_type, "env_name": self.env_name, "env": self}

    def set_max_steps(self, max_steps: int) -> None:
        self.max_steps = max_steps

    async def _run(self, single_action: str) -> str:
        """Execute a *single* action and return the processed observation string."""
        # self.logger.info(f"Running action: {single_action}")

        if not single_action:
            return ""

        if single_action.strip() == "[FINISH]":
            self._done = True
            return "Episode terminated by agent."

        if self._done:
            return "The environment has already terminated."

        self._step_count += 1
        if self.max_steps is not None and self._step_count > self.max_steps:
            self._done = True
            raise StepLimitError(f"Step limit of {self.max_steps} exceeded.")

        pattern = r"^(put\s+\S+(?:\s+\S+)*\s+)(in|on)(\s+\S+(?:\s+\S+)*)$"
        match = re.match(pattern, single_action.strip())
        if match:
            single_action = f"{match.group(1)}in/on{match.group(3)}"

        def _process_ob(ob: str) -> str:
            if ob.startswith('You arrive at loc '):
                ob = ob[ob.find('. ')+2:]
            return ob

        try:
            obs_raw, _, done, info = self.env.step([single_action])
            processed_obs = _process_ob(obs_raw[0])
            self._done = bool(done[0])
            self._success = "won" in info and bool(info["won"][0])
            return processed_obs
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error executing command '{single_action}': {e}")
            self._done = True
            self._success = False
            return f"Error: {e}"
        
    def report(self) -> dict:
        return {
            "success": self._success,
            "steps": self._step_count,
            "task_type": self.task_type,
            "reward": int(self._success)
        }

    async def close(self) -> None:
        """Close the ALFWorld environment and clean up resources."""
        if self.logger:
            self.logger.info("Closing ALFWorld environment")
        
        try:
            # Clean up the ALFWorld environment if it exists
            if hasattr(self, 'env') and self.env is not None:
                # ALFWorld environment cleanup
                self.env = None
                
            # Reset state variables
            self._step_count = 0
            self._done = False
            self._success = False
            self.game_files = None
            self.game_name = "unknown_game"
            
            if self.logger:
                self.logger.info("ALFWorld environment closed successfully")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error closing ALFWorld environment: {e}")
            raise