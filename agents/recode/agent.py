from __future__ import annotations

from pathlib import Path
from enum import Enum
from typing import List, Optional
from datetime import datetime, timezone

from base.agent import Agent
from utils.llm import AsyncLLM
from utils.executor import Executor
from utils.common import parse_xml_tag

from agents.recode.resources.prompts.default_new import EXPAND_PROMPT
from agents.recode.utils import (
    parse_raw_observation,
    split_blocks,
    validate_blocks,
    NodeStatus,
    CodeNode,
    get_variables,
)

DEFAULT_MAX_DEPTH = 10
DEFAULT_MAX_RETRY = 5
DEFAULT_MAX_REWRITE = 5


class ReCodeAgent(Agent):
    def __init__(
        self,
        logger=None,
        task_type: str = None,
    ) -> None:
        self.logger = logger
        self.llm = AsyncLLM()
        self.executor = Executor(if_run_print=True)

        self.root: Optional[CodeNode] = None
        self.current_node: Optional[CodeNode] = None
        self.previous_node: Optional[CodeNode] = None
        self.task_type: str = task_type
        self.is_start = False
 
    def reset(self, running_config: dict, init_info: dict=None) -> None:
        self.root = None
        self.current_node = None
        self.previous_node = None
        self.is_start = False

        self.max_depth: int = running_config.get('max_depth', DEFAULT_MAX_DEPTH)
        self.max_retry: int = running_config.get('max_retry', DEFAULT_MAX_RETRY)
        self.max_rewrite: int = running_config.get('max_rewrite', DEFAULT_MAX_REWRITE)
        
        if init_info and 'task_type' in init_info and init_info['task_type']:
            self.task_type = init_info['task_type'].lower()
        elif 'task_type' in running_config:
            self.task_type = running_config['task_type'].lower()

        if "profile" in running_config and running_config['profile']:
            self.logger.info(f"Using profile: {running_config['profile']}")
            self.llm = AsyncLLM(running_config['profile'])

        assert 'env_name' in init_info, "Envrioment must be specified"
        self.env_name = init_info['env_name']
        if self.env_name == "alfworld":
            self.logger.info("Setting max steps to 80")
            init_info['env'].set_max_steps(80)
        self.executor.set_env(init_info['env'])

        self._load_resources()

    def _load_resources(self):
        resources_path = Path("agents/recode/resources/prompts") / self.env_name
        self.available_actions = open(resources_path / "actions.txt", "r").read()

        fewshots_path = Path("agents/recode/resources/fewshots") / self.env_name
        if self.env_name == "alfworld":
            self.fewshots = open(fewshots_path / f"{self.task_type}.txt", "r").read()
        elif self.env_name == "webshop":
            self.fewshots = open(fewshots_path / "base.txt", "r").read()
            # self.fewshots = "(No Examples)"
        elif self.env_name == "sciworld":
            self.fewshots = open(fewshots_path / "base.txt", "r").read()
        else:
            raise ValueError(f"Unsupported environment in _load_resources: {self.env_name}")

    async def act(self, observations: List[str]) -> List[str]:
        if not self.is_start:
            assert len(observations) == 1, "Only one observation is allowed for the first node"
            self._init_code_tree(observations[0])
            self.is_start = True

        if self.current_node.status == NodeStatus.STUB:
            await self._handle_stub()
        elif self.current_node.status == NodeStatus.ERROR:
            return ["[FINISH]"]

        if not self.current_node:
            return ["[FINISH]"]
        
        self.logger.info(f"[Execute]\n{self.current_node.code}")
        result = self._execute(self.current_node.code)
        self.current_node.observations.extend(result["stdout"]) if result["stdout"] else None
        self.logger.info(f"[Exec Result]\n{result}")

        if result["success"]:
            self.logger.info(f"[Execution Stdout] {result['stdout']}")
            self.current_node.status = NodeStatus.COMPLETED
            self.previous_node = self.current_node
            self.current_node = self.current_node.next()
            if not self.current_node:
                return ["[FINISH]"]
        else:
            if "NeedExpansion" in result["error"]:
                self.current_node.status = NodeStatus.STUB
            else:
                self.current_node.status = NodeStatus.ERROR
                self.current_node.error = result["error"]

    async def _handle_stub(self) -> None:
        if self.current_node and self.current_node.depth >= self.max_depth:
            if self.logger:
                self.logger.warning("Max depth reached - terminating.")
            self.current_node = None
            return

        new_blocks = await self._expand()
        self.logger.info("[NEW_BLOCKS]\n" + "\n".join(new_blocks)) if new_blocks else None

        if self.current_node:
            if new_blocks is None:
                self.current_node = None
                return
            if new_blocks:
                for block in new_blocks:
                    child_node = CodeNode(code=block, parent=self.current_node)
                    self.current_node.children.append(child_node)
            else: 
                self.current_node.status = NodeStatus.SKIP

        self.current_node = self.current_node.next()

    async def _expand(self) -> Optional[List[str]]:
        attempt = 0
        retry_hint_added = False
        while True:
            user_prompt = self._build_expand_prompt()
            if retry_hint_added:
                user_prompt += (
                    "\n\n[Important] Your previous expansion produced syntactically invalid code and/or included disallowed constructs (e.g., def/async def). "
                    "Strictly follow the rules: output a single valid Python code block, and do not use def or async def."
                )
            if self.logger:
                self.logger.info("[LLM_IN]\n" + user_prompt)
            response, _cost = await self.llm(user_prompt)
            if self.logger:
                self.logger.info("[LLM_OUT]\n" + response.strip())

            thought = parse_xml_tag(response, "think").strip()
            self.current_node.thought = thought
            expanded_code = parse_xml_tag(response, "execute").strip()

            try:
                blocks = split_blocks(expanded_code)
                validate_blocks(blocks)
                return blocks
            except (SyntaxError, ValueError) as e:
                attempt += 1
                retry_hint_added = True
                if attempt >= self.max_rewrite:
                    if self.logger:
                        self.logger.info(
                            f"[STOP] Reached max re-expands ({self.max_rewrite}). Last error: {e}. Ending episode."
                        )
                    return None
                if self.logger:
                    self.logger.info(
                        f"[RE-EXPAND {attempt}/{self.max_rewrite}] Split/validation failed due to: {e}. Re-asking EXPAND..."
                    )

    def _execute(self, code: str) -> dict:
        return self.executor.execute(code)

    def _init_code_tree(self, observation: str) -> None:
        self.logger.info(f"[OBSERVATIONS]\n{observation}")
        initial_observation, instruction = parse_raw_observation(observation, self.env_name)
        self.executor.set_var('observation', initial_observation)
        self.executor.set_var('instruction', instruction)
        self.root = CodeNode(code=f"solve(instruction, observation)")
        self.current_node = self.root
        
    def _build_expand_prompt(self) -> str:
        # available_actions, examples, task, variables
        examples = self.fewshots if self.fewshots else "(No Examples)"
        task = self.current_node.code
        variables = get_variables(self.executor, self.current_node.code)
        variables = variables if variables else "(No Variables)"
        return EXPAND_PROMPT.format(available_actions=self.available_actions, examples=examples, task=task, variables=variables)

    def _get_max_depth(self, node: Optional[CodeNode]) -> int:
        if node is None:
            return 0
        max_depth = node.depth
        for child in node.children:
            child_max = self._get_max_depth(child)
            if child_max > max_depth:
                max_depth = child_max
        return max_depth