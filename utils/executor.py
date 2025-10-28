from typing import List, Dict, Any, Callable
import io, sys
import functools
import asyncio
import types
import re
import threading
import time

from utils.llm import AsyncLLM
from base.environment import Env

def print_output(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is not None:
            print(result, file=sys.stdout, flush=True)
        return result
    return wrapper

class Executor:
    def __init__(self, env: Env = None, if_run_print: bool = False) -> None:
        self.env = env
        self.actions: List[str] = []
        self._variables: Dict[str, Any] = {}
        self.if_run_print = if_run_print
        if self.if_run_print:
            self.run = print_output(self.run)
        self._base_globals = {
            "run": self.run,
            "re": re,
        }
        self._loop = None
        self._loop_thread = None
        self._start_loop_thread()

    def register_function(self, name: str, func: Callable):
        self._base_globals[name] = func
    
    def register_action_function(self, name: str, func: Callable):
        func_with_run = lambda *args, **kwargs: self.run(func(*args, **kwargs))
        self.register_function(name, func_with_run)

    def register_ask_llm(self, llm: AsyncLLM):
            def _ask_llm_sync(query: str) -> str:
                async def _ask_llm(query: str) -> str:
                    response, _cost = await llm(
                        prompt=query,
                    )
                    return response
                return self._submit_coro(_ask_llm(query))
            self.register_function("ask_llm", _ask_llm_sync)


    def skip(self, reason: str):
        return None
    
    def set_var(self, key: str, value: Any):
        self._variables[key] = value
    
    def get_var(self, key: str) -> Any:
        if key not in self._variables:
            return None
        return self._variables.get(key)
    
    def set_env(self, env: Env):
        self.env = env
    
    def _is_preserved_variable(self, key: str, value: Any) -> bool:
        if key.startswith('_') or key in self._base_globals:
            return False
        return not isinstance(value, (types.ModuleType, types.FunctionType, 
                                    types.BuiltinFunctionType, types.MethodType, type))
    
    def _infer_type_string(self, value: Any, depth: int = 0, max_depth: int = 2) -> str:
        if value is None:
            return "NoneType"
        if depth > max_depth:
            return type(value).__name__
        try:
            if isinstance(value, (bool, int, float, str)):
                return type(value).__name__
            if isinstance(value, list):
                if not value:
                    return "list"
                elem_types = {self._infer_type_string(v, depth + 1, max_depth) for v in value[:5]}
                if len(elem_types) == 1:
                    return f"list[{next(iter(elem_types))}]"
                return "list"
            if isinstance(value, tuple):
                if not value:
                    return "tuple"
                elem_types = [self._infer_type_string(v, depth + 1, max_depth) for v in value[:5]]
                if all(t == elem_types[0] for t in elem_types):
                    return f"tuple[{elem_types[0]}]"
                return f"tuple[{', '.join(elem_types)}]"
            if isinstance(value, set):
                if not value:
                    return "set"
                sample = list(value)[:5]
                elem_types = {self._infer_type_string(v, depth + 1, max_depth) for v in sample}
                if len(elem_types) == 1:
                    return f"set[{next(iter(elem_types))}]"
                return "set"
            if isinstance(value, dict):
                if not value:
                    return "dict"
                items = list(value.items())[:5]
                key_types = {self._infer_type_string(k, depth + 1, max_depth) for k, _ in items}
                val_types = {self._infer_type_string(v, depth + 1, max_depth) for _, v in items}
                if len(key_types) == 1 and len(val_types) == 1:
                    return f"dict[{next(iter(key_types))}, {next(iter(val_types))}]"
                return "dict"
            return type(value).__name__
        except Exception:
            return type(value).__name__
    
    def run(self, action: str) -> str:
        if self.env is None:
            raise RuntimeError("Environment not set. Call set_env() first.")
        
        result = self._submit_coro(self.env.run(action))
        
        self.actions.append(action)
        if isinstance(result, list):
            result = "\n".join(result)
        return result
    
    def get_actions(self) -> List[str]:
        actions = self.actions.copy()
        self.actions.clear()
        return actions
    
    def get_variables(self) -> str:
        return "\n".join([f"- {key} ({self._infer_type_string(value)}): {value}" for key, value in self._variables.items()])
    
    def reset(self):
        self.actions.clear()
        self._variables.clear()

    def _start_loop_thread(self):
        if self._loop and self._loop.is_running():
            return
        def _loop_runner():
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            loop.run_forever()
        t = threading.Thread(target=_loop_runner, daemon=True)
        t.start()
        while self._loop is None or not self._loop.is_running():
            time.sleep(0.01)
        self._loop_thread = t

    def _submit_coro(self, coro):
        self._start_loop_thread()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self):
        if self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
            if self._loop_thread:
                self._loop_thread.join(timeout=1)
        self._loop = None
        self._loop_thread = None

    def execute(self, code: str) -> Dict[str, Any]:
        success, stdout_lines, error_msg = self._run_block(code)
        return {"code": code, "stdout": stdout_lines, "error": error_msg, "success": success}

    def _run_block(self, block: str) -> tuple[bool, List[str], str]:
        output = []
        
        class OutputCapture:
            def __init__(self):
                self.lines = []
            def write(self, text):
                if text and text != '\n':
                    self.lines.extend(line for line in text.splitlines() if line.strip())
            def flush(self): pass

        capture = OutputCapture()
        old_stdout = sys.stdout
        sys.stdout = capture
        
        exec_globals = {**self._base_globals, **self._variables}
        
        try:
            exec(block, exec_globals)
            
            for key, value in exec_globals.items():
                if self._is_preserved_variable(key, value):
                    self._variables[key] = value
            
            return True, capture.lines, ""
        except NameError as e:
            match = re.search(r"name '(.+?)' is not defined", str(e))
            if match and f"{match.group(1)}(" in block:
                return False, capture.lines, f"NeedExpansion: `{match.group(1)}` needs to be expanded."
            return False, capture.lines, f"NameError: {e}"
        except Exception as e:
            return False, capture.lines, f"{e.__class__.__name__}: {e}"
        finally:
            sys.stdout = old_stdout