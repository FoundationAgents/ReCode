import ast
from dataclasses import dataclass, field
from enum import Enum
import uuid
from typing import List, Optional, Any
from utils.executor import Executor
import re

def parse_raw_observation(raw_observation: str, env_name: str) -> tuple[str, str, str]:
    if env_name == "alfworld" or env_name == "travelplanner":
        lines = raw_observation.split("\n")
        if "Your task is to:" in lines[1]:
            task_description = lines[1].split("Your task is to:")[-1].strip().removesuffix(".")
            code = task_description.replace(' ', '_') + '()'
        return lines[0], task_description
    elif env_name == "webshop":
        task_description = raw_observation.strip().split('\n')[0].strip()
        return raw_observation.strip(), task_description
    elif env_name == "sciworld":
        lines = raw_observation.split("\n")
        return '\n'.join(lines[2:]), lines[1]
    else:
        raise ValueError(f"Unsupported environment in parse_raw_observation: {env_name}")
    
class NodeStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    STUB = "STUB"
    ERROR = "ERROR"
    SKIP = "SKIP"

@dataclass
class CodeNode:
    thought: str = ""
    code: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent: Optional['CodeNode'] = None
    children: List['CodeNode'] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    depth: int = 0
    error: str = None
    observations: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.depth = 0 if not self.parent else self.parent.depth + 1

    def next(self) -> Optional['CodeNode']:
        for child in self.children:
            if child.status == NodeStatus.PENDING:
                return child

        if self.parent:
            siblings = self.parent.children
            try:
                current_index = siblings.index(self)
                for i in range(current_index + 1, len(siblings)):
                    if siblings[i].status == NodeStatus.PENDING:
                        return siblings[i]
            except ValueError:
                pass
        if self.parent:
            return self.parent.next()
        return None
    
    def clear(self) -> None:
        self.status = NodeStatus.PENDING
        self.code = ""
        self.error = None
        self.observations = []
    
def split_blocks(source: str) -> List[str]:
    if not source.strip():
        return []

    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                raise ValueError(
                    "Function definitions (def/async def) are not allowed in expanded code"
                )
        lines = source.splitlines(True)
        return [
            "".join(lines[node.lineno - 1 : getattr(node, "end_lineno", node.lineno)])
            for node in tree.body
        ]
    except SyntaxError:
        pass

    import codeop
    blocks: List[str] = []
    buf: List[str] = []
    compiler = codeop.CommandCompiler()

    def flush_buf():
        if buf:
            blocks.append("".join(buf))
            buf.clear()

    for line in source.splitlines(True):
        buf.append(line)
        try:
            compiled = compiler("".join(buf), symbol="exec")
        except (SyntaxError, ValueError, OverflowError):
            prev = buf[:-1]
            try:
                prev_compiled = compiler("".join(prev), symbol="exec") if prev else None
            except Exception:
                prev_compiled = None

            if prev and prev_compiled:
                blocks.append("".join(prev))
                buf[:] = [line]
                try:
                    compiler(line, symbol="exec")
                except Exception:
                    blocks.append(line)
                    buf.clear()
                continue

            last = buf.pop()
            blocks.append(last)
            continue

        if compiled is not None:
            flush_buf()

    if buf:
        blocks.append("".join(buf))

    return blocks

def validate_blocks(blocks: List[str]) -> None:
    import codeop
    compiler = codeop.CommandCompiler()
    for block in blocks:
        try:
            compiled = compiler(block, symbol="exec")
        except Exception as e:
            raise SyntaxError(f"Invalid Python block: {e}")
        if compiled is None:
            raise SyntaxError("Incomplete Python block produced by EXPAND.")
        try:
            tree = ast.parse(block)
        except SyntaxError as e:
            raise e
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                raise ValueError("Function definitions (def/async def) are not allowed in expanded code")

def get_variables(executor: Executor, code: str) -> str:
    if not code:
        raise ValueError("No code provided to get_variables")

    def try_literal_eval(node: ast.AST):
        try:
            return ast.literal_eval(node)
        except Exception:
            return None

    discovered_var_names: List[str] = []
    discovered_var_set = set()

    try:
        tree = ast.parse(code)
    except Exception:
        raise ValueError("Invalid code when getting variables")

    def collect_from_call(call: ast.Call):
        nonlocal discovered_var_names, discovered_var_set
        for arg in call.args:
            if isinstance(arg, ast.Name):
                var_name = arg.id
                if var_name not in discovered_var_set:
                    discovered_var_set.add(var_name)
                    discovered_var_names.append(var_name)
            for kw in call.keywords:
                if kw.arg is None:
                    continue
                literal_value = try_literal_eval(kw.value)
                if literal_value is not None:
                    executor.set_var(kw.arg, literal_value)
                    if kw.arg not in discovered_var_set:
                        discovered_var_set.add(kw.arg)
                        discovered_var_names.append(kw.arg)
                    continue
                if isinstance(kw.value, ast.Name):
                    var_name = kw.value.id
                    if var_name not in discovered_var_set:
                        discovered_var_set.add(var_name)
                        discovered_var_names.append(var_name)

    for stmt in getattr(tree, "body", []):
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            collect_from_call(stmt.value)
            break
        if isinstance(stmt, ast.AnnAssign) and isinstance(getattr(stmt, "value", None), ast.Call):
            collect_from_call(stmt.value)
            break
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            collect_from_call(stmt.value)
            break

    if not discovered_var_names:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                collect_from_call(node)
                break

    if not discovered_var_names:
        return ""

    lines: List[str] = []
    for name in discovered_var_names:
        value = executor.get_var(name)
        if hasattr(executor, "_infer_type_string"):
            value_type = executor._infer_type_string(value)
        else:
            value_type = type(value).__name__ if value is not None else "NoneType"
        lines.append(f"- {name} ({value_type}): {value}")

    return "\n".join(lines)