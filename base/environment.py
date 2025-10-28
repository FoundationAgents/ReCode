from abc import ABC, abstractmethod
from typing import Union, List, Any, Optional


class Env(ABC):
    id: str
    _step_count: int = 0
    _done: bool = False
    _success: bool = False

    @abstractmethod
    async def _run(self, action: str) -> Any:
        pass

    async def run(self, action: List[str]) -> List[str]:
        if isinstance(action, str):
            action = [action]

        if not action:
            return []

        observations: List[Any] = []
        for single_action in action:
            observations.append(await self._run(single_action))
            if self.is_success():
                self._done = True
            if self.is_done():
                break
        return observations

    def is_done(self) -> bool:
        return self._done

    def is_success(self) -> bool:
        return self._success

    @abstractmethod
    def reset(self, running_config: dict, id: Optional[str] = None) -> dict:
        pass

    def get_step_count(self) -> int:
        return self._step_count    

    @abstractmethod
    def report(self) -> dict:
        pass

    async def close(self) -> None:
        pass