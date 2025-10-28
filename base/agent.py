from abc import ABC, abstractmethod
from typing import List

class Agent(ABC):
    @abstractmethod
    async def act(self, observations: List[str]) -> List[str]:
        pass

    @abstractmethod
    def reset(self, running_config: dict, init_info: dict=None) -> None:
        pass

    @abstractmethod
    def report(self) -> dict:
        pass