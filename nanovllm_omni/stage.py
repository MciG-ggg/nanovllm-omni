"""Stage contract for MiniMind-Omni pipelines."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")


class Stage(ABC, Generic[Input, Output]):
    """A named synchronous processing boundary."""

    name: str

    @abstractmethod
    def execute(self, payload: Input) -> Output:
        """Transform one typed payload into the next."""
