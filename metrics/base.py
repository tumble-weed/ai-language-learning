from abc import ABC, abstractmethod

class MetricBase(ABC):
  @abstractmethod
  def name(self)->str:
    """Unique Name for the metric"""
    pass

  def compute(self, sentence: str, context: dict)-> float:
    """Compute the metric for a given sentence"""
    pass