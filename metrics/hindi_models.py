from metrics.base import MetricBase
from metrics.awl import AvgWordLengthMetric
from metrics.psw import PolySyllabicWordMetric
from metrics.jdk import JodaAksharMetric

class IndicReadabilityRH1(MetricBase):
    """
    RH1 = -2.34 + 2.14 * AWL + 0.01 * PSW
    Hindi readability formula based on:
    - AWL: Average Word Length
    - PSW: Count of Polysyllabic Words
    """

    def name(self) -> str:
        return "IndicReadabilityRH1"

    def compute(self, sentence: str) -> float:
        awl = AvgWordLengthMetric().compute(sentence)
        psw = PolySyllabicWordMetric().compute(sentence)

        awl = awl if awl is not None else 0.0
        psw = psw if psw is not None else 0.0

        readability_score = -2.34 + (2.14 * awl) + (0.01 * psw)
        return round(readability_score, 3)
    
class IndicReadabilityRH2(MetricBase):
    """
    RH2 = 0.211 + 1.37 * AWL + 0.005 * JUK
    Hindi readability formula based on:
    - AWL: Average Word Length
    - JUK: Count of Joda Akshars (Conjunct Consonants)
    """

    def name(self) -> str:
        return "IndicReadabilityRH2"

    def compute(self, sentence: str, context: dict = None) -> float:
        awl = AvgWordLengthMetric().compute(sentence)
        juk = JodaAksharMetric().compute(sentence)

        awl = awl if awl is not None else 0.0
        juk = juk if juk is not None else 0.0

        readability_score = 0.211 + (1.37 * awl) + (0.005 * juk)
        return round(readability_score, 3)
