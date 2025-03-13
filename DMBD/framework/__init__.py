"""
Dynamic Markov Blanket Detection (DMBD) Framework
================================================

A comprehensive framework for detecting and analyzing Markov blankets
in dynamic systems using probabilistic techniques.
"""

__version__ = '0.1.0'
__all__ = ['markov_blanket', 'data_partitioning', 'cognitive_identification', 'visualization']

from .markov_blanket import MarkovBlanket, DynamicMarkovBlanket
from .data_partitioning import DataPartitioning
from .cognitive_identification import CognitiveIdentification
from .visualization import MarkovBlanketVisualizer
