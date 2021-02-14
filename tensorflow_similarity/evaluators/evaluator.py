from abc import ABC, abstractmethod


class Evaluator(ABC):
    """Evaluate index performance and calibrates it

    Note: Evaluators are derived from this abstract class to allow users to
    override the evaluation to use additional data or interface with different
    evaluation system. For example fetching data from a remote database.
    """

    # FIXME: add abstract methods when API is stable
    # -> see memoryEvaluator for now
