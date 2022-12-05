from abc import ABC, abstractmethod
from fypy.termstructures.ForwardCurve import ForwardCurve


class Drift(ABC):
    """
    Drift base class, used by processes
    """
    @abstractmethod
    def __call__(self, t: float) -> float:
        """ Evaluate the Drift at a given time """
        raise NotImplementedError

    @abstractmethod
    def avg(self, t: float, dt: float) -> float:
        """ Evaluate the average Drift between [t,t+dt) """
        raise NotImplementedError


class Drift_FC(Drift):
    def __init__(self, fwd: ForwardCurve):
        """
        A forward curve implied drift
        :param fwd: ForwardCurve, the forward curve
        """
        self._fwd = fwd

    def __call__(self, t: float) -> float:
        """ Evaluate the Drift at a given time """
        return self.avg(t)

    def avg(self, t: float, dt: float = 1e-05) -> float:
        """ Evaluate the average Drift between [t,t+dt) """
        return self._fwd.drift(t, t+dt)
