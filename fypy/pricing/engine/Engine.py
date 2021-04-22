from abc import ABC, abstractmethod
from fypy.instrument.Instrument import Instrument
from typing import Iterable, Generator


class Engine(ABC):
    """
    Abstract base class for a pricing engine, which prices instruments
    """

    @abstractmethod
    def price_instrument(self, inst: Instrument) -> float:
        """
        Price an instrument
        :param inst: Instrument object, the instrument to price
        :return: price of instrument
        """
        raise NotImplementedError

    def price_instruments(self, instruments: Iterable[Instrument]) -> Generator:
        """
        Price an iterable (e.g. List) of instruments
        :param instruments:
        :return: a Generator to populate some container with the prices of instruments, or perform some other comp
        """
        for inst in instruments:
            yield self.price_instrument(inst)
