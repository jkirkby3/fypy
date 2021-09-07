from abc import ABC, abstractmethod
from fypy.date.Date import Date


class DayCounter(ABC):
    """
    Day counter class, used to count the number of days between dates, and to compute the year fraction
    between dates.  New day counters can be created by constructing your own calendar, which determines
    which dates are business dates (e.g. clearing/exchange/libor calendars)
    """

    @abstractmethod
    def year_fraction_from_days(self, days: int) -> float:
        """
        Convert a number of days into a year fraction.
        This is one of two main methods to override
        :param days: int, number of days between two dates
        :return: float, the year fraction corresponding to those days.
        """
        raise NotImplementedError

    @staticmethod
    def days_between(start: Date, end: Date) -> int:
        """
        Count the number of calendar days between two dates.
        This is one of two main methods to override. By default, it computes the actual number of days
        :param start: Date, the start date
        :param end: Date, the end date
        :return: int, number of days between start and end
        """
        return Date.days_between(start=start, end=end)

    def year_fraction(self, start: Date, end: Date) -> float:
        """
        Calculate the year fraction between two dates.
        :param start: Date, the start date
        :param end: Date, the end date
        :return: float, year fraction between two dates.
        """
        return self.year_fraction_from_days(days=self.days_between(start=start, end=end))


class DayCounter_365(DayCounter):
    """
    Day counter which assumes 365 days in a year, and counts days based on actual dates.
    """

    def year_fraction_from_days(self, days: int) -> float:
        return days / 365
