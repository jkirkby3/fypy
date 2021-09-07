from datetime import datetime, timedelta
from typing import Union


class Date(datetime):
    """
    fypy Date class, extends datetime.datetime, adding some convenience functionality
    """

    @classmethod
    def from_datetime(cls, date: datetime) -> 'Date':
        return cls(year=date.year, month=date.month, day=date.day, hour=date.hour, minute=date.minute,
                   second=date.second, microsecond=date.microsecond, tzinfo=date.tzinfo, fold=date.fold)

    @classmethod
    def today(cls) -> 'Date':
        date_now = cls.now()
        return Date(day=date_now.day, year=date_now.year, month=date_now.month)

    def __add__(self, amount: Union[timedelta, int]):
        if isinstance(amount, timedelta):
            return super().__add__(amount)

        return super().__add__(timedelta(days=amount))

    def __sub__(self, amount: Union[timedelta, int]):
        if isinstance(amount, timedelta):
            return super().__sub__(amount)

        return super().__sub__(timedelta(days=amount))

    @staticmethod
    def days_between(start: 'Date', end: 'Date') -> int:
        """
        Calculate the number of days between two dates, ignoring granularity below a day
        :param start: Date, the start date
        :param end: Date, the end date
        :return: int, number of days between start and end
        """
        return (datetime.date(end) - datetime.date(start)).days


if __name__ == '__main__':
    d = Date(year=2015, month=12, day=31)
    print(d)

    d2 = Date.from_datetime(d)
    print(d2)

    d3 = d2+timedelta(days=3)
    print(d3)

    d4 = d2+3
    print(d4)

    dd = Date.days_between(start=d2, end=d4)
    print(dd)

    dn = Date.today()
    print(dn)
