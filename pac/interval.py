from enum import Enum
from numbers import Number
from z3 import And


class Bound(Enum):
    OPEN = 1
    CLOSED = 2


class Interval:
    def __init__(self, variable, *args, **kwargs):
        """
        You can create intervals in the following ways:
        1. (x,y): tuple for open intervals
        2. [x,y]: list for closed intervals
        3. '(x,y]': string for mixed intervals
        4. x: number for assignments
        5. lower=x, lower_bound='closed', upper=y, upper_bound='open': kwargs for mixed intervals
        """
        self.variable = variable
        if len(args) == 1:
            [interval] = args
            if type(interval) is tuple:
                if len(interval) == 2:
                    self.set_interval(interval[0], interval[1])
                    self.lower_bound = Bound.OPEN
                    self.upper_bound = Bound.OPEN
                else:
                    raise TypeError("The tuple has to be of length 2.")
            elif type(interval) is list:
                if len(interval) == 2:
                    self.set_interval(interval[0], interval[1])
                    self.lower_bound = Bound.CLOSED
                    self.upper_bound = Bound.CLOSED
                else:
                    raise TypeError("The list has to be of length 2.")
            elif type(interval) is str:
                if interval[0] == "(":
                    self.lower_bound = Bound.OPEN
                elif interval[0] == "[":
                    self.lower_bound = Bound.CLOSED
                else:
                    raise ValueError("The first symbol has to be ( or [.")
                if interval[-1] == ")":
                    self.upper_bound = Bound.OPEN
                elif interval[-1] == "]":
                    self.upper_bound = Bound.CLOSED
                else:
                    raise ValueError("The last symbol has to be ) or ].")
                values = interval[1:-1].split(",")
                self.set_interval(values[0], values[1])
            elif isinstance(interval, Number):
                self.set_interval(interval, interval)
                self.lower_bound = Bound.CLOSED
                self.upper_bound = Bound.CLOSED
            else:
                raise TypeError("Wrong input type. Must either be tuple, list, string or number.")
        elif kwargs:
            self.set_interval(kwargs["lower"], kwargs["upper"])
            if kwargs["lower_bound"] == "open":
                self.lower_bound = Bound.OPEN
            elif kwargs["lower_bound"] == "closed":
                self.lower_bound = Bound.CLOSED
            else:
                raise ValueError("Lower bound must either be 'open' or 'closed'.")
            if kwargs["upper_bound"] == "open":
                self.upper_bound = Bound.OPEN
            elif kwargs["upper_bound"] == "closed":
                self.upper_bound = Bound.CLOSED
            else:
                raise ValueError("Upper bound must either be 'open' or 'closed'.")
        else:
            raise TypeError("Only one argument allowed. To set numbers and bounds individually, use keyword arguments "
                            "lower, upper, lower_bound and upper_bound")

    def set_interval(self, lower, upper):
        """
        Sets lower and upper bound of interval.
        :param lower: Lower bound.
        :param upper: Upper bound.
        """
        if type(lower) is str:
            self.lower = float(lower)
        else:
            self.lower = lower
        if type(upper) is str:
            self.upper = float(upper)
        else:
            self.upper = upper

    def create_formula(self):
        """
        Put the variable into the interval as a Z3 formula.
        :return: A Z3 formula.
        """
        if self.lower == self.upper:
            # It is an assignment
            return self.variable == self.lower

        formula = []

        if not self.lower == float("-inf"):
            # Domain is bounded below
            if self.lower_bound == Bound.OPEN:
                formula.append(self.lower < self.variable)
            elif self.lower_bound == Bound.CLOSED:
                formula.append(self.lower <= self.variable)
        if not self.upper == float("inf"):
            # Domain is bounded above
            if self.upper_bound == Bound.OPEN:
                formula.append(self.variable < self.upper)
            elif self.upper_bound == Bound.CLOSED:
                formula.append(self.variable <= self.upper)

        return formula[0] if len(formula) == 1 else And(formula)
