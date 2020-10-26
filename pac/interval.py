from enum import Enum
from numbers import Number
from z3 import And


class Bound(Enum):
    OPEN = 1
    CLOSED = 2


class Interval:
    def __init__(self, *args, **kwargs):
        """
        You can create intervals in the following ways:
        1. (x,y): tuple for open intervals
        2. [x,y]: list for closed intervals
        3. '(x,y]': string for mixed intervals
        4. x: number for assignments
        5. lower=x, lower_bound='closed', upper=y, upper_bound='open': kwargs for mixed intervals
        """
        if len(args) == 1:
            [interval] = args
            if type(interval) is tuple:
                if len(interval) == 2:
                    self.lower = interval[0]
                    self.upper = interval[1]
                    self.lower_bound = Bound.OPEN
                    self.upper_bound = Bound.OPEN
                else:
                    raise TypeError("The tuple has to be of length 2.")
            elif type(interval) is list:
                if len(interval) == 2:
                    self.lower = interval[0]
                    self.upper = interval[1]
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
                self.lower = float(values[0])
                self.upper = float(values[1])
            elif isinstance(interval, Number):
                self.lower = interval
                self.upper = interval
                self.lower_bound = Bound.CLOSED
                self.upper_bound = Bound.CLOSED
            else:
                raise TypeError("Wrong input type. Must either be tuple, list, string or number.")
        elif kwargs:
            self.lower = kwargs["lower"]
            self.upper = kwargs["upper"]
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

    def create_formula(self, variable):
        """
        Put the variable into the interval as a Z3 formula.
        :param variable: A Z3 variable.
        :return: A Z3 formula.
        """
        if self.lower == self.upper:
            # It is an assignment
            return variable == self.lower

        formula = []

        if not self.lower == float("-inf"):
            # Domain is bounded below
            if self.lower_bound == Bound.OPEN:
                formula.append(self.lower < variable)
            elif self.lower_bound == Bound.CLOSED:
                formula.append(self.lower <= variable)
        if not self.upper == float("inf"):
            # Domain is bounded above
            if self.upper_bound == Bound.OPEN:
                formula.append(variable < self.upper)
            elif self.upper_bound == Bound.CLOSED:
                formula.append(variable <= self.upper)

        return And(formula) if len(formula) > 1 else formula[0]
