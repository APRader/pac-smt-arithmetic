from z3 import And
from pac import interval as inter


class Domain:
    def __init__(self, variables, intervals):
        if variables is not list:
            variables = [variables]
        if intervals is not list:
            intervals = [intervals]
        if len(variables) == len(intervals):
            # All Interval objects are put into the dict directly
            self.domain = {variables[i]: intervals[i] for i in range(len(variables))
                           if isinstance(intervals[i], inter.Interval)}
            # All non-Interval objects are converted to Interval type first
            self.domain.update({variables[i]: inter.Interval(intervals[i]) for i in range(len(variables))
                                if not isinstance(intervals[i], inter.Interval)})
        else:
            raise ValueError("Variables and domains must have same length.")

    def add_domain(self, variable, interval):
        """
        Add domain to variable.
        :param variable: Z3 variable.
        :param interval: Interval object.
        """
        if not isinstance(interval, inter.Interval):
            interval = inter.Interval(interval)
        self.domain[variable] = interval

    def add_domains(self, variables, intervals):
        """
        Add domains to variables.
        :param variables: List of Z3 variables.
        :param intervals: List of Interval objects.
        """
        if len(variables) == len(intervals):
            for i in range(len(variables)):
                self.add_domain(variables[i], intervals[i])
        else:
            raise ValueError("Variables and domains must have same length.")

    def remove_domain(self, variable):
        """
        Remove domain from variable.
        :param variable: Z3 variable.
        """
        self.domain.pop(variable)

    def remove_domains(self, variables):
        """
        Remove domains from variables.
        :param variables: List of Z3 variables.
        """
        for variable in variables:
            self.remove_domain(variable)

    def create_formula(self):
        """
        Create a Z3 formula that restricts the variables to their domains.
        :return: A Z3 formula.
        """
        return And(domain.create_formula(variable) for variable, domain in self.domain)
