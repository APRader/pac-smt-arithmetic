import unittest

from z3 import Int, Ints, Real, Reals, Bool, And, Solver, unsat

from pac import pac_learner, interval, domain


class TestPAC(unittest.TestCase):
    def test_decide_pac_knowledge_base(self):
        """
        Test that DecidePAC takes knowledge base into account.
        """
        x, y = Ints('x y')
        learner = pac_learner.PACLearner(x >= 0)
        examples = [y == 1, y == 2]
        query = x + y > 0
        # Without knowledge base, query would be rejected; with it, it should be accepted
        self.assertTrue(learner.decide_pac(examples, query))

    def test_decide_pac_validity(self):
        """
        Test that DecidePAC takes validity into account.
        """
        x = Real('x')
        learner = pac_learner.PACLearner()
        examples = [x >= 0, x >= 1, x >= -1]
        query = x >= 0
        # The query is only true for 2 out of 3 examples
        self.assertTrue(learner.decide_pac(examples, query, 0.6))
        self.assertFalse(learner.decide_pac(examples, query, 0.7))

    def test_decide_pac_return_actual_validity(self):
        """
        Test that DecidePAC calculates actual validity correctly.
        """
        x = Bool('x')
        learner = pac_learner.PACLearner()
        examples = [x == True, x == False]
        query = x
        state, validity = learner.decide_pac(examples, query, return_actual_validity=True)
        # The query is valid in half the cases
        self.assertEqual(validity, 0.5)

    def test_decide_pac_validity_assertion(self):
        """
        Test that DecidePAC asserts that the validity is between 0 and 1.
        """
        learner = pac_learner.PACLearner()
        # Wrong value should raise ValueError
        with self.assertRaises(ValueError):
            learner.decide_pac([], True, -1)

    def test_interval_assignment(self):
        """
        Test that you can create an Interval object that is an assignment.
        """
        ass = interval.Interval(23)
        x = Int("x")
        formula = ass.create_formula(x)
        s = Solver()
        s.add(formula != (x == 23))
        # Check that the formulas are equivalent
        self.assertEqual(s.check(), unsat)

    def test_interval_tuple(self):
        """
        Test that you can create an open interval using a tuple.
        """
        open_interval = interval.Interval((-2, 0))
        x = Int("x")
        formula = open_interval.create_formula(x)
        s = Solver()
        s.add(formula != And(x > -2, x < 0))
        # Check that the formulas are equivalent
        self.assertEqual(s.check(), unsat)

    def test_interval_list(self):
        """
        Test that you can create a closed interval using a list.
        """
        closed_interval = interval.Interval([-3.8, 8.9])
        x = Real("x")
        formula = closed_interval.create_formula(x)
        s = Solver()
        s.add(formula != And(x >= -3.8, x <= 8.9))
        # Check that the formulas are equivalent
        self.assertEqual(s.check(), unsat)

    def test_interval_string(self):
        """
        Test that you can create a mixed interval using a string.
        """
        mixed_interval = interval.Interval("(-inf,-3]")
        x = Int("x")
        formula = mixed_interval.create_formula(x)
        s = Solver()
        s.add(formula != And(x <= -3))
        # Check that the formulas are equivalent
        self.assertEqual(s.check(), unsat)

    def test_interval_kwargs(self):
        """
        Test that you can create a mixed interval using keyword arguments.
        """
        mixed_interval = interval.Interval(lower=4.9, lower_bound="closed", upper=float("inf"), upper_bound="open")
        x = Int("x")
        formula = mixed_interval.create_formula(x)
        s = Solver()
        s.add(formula != And(x >= 4.9))
        # Check that the formulas are equivalent
        self.assertEqual(s.check(), unsat)

    def test_add_domain(self):
        """
        Test that you can add a domain to your existing domains.
        """
        x, y = Reals("x y")
        x_domain = interval.Interval("(-2.4,3.1]")
        y_domain = interval.Interval("[-9.7,6.5)")
        domains = domain.Domain(x, x_domain)
        domains.add_domain(y, y_domain)
        formula = domains.create_formula()
        s = Solver()
        s.add(formula != And(x > -2.4, x <= 3.1, y >= -9.7, y < 6.5))
        # Check that the formulas are equivalent
        self.assertEqual(s.check(), unsat)

    def test_add_domains(self):
        """
        Test that you can add domains to your existing domains.
        """
        x, y, z = Ints("x y z")
        x_domain = interval.Interval("(0,inf)")
        y_domain = interval.Interval((float("-inf"), 0))
        z_domain = interval.Interval(lower=1, lower_bound="closed", upper=1, upper_bound="closed")
        domains = domain.Domain(x, x_domain)
        domains.add_domains([y, z], [y_domain, z_domain])
        formula = domains.create_formula()
        s = Solver()
        s.add(formula != And(x > 0, y < 0, z == 1))
        # Check that the formulas are equivalent
        self.assertEqual(s.check(), unsat)

    def test_remove_domain(self):
        """
        Test that you can remove a domain from your existing domains.
        """
        x, y = Reals("x y")
        x_domain = interval.Interval(lower=-6.4, lower_bound="closed", upper=0.8, upper_bound="open")
        y_domain = interval.Interval([-3.6, 0])
        domains = domain.Domain([x, y], [x_domain, y_domain])
        domains.remove_domain(y)
        formula = domains.create_formula()
        s = Solver()
        s.add(formula != And(x >= -6.4, x < 0.8))
        # Check that the formulas are equivalent
        self.assertEqual(s.check(), unsat)

    def test_remove_domains(self):
        """
        Test that you can remove domains from your existing domains.
        """
        x, y, z = Ints("x y z")
        x_domain = interval.Interval(lower=-6, lower_bound="closed", upper=-3.4, upper_bound="open")
        domains = domain.Domain([x, y, z], [x_domain, "(0.4,1.8]", "[-5,1)"])
        domains.remove_domains([x, y])
        formula = domains.create_formula()
        s = Solver()
        s.add(formula != And(z >= -5, z < 1))
        # Check that the formulas are equivalent
        self.assertEqual(s.check(), unsat)


if __name__ == '__main__':
    unittest.main()
