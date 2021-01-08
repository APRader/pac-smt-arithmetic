import unittest

from z3 import Int, Ints, Real, Reals, Bool, And, Solver, unsat

from pac import pac_learner, interval


class TestPAC(unittest.TestCase):
    def test_interval_assignment(self):
        """
        Test that you can create an Interval object that is an assignment.
        """
        x = Int("x")
        ass = interval.Interval(x, 23)
        formula = ass.create_formula()
        s = Solver()
        s.add(formula != (x == 23))
        # Check that the formulas are equivalent
        self.assertEqual(s.check(), unsat)

    def test_interval_tuple(self):
        """
        Test that you can create an open interval using a tuple.
        """
        x = Int("x")
        open_interval = interval.Interval(x, (-2, 0))
        formula = open_interval.create_formula()
        s = Solver()
        s.add(formula != And(x > -2, x < 0))
        # Check that the formulas are equivalent
        self.assertEqual(s.check(), unsat)

    def test_interval_list(self):
        """
        Test that you can create a closed interval using a list.
        """
        x = Real("x")
        closed_interval = interval.Interval(x, [-3.8, 8.9])
        formula = closed_interval.create_formula()
        s = Solver()
        s.add(formula != And(x >= -3.8, x <= 8.9))
        # Check that the formulas are equivalent
        self.assertEqual(s.check(), unsat)

    def test_interval_string(self):
        """
        Test that you can create a mixed interval using a string.
        """
        x = Int("x")
        mixed_interval = interval.Interval(x, "(-inf,-3]")
        formula = mixed_interval.create_formula()
        s = Solver()
        s.add(formula != And(x <= -3))
        # Check that the formulas are equivalent
        self.assertEqual(s.check(), unsat)

    def test_interval_kwargs(self):
        """
        Test that you can create a mixed interval using keyword arguments.
        """
        x = Int("x")
        mixed_interval = interval.Interval(x, lower=4.9, lower_bound="closed", upper=float("inf"), upper_bound="open")
        formula = mixed_interval.create_formula()
        s = Solver()
        s.add(formula != And(x >= 4.9))
        # Check that the formulas are equivalent
        self.assertEqual(s.check(), unsat)

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

    def test_decide_pac_domain(self):
        """
        Test that DecidePAC takes domain into account.
        """
        x, y = Reals('x y')
        domain = [interval.Interval(x, [0, 1]), interval.Interval(y, [2, 3])]
        learner = pac_learner.PACLearner(True, domain)
        examples = [[interval.Interval(x, ("-inf", "inf")), interval.Interval(y, (2.1, 2.9))],
                    [interval.Interval(x, (0.2, "inf")), interval.Interval(y, (2.2, 2.8))]]
        query = y - x >= 1
        # The query is only valid if x is restricted to its domain.
        self.assertTrue(learner.decide_pac(examples, query, 0.9))


if __name__ == '__main__':
    unittest.main()
