import unittest

from pac import pac_learner
from z3 import Ints, Real, Bool


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
        with self.assertRaises(ValueError):
            learner.decide_pac([], True, -1)


if __name__ == '__main__':
    unittest.main()
