import unittest
from verification.verify_solution import verify, verify_report


class VerifySolutionTest(unittest.TestCase):
    def setUp(self):
        self.graph = {'nodes': {
            'r': {'eclass': '0', 'cost': 3, 'children': ['a', 'a']},
            'a': {'eclass': '1', 'cost': 2, 'children': []},
            'b': {'eclass': '1', 'cost': 4, 'children': []}}, 'root_eclasses': ['0', '0']}

    def solution(self, names, cost):
        return {'selected_enodes': names, 'objective': cost}

    def test_valid_shared_children_and_repeated_roots(self):
        self.assertEqual(verify(self.graph, self.solution(['r', 'a'], 5))['objective'], 5)

    def test_missing_child(self):
        with self.assertRaisesRegex(ValueError, 'missing child'):
            verify(self.graph, self.solution(['r'], 3))

    def test_missing_root(self):
        with self.assertRaisesRegex(ValueError, 'Missing roots'):
            verify(self.graph, self.solution(['a'], 2))

    def test_multiple_choices(self):
        with self.assertRaisesRegex(ValueError, 'Multiple choices'):
            verify(self.graph, self.solution(['r', 'a', 'b'], 9))

    def test_cycle_and_self_cycle(self):
        self.graph['nodes']['a']['children'] = ['r']
        with self.assertRaisesRegex(ValueError, 'cycle'):
            verify(self.graph, self.solution(['r', 'a'], 5))
        self.graph['nodes']['a']['children'] = ['a']
        with self.assertRaisesRegex(ValueError, 'cycle'):
            verify(self.graph, self.solution(['r', 'a'], 5))

    def test_wrong_cost(self):
        with self.assertRaisesRegex(ValueError, 'Objective mismatch'):
            verify(self.graph, self.solution(['r', 'a'], 7))

    def test_integer_class_references_and_truncation(self):
        self.graph['nodes']['r']['children'] = [1, 1]
        self.graph['nodes']['a']['cost'] = 2.9
        self.assertTrue(verify(self.graph, self.solution(['r', 'a'], 5))['valid'])

    def test_no_solution_is_not_a_pass(self):
        with self.assertRaises(ValueError):
            verify_report(self.graph, [{'solver': {'status': 'UNKNOWN'}}])


if __name__ == '__main__':
    unittest.main()
