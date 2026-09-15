import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.scripts.partition_comparison import (
    best_result,
    load_summary,
    make_cases,
    render_table,
    run_experiments,
    select_cases,
    table_rows,
)


class PartitionComparisonTests(unittest.TestCase):
    def test_experiment_matrix_matches_tables_iv_and_v(self):
        cases = make_cases()

        self.assertEqual(len(cases), 18)
        self.assertEqual(
            {(case.size, case.W) for case in cases if case.table == 'IV'},
            {(256, 40), (512, 40), (1024, 80)},
        )
        self.assertEqual(
            {(case.size, case.W) for case in cases if case.table == 'V'},
            {(192, 20), (384, 80), (768, 80)},
        )

    def test_case_filters(self):
        cases = select_cases('V', ('QFT',), (384,))

        self.assertEqual([case.key for case in cases], ['V:QFT-384'])

    def test_best_result_uses_earliest_time_for_best_objective(self):
        history = [(1.0, 20.0), (2.0, 10.0), (3.0, 10.0)]

        self.assertEqual(best_result(history), (10.0, 2.0))
        self.assertEqual(best_result(None), (None, None))

    def test_table_has_parallel_ec_and_kl_columns(self):
        case = select_cases('IV', ('QFT',), (256,))[0]
        summary = {
            'cases': {
                case.key: {
                    'kl_seconds': 0.125,
                    'solvers': {
                        'TACOL': {
                            'best_objective': 100.0,
                            'time_to_best_seconds': 2.0,
                        },
                        'TACOPA': {
                            'best_objective': 90.0,
                            'time_to_best_seconds': 1.0,
                        },
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            rows = table_rows([case], summary, Path(directory))

        self.assertEqual(
            list(rows[0]),
            [
                'Task', 'EC', 'KL', 'TACO-L+EC', 'TACO-L+KL',
                'TACO-L+EC+PA', 'TACO-L+KL+PA',
            ],
        )
        self.assertEqual(rows[0]['TACO-L+KL'], '100 / 2.0')
        self.assertEqual(rows[0]['TACO-L+KL+PA'], '90 / 1.0')

    def test_old_kl_checkpoint_is_migrated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'summary.json'
            path.write_text(
                '{"format_version": 1, "cases": {'
                '"IV:QFT-256": {"kl_seconds": 0.1, "kl_seed": 42, '
                '"partition_count": 16, "cut_demand": 10, "solvers": {}}}}'
            )

            summary = load_summary(path)

        self.assertEqual(summary['format_version'], 2)
        self.assertEqual(
            summary['cases']['IV:QFT-256']['methods']['KL']['seconds'],
            0.1,
        )

    def test_better_pa_result_is_bold_in_markdown_and_latex(self):
        case = select_cases('IV', ('QFT',), (256,))[0]
        summary = {
            'cases': {
                case.key: {
                    'methods': {
                        'EC': {'solvers': {'TACOPA': {
                            'best_objective': 90.0,
                            'time_to_best_seconds': 20.0,
                        }}},
                        'KL': {'solvers': {'TACOPA': {
                            'best_objective': 90.0,
                            'time_to_best_seconds': 10.0,
                        }}},
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = table_rows([case], summary, output)
            render_table('IV', rows, output)
            markdown = (output / 'table-IV.md').read_text()
            latex = (output / 'table-IV.tex').read_text()
            csv_text = (output / 'table-IV.csv').read_text()

        self.assertEqual(rows[0].bold_columns, {'TACO-L+KL+PA'})
        self.assertIn('**90 / 10.0**', markdown)
        self.assertNotIn('**90 / 20.0**', markdown)
        self.assertIn(r'\textbf{90 / 10.0}', latex)
        self.assertNotIn(r'\textbf{90 / 20.0}', latex)
        self.assertNotIn('**', csv_text)

    def test_smaller_objective_wins_even_if_it_takes_longer(self):
        from src.scripts.partition_comparison import better_result_columns

        winners = better_result_columns(
            'EC', (89.0, 100.0),
            'KL', (90.0, 1.0),
        )

        self.assertEqual(winners, {'EC'})

    def test_runner_executes_both_methods_and_checkpoints(self):
        case = select_cases('IV', ('QFT',), (256,))[0]
        fake_result = {
            'best_objective': 10.0,
            'time_to_best_seconds': 1.0,
            'build_seconds': 0.1,
            'solve_seconds': 1.0,
            'gurobi_status': 2,
            'solution_count': 1,
            'mip_gap': 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch(
                    'src.scripts.partition_comparison.run_solver',
                    side_effect=lambda *args: (
                        fake_result.copy(), [(1.0, 10.0)], [],
                    ),
                    ) as mocked_solver:
                summary = run_experiments(
                    [case], ('EC', 'KL'), ('TACOL', 'TACOPA'),
                    total_time_limit=1, seed=42, output_dir=output,
                    overwrite=False,
                )

            methods = summary['cases'][case.key]['methods']
            self.assertEqual(set(methods), {'EC', 'KL'})
            self.assertEqual(summary['total_budget_seconds'], 1)
            self.assertEqual(mocked_solver.call_count, 4)
            for call in mocked_solver.call_args_list:
                self.assertGreater(call.args[3], 0)
                self.assertLess(call.args[3], 1)
            for method in methods.values():
                for result in method['solvers'].values():
                    self.assertEqual(result['total_budget_seconds'], 1)
                    self.assertLess(result['solver_time_limit_seconds'], 1)
            self.assertTrue((output / 'objs-QFT-256-TACOL-EC.pkl').exists())
            self.assertTrue((output / 'objs-QFT-256-TACOL-KL.pkl').exists())


if __name__ == '__main__':
    unittest.main()
