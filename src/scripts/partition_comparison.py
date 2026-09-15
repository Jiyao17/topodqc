"""Run the EC/KL experiments used to extend Tables IV and V.

Legacy EC results in ``result/efficiency`` are used only as a fallback. New
controlled EC/KL histories and table-ready CSV, Markdown, and LaTeX files are
written under ``result/partition_comparison`` by default.
"""

import argparse
import copy
import csv
import json
import pickle
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from src.circuit import QASM_FILES, QIG


TASKS = ('MCMT', 'QFT', 'Grover')
METHODS = ('EC', 'KL')
SOLVERS = ('TACOL', 'TACOPA')


@dataclass(frozen=True)
class ExperimentCase:
    table: str
    task: str
    size: int
    mems: tuple[int, ...]
    comms: tuple[int, ...]
    W: int

    @property
    def key(self) -> str:
        return f'{self.table}:{self.task}-{self.size}'


def make_cases() -> list[ExperimentCase]:
    """Return the exact experiment matrix from manuscript Tables II/IV/V."""
    settings = {
        # Table IV: homogeneous QPU clusters.
        ('IV', 256): ((16,) * 16, (4,) * 16, 40),
        ('IV', 512): ((32,) * 16, (8,) * 16, 40),
        ('IV', 1024): ((32,) * 32, (8,) * 32, 80),
        # Table V: heterogeneous QPU clusters.
        ('V', 192): ((16,) * 8 + (8,) * 8, (4,) * 8 + (3,) * 8, 20),
        ('V', 384): ((16,) * 16 + (8,) * 16, (4,) * 16 + (3,) * 16, 80),
        ('V', 768): ((32,) * 16 + (16,) * 16, (8,) * 16 + (4,) * 16, 80),
    }
    table_sizes = {
        'IV': (256, 512, 1024),
        'V': (192, 384, 768),
    }
    return [
        ExperimentCase(table, task, size, *settings[table, size])
        for table in ('IV', 'V')
        for task in TASKS
        for size in table_sizes[table]
    ]


def select_cases(
        table: str = 'all',
        tasks: tuple[str, ...] = TASKS,
        sizes: tuple[int, ...] | None = None,
        ) -> list[ExperimentCase]:
    selected = []
    for case in make_cases():
        if table != 'all' and case.table != table:
            continue
        if case.task not in tasks:
            continue
        if sizes is not None and case.size not in sizes:
            continue
        selected.append(case)
    return selected


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {'format_version': 2, 'cases': {}}
    with path.open() as handle:
        summary = json.load(handle)

    legacy_budget = summary.get(
        'total_budget_seconds',
        summary.get('timeout_seconds'),
    )

    # Migrate the original KL-only checkpoint schema in memory. Existing KL
    # solver runs remain resumable and do not need to be repeated.
    for case in summary.get('cases', {}).values():
        if 'methods' in case:
            continue
        if 'kl_seconds' in case:
            case['methods'] = {
                'KL': {
                    'seconds': case.pop('kl_seconds'),
                    'seed': case.pop('kl_seed', 42),
                    'partition_count': case.pop('partition_count', None),
                    'cut_demand': case.pop('cut_demand', None),
                    'solvers': case.pop('solvers', {}),
                },
            }
        for method in case.get('methods', {}).values():
            for result in method.get('solvers', {}).values():
                if legacy_budget is not None:
                    result.setdefault('total_budget_seconds', legacy_budget)
    summary['format_version'] = 2
    return summary


def save_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')
    temporary.replace(path)


def best_result(history) -> tuple[float | None, float | None]:
    if not history:
        return None, None
    best_objective = min(objective for _, objective in history)
    time_to_best = min(
        elapsed for elapsed, objective in history
        if objective == best_objective
    )
    return float(best_objective), float(time_to_best)


def validate_partition(qig: QIG, case: ExperimentCase, method: str = 'KL') -> None:
    assigned = [
        qubit
        for node in qig.graph.nodes
        for qubit in qig.graph.nodes[node]['qubits']
    ]
    if sorted(assigned) != list(range(case.size)):
        raise RuntimeError(
            f'{case.key}: {method} did not preserve every logical qubit exactly once.'
        )

    sizes = sorted(
        (len(qig.graph.nodes[node]['qubits']) for node in qig.graph.nodes),
        reverse=True,
    )
    remaining = sorted(case.mems, reverse=True)
    for size in sizes:
        if not remaining or size > remaining[0]:
            raise RuntimeError(
                f'{case.key}: {method} partitions violate QPU memory capacity.'
            )
        remaining[0] -= size
        remaining.sort(reverse=True)


def solver_class(name: str):
    # Import lazily so --dry-run and --render-only do not require a Gurobi
    # license checkout.
    from src.solver import TACOL, TACOPA

    return {'TACOL': TACOL, 'TACOPA': TACOPA}[name]


def run_solver(
        name: str,
        qig: QIG,
        case: ExperimentCase,
        timeout: float,
        ) -> tuple[dict, list | None, list | None]:
    SolverClass = solver_class(name)
    solver = SolverClass(
        copy.deepcopy(qig),
        list(case.mems),
        list(case.comms),
        case.W,
        None,
        timeout,
    )

    build_started = time.perf_counter()
    solver.build()
    build_seconds = time.perf_counter() - build_started

    solve_started = time.perf_counter()
    solver.solve()
    solve_seconds = time.perf_counter() - solve_started

    history = solver.get_objs()
    topology = solver.get_topology()
    objective, time_to_best = best_result(history)
    model = solver.model
    result = {
        'best_objective': objective,
        'time_to_best_seconds': time_to_best,
        'build_seconds': build_seconds,
        'solve_seconds': solve_seconds,
        'gurobi_status': getattr(model, 'Status', None),
        'solution_count': getattr(model, 'SolCount', None),
        'mip_gap': getattr(model, 'MIPGap', None) if getattr(model, 'SolCount', 0) else None,
    }
    return result, history, topology


def run_experiments(
        cases: list[ExperimentCase],
        methods: tuple[str, ...],
        solvers: tuple[str, ...],
        total_time_limit: float,
        seed: int,
        output_dir: Path,
        overwrite: bool,
        preprocess_only: bool = False,
        ) -> dict:
    summary_path = output_dir / 'summary.json'
    summary = load_summary(summary_path)
    summary['environment'] = {
        'python': platform.python_version(),
        'networkx': nx.__version__,
    }
    summary['total_budget_seconds'] = total_time_limit
    summary.pop('timeout_seconds', None)
    output_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        case_result = summary['cases'].setdefault(case.key, {
            'table': case.table,
            'task': case.task,
            'size': case.size,
            'mems': list(case.mems),
            'comms': list(case.comms),
            'W': case.W,
            'methods': {},
        })
        case_result.setdefault('methods', {})
        summary['cases'][case.key] = case_result

        for method in methods:
            existing_method = case_result['methods'].get(method, {})
            completed = existing_method.get('solvers', {})
            requested_complete = all(
                name in completed
                and completed[name].get('total_budget_seconds') == total_time_limit
                for name in solvers
            )
            same_configuration = method == 'EC' or existing_method.get('seed') == seed
            if requested_complete and same_configuration and not overwrite and not preprocess_only:
                print(f'[skip] {case.key}/{method}: requested results already exist')
                continue
            if preprocess_only and completed and not same_configuration:
                raise ValueError(
                    f'{case.key}/{method}: cannot retain solver results while changing '
                    'the preprocessing configuration.'
                )

            print(f'[{method}] {case.key}: preprocessing {case.size} logical qubits')
            qig = QIG.from_qasm(QASM_FILES[case.task, case.size])
            preprocess_started = time.perf_counter()
            if method == 'EC':
                qig.contract_hdware_constrained(list(case.mems), inplace=True)
            else:
                qig.partition_kernighan_lin(list(case.mems), inplace=True, seed=seed)
            preprocess_seconds = time.perf_counter() - preprocess_started
            validate_partition(qig, case, method)

            cut_demand = sum(
                data['demand'] for _, _, data in qig.graph.edges(data=True)
            )
            method_result = {
                'seconds': preprocess_seconds,
                'partition_count': qig.graph.number_of_nodes(),
                'cut_demand': cut_demand,
                'solvers': completed if preprocess_only or (
                    same_configuration and not overwrite
                ) else {},
            }
            if method == 'KL':
                method_result['seed'] = seed
            case_result['methods'][method] = method_result
            save_summary(summary_path, summary)

            print(
                f'     {preprocess_seconds:.3f}s, '
                f'{qig.graph.number_of_nodes()} partitions, cut demand {cut_demand}'
            )
            if preprocess_only:
                continue
            solver_time_limit = max(0.0, total_time_limit - preprocess_seconds)
            for name in solvers:
                existing_solver = method_result['solvers'].get(name)
                if (
                    existing_solver is not None
                    and existing_solver.get('total_budget_seconds') == total_time_limit
                    and not overwrite
                ):
                    print(f'[skip] {case.key}/{method}/{name}: result already exists')
                    continue

                label = f"TACO-L+{method}{'+PA' if name == 'TACOPA' else ''}"
                print(
                    f'[run]  {case.key}/{label} '
                    f'(total={total_time_limit:.1f}s, solver={solver_time_limit:.3f}s)'
                )
                if solver_time_limit <= 0:
                    result = {
                        'best_objective': None,
                        'time_to_best_seconds': None,
                        'build_seconds': 0.0,
                        'solve_seconds': 0.0,
                        'gurobi_status': None,
                        'solution_count': 0,
                        'mip_gap': None,
                        'skip_reason': 'preprocessing exhausted the total time budget',
                    }
                    history, topology = None, None
                else:
                    result, history, topology = run_solver(
                        name, qig, case, solver_time_limit,
                    )
                result['total_budget_seconds'] = total_time_limit
                result['preprocess_seconds'] = preprocess_seconds
                result['solver_time_limit_seconds'] = solver_time_limit
                history_name = f'objs-{case.task}-{case.size}-{name}-{method}.pkl'
                topology_name = f'topology-{case.task}-{case.size}-{name}-{method}.pkl'
                with (output_dir / history_name).open('wb') as handle:
                    pickle.dump(history, handle)
                with (output_dir / topology_name).open('wb') as handle:
                    pickle.dump(topology, handle)
                result['history_file'] = history_name
                result['topology_file'] = topology_name
                method_result['solvers'][name] = result
                save_summary(summary_path, summary)

                if result['best_objective'] is None:
                    print('     no feasible solution')
                else:
                    print(
                        f"     objective={result['best_objective']:.0f}, "
                        f"time-to-best={result['time_to_best_seconds']:.3f}s"
                    )

    return summary


def load_legacy_ec(ec_dir: Path) -> dict:
    time_path = ec_dir / 'contraction_times.pkl'
    contraction_times = {}
    if time_path.exists():
        with time_path.open('rb') as handle:
            contraction_times = pickle.load(handle)
    return contraction_times


def legacy_solver_result(ec_dir: Path, case: ExperimentCase, solver: str):
    path = ec_dir / f'objs-{case.task}-{case.size}-{solver}.pkl'
    if not path.exists():
        return None, None
    with path.open('rb') as handle:
        return best_result(pickle.load(handle))


def format_time(value, digits: int = 1) -> str:
    return '-' if value is None else f'{value:.{digits}f}'


def format_result(objective, elapsed) -> str:
    if objective is None:
        return '-'
    return f'{objective:.0f} / {elapsed:.1f}'


class TableRow(dict):
    """Table data plus output-formatting metadata excluded from CSV fields."""

    def __init__(self, values, bold_columns=()):
        super().__init__(values)
        self.bold_columns = set(bold_columns)


def better_result_columns(left_column, left_result, right_column, right_result):
    """Select the better feasible result by objective, then time-to-best."""
    candidates = []
    for column, (objective, elapsed) in (
            (left_column, left_result),
            (right_column, right_result),
            ):
        if objective is not None and elapsed is not None:
            candidates.append((column, (objective, elapsed)))

    if not candidates:
        return set()
    best_key = min(key for _, key in candidates)
    return {column for column, key in candidates if key == best_key}


def table_rows(
        cases: list[ExperimentCase],
        summary: dict,
        ec_dir: Path,
        ) -> list[dict[str, str]]:
    contraction_times = load_legacy_ec(ec_dir)
    rows = []
    for case in cases:
        case_result = summary.get('cases', {}).get(case.key, {})
        methods = case_result.get('methods', {})
        # Accept an unmigrated in-memory dictionary in unit tests and callers.
        if not methods and 'kl_seconds' in case_result:
            methods = {
                'KL': {
                    'seconds': case_result['kl_seconds'],
                    'solvers': case_result.get('solvers', {}),
                },
            }
        ec = methods.get('EC', {})
        kl = methods.get('KL', {})
        ec_solvers = ec.get('solvers', {})
        kl_solvers = kl.get('solvers', {})
        legacy_tacol = legacy_solver_result(ec_dir, case, 'TACOL')
        legacy_tacopa = legacy_solver_result(ec_dir, case, 'TACOPA')

        def result_for(method_solvers, name, fallback=(None, None)):
            result = method_solvers.get(name)
            if result is None:
                return fallback
            return result.get('best_objective'), result.get('time_to_best_seconds')

        ec_tacol = result_for(ec_solvers, 'TACOL', legacy_tacol)
        ec_tacopa = result_for(ec_solvers, 'TACOPA', legacy_tacopa)
        kl_tacol = result_for(kl_solvers, 'TACOL')
        kl_tacopa = result_for(kl_solvers, 'TACOPA')
        bold_columns = better_result_columns(
            'TACO-L+EC+PA', ec_tacopa,
            'TACO-L+KL+PA', kl_tacopa,
        )
        rows.append(TableRow(
            {
                'Task': f'{case.task}-{case.size}',
                'EC': format_time(
                    ec.get('seconds', contraction_times.get((case.task, case.size))),
                    digits=3 if ec else 1,
                ),
                'KL': format_time(kl.get('seconds'), digits=3),
                'TACO-L+EC': format_result(*ec_tacol),
                'TACO-L+KL': format_result(*kl_tacol),
                'TACO-L+EC+PA': format_result(*ec_tacopa),
                'TACO-L+KL+PA': format_result(*kl_tacopa),
            },
            bold_columns=bold_columns,
        ))
    return rows


def render_table(table: str, rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        return
    fields = list(rows[0])

    csv_path = output_dir / f'table-{table}.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        '| ' + ' | '.join(fields) + ' |',
        '| ' + ' | '.join('---' for _ in fields) + ' |',
    ]

    def markdown_cell(row, field):
        value = row[field]
        if field in getattr(row, 'bold_columns', ()) and value != '-':
            return f'**{value}**'
        return value

    markdown.extend(
        '| ' + ' | '.join(markdown_cell(row, field) for field in fields) + ' |'
        for row in rows
    )
    markdown_text = '\n'.join(markdown) + '\n'
    (output_dir / f'table-{table}.md').write_text(markdown_text)

    latex_fields = [field.replace('+', r'\,+\,') for field in fields]
    latex = [
        r'\begin{tabular}{lrrrrrr}',
        r'\hline',
        ' & '.join(latex_fields) + r' \\',
        r'\hline',
    ]

    def latex_cell(row, field):
        value = row[field]
        if field in getattr(row, 'bold_columns', ()) and value != '-':
            return rf'\textbf{{{value}}}'
        return value

    latex.extend(
        ' & '.join(latex_cell(row, field) for field in fields) + r' \\'
        for row in rows
    )
    latex.extend([r'\hline', r'\end{tabular}', ''])
    (output_dir / f'table-{table}.tex').write_text('\n'.join(latex))

    print(f'\nTable {table}')
    print(markdown_text)


def render_tables(
        cases: list[ExperimentCase],
        summary: dict,
        ec_dir: Path,
        output_dir: Path,
        ) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for table in ('IV', 'V'):
        table_cases = [case for case in cases if case.table == table]
        render_table(table, table_rows(table_cases, summary, ec_dir), output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', choices=('IV', 'V', 'all'), default='all')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--sizes', nargs='+', type=int)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--solvers', nargs='+', choices=SOLVERS, default=list(SOLVERS))
    parser.add_argument(
        '--total-time-limit', '--timeout',
        dest='total_time_limit',
        type=float,
        default=600.0,
        help='total preprocessing plus solver budget in seconds (default: 600)',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('result/partition_comparison'))
    parser.add_argument('--ec-results', type=Path, default=Path('result/efficiency'))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--preprocess-only',
        action='store_true',
        help='refresh EC/KL timings without running or deleting solver results',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_time_limit <= 0:
        raise SystemExit('--total-time-limit must be positive.')
    cases = select_cases(args.table, tuple(args.tasks), tuple(args.sizes) if args.sizes else None)
    if not cases:
        raise SystemExit('No experiment cases match the requested filters.')

    if args.dry_run:
        for case in cases:
            print(
                f'{case.key}: {len(case.mems)} QPUs, W={case.W}, '
                f"methods={','.join(args.methods)}, solvers={','.join(args.solvers)}"
            )
        return

    summary_path = args.output / 'summary.json'
    if args.render_only:
        summary = load_summary(summary_path)
    else:
        summary = run_experiments(
            cases,
            tuple(args.methods),
            tuple(args.solvers),
            args.total_time_limit,
            args.seed,
            args.output,
            args.overwrite,
            args.preprocess_only,
        )
    render_tables(cases, summary, args.ec_results, args.output)


if __name__ == '__main__':
    main()
