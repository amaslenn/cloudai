from pathlib import Path

import nbformat as nbf
import pytest

from cloudai import BaseJob, Reporter, System, Test, TestRun, TestScenario, TestTemplate
from cloudai.systems.slurm.slurm_system import SlurmSystem
from tests.conftest import MyTestDefinition


class MockSystem(System):
    def update(self):
        pass

    def is_job_running(self, job: BaseJob) -> bool:
        return False

    def submit_job(self):
        pass

    def get_job_status(self):
        return "COMPLETED"

    def is_job_completed(self, job: BaseJob) -> bool:
        return True

    def kill(self, job: BaseJob) -> None:
        pass


@pytest.fixture
def test_scenario(slurm_system: SlurmSystem):
    test_def1 = MyTestDefinition(
        name="test1",
        description="Test 1",
        test_template_name="mock_template",
        cmd_args={},
    )
    test_def2 = MyTestDefinition(
        name="test2",
        description="Test 2",
        test_template_name="mock_template",
        cmd_args={},
    )
    return TestScenario(
        name="test_scenario",
        test_runs=[
            TestRun(
                name="test1",
                iterations=2,
                test=Test(test_def1, TestTemplate(slurm_system, "mock_template")),
                num_nodes=1,
                nodes=["node1"],
            ),
            TestRun(
                name="test2",
                iterations=1,
                test=Test(test_def2, TestTemplate(slurm_system, "mock_template")),
                num_nodes=1,
                nodes=["node1"],
            ),
        ],
    )


@pytest.fixture
def results_root(tmp_path: Path):
    # Create some mock log directories
    (tmp_path / "test1" / "0").mkdir(parents=True)
    (tmp_path / "test1" / "1").mkdir(parents=True)
    (tmp_path / "test2" / "0").mkdir(parents=True)
    return tmp_path


def test_generate_scenario_notebook(test_scenario: TestScenario, slurm_system: SlurmSystem, results_root: Path):
    reporter = Reporter(slurm_system, test_scenario, results_root)
    reporter.generate_scenario_notebook()

    notebook_path = results_root / f"{test_scenario.name}.ipynb"
    assert notebook_path.exists()

    notebook = nbf.read(notebook_path, as_version=4)
    assert len(notebook.cells) == 2

    title_cell = notebook.cells[0]
    assert title_cell.cell_type == "markdown"
    assert "# test_scenario" in title_cell.source
    assert "## Test Results Summary" in title_cell.source

    table_cell = notebook.cells[1]
    assert table_cell.cell_type == "markdown"
    table_content = table_cell.source

    assert "| Test | Status |" in table_content
    assert "|------|--------|" in table_content

    assert "| test1.0 |" in table_content
    assert "| test1.1 |" in table_content
    assert "| test2.0 |" in table_content

    assert "[logs](./test1/0)" in table_content
    assert "[logs](./test1/1)" in table_content
    assert "[logs](./test2/0)" in table_content


def test_generate_scenario_notebook_no_logs(test_scenario: TestScenario, slurm_system: SlurmSystem, tmp_path: Path):
    reporter = Reporter(slurm_system, test_scenario, tmp_path)
    reporter.generate_scenario_notebook()

    notebook_path = tmp_path / f"{test_scenario.name}.ipynb"
    notebook = nbf.read(notebook_path, as_version=4)
    table_content = notebook.cells[1].source

    # Verify "no logs" is shown when logs don't exist
    assert "| test1.0 | no logs |" in table_content
    assert "| test1.1 | no logs |" in table_content
    assert "| test2.0 | no logs |" in table_content


def test_generate_scenario_notebook_empty_scenario(slurm_system: SlurmSystem, tmp_path: Path):
    empty_scenario = TestScenario(name="empty_scenario", test_runs=[])
    reporter = Reporter(slurm_system, empty_scenario, tmp_path)
    reporter.generate_scenario_notebook()

    notebook_path = tmp_path / "empty_scenario.ipynb"
    notebook = nbf.read(notebook_path, as_version=4)
    table_content = notebook.cells[1].source

    # Verify table only contains headers
    assert "| Test | Status |" in table_content
    assert "|------|--------|" in table_content
    assert len(table_content.split("\n")) == 3  # headers + separator + empty line
