from pathlib import Path

import nbformat as nbf
import pytest

from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun
from cloudai._core.test_template import TestTemplate
from cloudai.core import TestScenario
from cloudai.models.scenario import ReportConfig
from cloudai.models.workload import CmdArgs, TestDefinition
from cloudai.reporter import NotebookReport
from cloudai.systems.slurm.slurm_system import SlurmSystem


@pytest.fixture
def results_root(tmp_path: Path):
    (tmp_path / "test1" / "0").mkdir(parents=True)
    (tmp_path / "test1" / "1").mkdir(parents=True)
    (tmp_path / "test2" / "0").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def test_scenario(slurm_system: SlurmSystem):
    test_def1 = TestDefinition(
        name="test1", description="Test 1", test_template_name="mock_template", cmd_args=CmdArgs()
    )
    test_def2 = TestDefinition(
        name="test2", description="Test 2", test_template_name="mock_template", cmd_args=CmdArgs()
    )
    return TestScenario(
        name="test_scenario",
        test_runs=[
            TestRun(
                name="test1", iterations=2, test=Test(test_def1, TestTemplate(slurm_system)), num_nodes=1, nodes=[]
            ),
            TestRun(
                name="test2", iterations=1, test=Test(test_def2, TestTemplate(slurm_system)), num_nodes=1, nodes=[]
            ),
        ],
    )


def test_generate_scenario_notebook(test_scenario: TestScenario, slurm_system: SlurmSystem, results_root: Path):
    reporter = NotebookReport(slurm_system, test_scenario, results_root, ReportConfig(enable=True))
    reporter.generate()

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

    assert "| Test | Output dir |" in table_content
    assert "|------|--------|" in table_content

    assert f"| test1 | {results_root / 'test1' / '0'}" in table_content
    assert f"| test1 | {results_root / 'test1' / '1'}" in table_content
    assert f"| test2 | {results_root / 'test2' / '0'}" in table_content


def test_generate_scenario_notebook_empty_scenario(slurm_system: SlurmSystem, tmp_path: Path):
    empty_scenario = TestScenario(name="empty_scenario", test_runs=[])
    reporter = NotebookReport(slurm_system, empty_scenario, tmp_path, ReportConfig(enable=True))
    reporter.generate()

    notebook_path = tmp_path / "empty_scenario.ipynb"
    notebook = nbf.read(notebook_path, as_version=4)
    table_content = notebook.cells[1].source

    # Verify table only contains headers
    assert "| Test | Output dir |" in table_content
    assert "|------|--------|" in table_content
    assert len(table_content.split("\n")) == 3  # headers + separator + empty line
