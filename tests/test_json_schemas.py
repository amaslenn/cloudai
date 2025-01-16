import json
from pathlib import Path

import pytest

from cloudai import TestDefinition
from cloudai.test_definitions import (
    ChakraReplayTestDefinition,
    GPTTestDefinition,
    GrokTestDefinition,
    NCCLTestDefinition,
    NeMoLauncherTestDefinition,
    NeMoRunTestDefinition,
    NemotronTestDefinition,
    SleepTestDefinition,
    SlurmContainerTestDefinition,
    UCCTestDefinition,
)


@pytest.mark.parametrize(
    "tdef",
    [
        ChakraReplayTestDefinition,
        GPTTestDefinition,
        GrokTestDefinition,
        NCCLTestDefinition,
        NeMoLauncherTestDefinition,
        NeMoRunTestDefinition,
        NemotronTestDefinition,
        SleepTestDefinition,
        SlurmContainerTestDefinition,
        UCCTestDefinition,
    ],
)
def test_test_definition_schemas(tdef: type[TestDefinition]):
    schema_root = Path.cwd() / "conf" / "_schemas" / "tests"
    name = tdef.__name__.lower().replace("testdefinition", "")
    schema_file = schema_root / f"{name}.schema.json"
    assert schema_file.exists()
    assert tdef.model_json_schema() == json.loads(
        schema_file.read_text()
    ), f"Schema mismatch for {tdef.__name__}, consider running `cloudai json-schema --generate`"
