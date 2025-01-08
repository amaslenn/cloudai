# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from pathlib import Path
from pydantic import ValidationError

from cloudai._core.system import System
from cloudai._core.test import Test, TestTemplate
from cloudai.test_definitions.grok import GrokTestDefinition, GrokCmdArgs, GrokFdl, GrokProfileXLAFlags, GrokPerfXLAFlags


class MockSystem(System):
    def update(self) -> None:
        pass

    def is_job_running(self, job: "BaseJob") -> bool:
        return False

    def is_job_completed(self, job: "BaseJob") -> bool:
        return True

    def kill(self, job: "BaseJob") -> None:
        pass


class MockTestTemplate(TestTemplate):
    pass


class GrokTestDefinitionWrapper(GrokTestDefinition):
    def __init__(self, **data):
        super().__init__(**data)
        self._cmd_args_dict = data.get("cmd_args", {})

    @property
    def cmd_args_dict(self):
        return self._cmd_args_dict

    @property
    def installables(self):
        return []


def test_grok_cmd_args_with_mixed_values():
    data = {
        "name": "example Grok",
        "description": "Example Grok",
        "test_template_name": "ExampleEnv",
        "cmd_args": {
            "fdl": GrokFdl(),
            "fdl_config": "paxml.tasks.lm.params.nvidia.Grok_Proxy",
            "enable_pgle": True,
            "setup_flags": {},
            "profile": GrokProfileXLAFlags(
                xla_gpu_disable_async_collectives="ALLREDUCE,ALLGATHER,REDUCESCATTER,COLLECTIVEBROADCAST,ALLTOALL,COLLECTIVEPERMUTE",
                xla_gpu_run_post_layout_collective_pipeliner=False,
                xla_gpu_enable_latency_hiding_scheduler=False,
            ),
            "perf": GrokPerfXLAFlags(
                combine_threshold_bytes=[301989888, 401989888, 501989888],
                xla_gpu_run_post_layout_collective_pipeliner=False,
                xla_gpu_use_memcpy_local_p2p=False,
                xla_gpu_pgle_profile_file_or_directory_path="/opt/paxml/workspace/pgle_output_profile.pbtxt",
            ),
            "docker_image_url": "docker://example_image",
            "output_path": "/path/to/output",
        },
        "extra_env_vars": {
            "ENV1": "0",
            "ENV2": "1",
            "ENV3": "3221225472",
        },
    }

    test_def = GrokTestDefinitionWrapper(**data)

    assert test_def.cmd_args_dict == {
        "fdl": GrokFdl(),
        "fdl_config": "paxml.tasks.lm.params.nvidia.Grok_Proxy",
        "enable_pgle": True,
        "setup_flags": {},
        "profile": GrokProfileXLAFlags(
            xla_gpu_disable_async_collectives="ALLREDUCE,ALLGATHER,REDUCESCATTER,COLLECTIVEBROADCAST,ALLTOALL,COLLECTIVEPERMUTE",
            xla_gpu_run_post_layout_collective_pipeliner=False,
            xla_gpu_enable_latency_hiding_scheduler=False,
        ),
        "perf": GrokPerfXLAFlags(
            combine_threshold_bytes=[301989888, 401989888, 501989888],
            xla_gpu_run_post_layout_collective_pipeliner=False,
            xla_gpu_use_memcpy_local_p2p=False,
            xla_gpu_pgle_profile_file_or_directory_path="/opt/paxml/workspace/pgle_output_profile.pbtxt",
        ),
        "docker_image_url": "docker://example_image",
        "output_path": "/path/to/output",
    }

    mock_system = MockSystem(
        name="mock_system",
        scheduler="mock_scheduler",
        install_path=Path("/mock/install/path"),
        output_path=Path("/mock/output/path"),
    )
    test_template = MockTestTemplate(system=mock_system, name="example_template")
    test = Test(test_definition=test_def, test_template=test_template)

    assert test.cmd_args == test_def.cmd_args_dict
