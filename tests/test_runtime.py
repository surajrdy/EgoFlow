from __future__ import annotations

import re
import unittest

from egoverse_modal.runtime import (
    build_egoverse_command,
    new_run_id,
    parse_override_string,
    validate_overrides,
)


class RuntimeTests(unittest.TestCase):
    def test_new_run_id_is_safe_and_sortable(self) -> None:
        self.assertRegex(new_run_id(), r"^\d{8}-\d{6}-[0-9a-f]{8}$")

    def test_parse_override_string_respects_quotes(self) -> None:
        self.assertEqual(
            parse_override_string('trainer=debug description="short run"'),
            ["trainer=debug", "description=short run"],
        )

    def test_command_adds_persistent_paths_and_debug_logger(self) -> None:
        run_id = "20260815-120000-deadbeef"
        command = build_egoverse_command(
            "train_zarr_cartesian",
            ["trainer=debug", "data=eva"],
            run_id,
        )
        self.assertEqual(command[0], "python")
        self.assertIn("logger=debug", command)
        self.assertIn("paths.dataset_dir=/vol/datasets", command)
        self.assertIn(f"hydra.run.dir=/vol/outputs/{run_id}", command)
        self.assertFalse(any(re.search(r"[;&|`]", token) for token in command))

    def test_explicit_logger_is_preserved(self) -> None:
        command = build_egoverse_command(
            "train_zarr_cartesian",
            ["logger=wandb"],
            "20260815-120000-deadbeef",
        )
        self.assertIn("logger=wandb", command)
        self.assertNotIn("logger=debug", command)

    def test_reserved_paths_cannot_be_overridden(self) -> None:
        with self.assertRaisesRegex(ValueError, "managed by the Modal runner"):
            validate_overrides(["hydra.run.dir=/tmp/elsewhere"])

    def test_cli_flags_are_not_accepted_as_overrides(self) -> None:
        with self.assertRaisesRegex(ValueError, "key=value"):
            validate_overrides(["--multirun"])


if __name__ == "__main__":
    unittest.main()

