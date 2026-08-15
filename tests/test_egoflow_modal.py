from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from hackathon.egoflow import modal_app


class EgoFlowModalTests(unittest.TestCase):
    def test_resource_caps_match_the_hackathon_brief(self) -> None:
        self.assertEqual(modal_app.FEATURE_GPU, "L40S")
        self.assertEqual(modal_app.TRAIN_GPU, "H100")
        self.assertEqual(modal_app.MAX_FEATURE_WORKERS, 8)
        self.assertEqual(modal_app.MAX_SIMULTANEOUS_TRAINING_GPUS, 2)
        self.assertEqual(modal_app.TRAIN_TIMEOUT_SECONDS, 25 * 60)

    def test_largest_allowed_run_stays_below_cost_guard(self) -> None:
        estimate = modal_app.estimate_pipeline_cost(60, 3)
        self.assertLess(
            estimate["timeout_bound_usd"], modal_app.ESTIMATED_COST_GUARD_USD
        )
        modal_app.validate_run_request(60, 3, 2_500)

    def test_invalid_scale_requests_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_episodes"):
            modal_app.validate_run_request(61, 1, 750)
        with self.assertRaisesRegex(ValueError, "training_runs"):
            modal_app.validate_run_request(2, 4, 750)
        with self.assertRaisesRegex(ValueError, "max_steps"):
            modal_app.validate_run_request(2, 1, 2_501)

    def test_episode_selection_csv_preserves_order_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.csv"
            path.write_text(
                "episode_id,task,notes\nsecond,organizing_plushies,hero\n"
                "first,organizing_plushies,\nsecond,organizing_plushies,duplicate\n"
            )
            self.assertEqual(modal_app._read_episode_ids(str(path)), ["second", "first"])

    def test_episode_metadata_retains_human_review_fields(self) -> None:
        records = modal_app._read_episode_metadata(
            "hackathon/egoflow/episode_selection.csv"
        )
        self.assertEqual(len(records), 18)
        self.assertEqual(records[1]["episode_id"], "69bb1239efeadec2abedad96")
        self.assertEqual(records[1]["task"], "organizing_plushies")
        self.assertEqual(records[1]["review_status"], "reviewed")

    def test_nonblocking_poll_accepts_builtin_timeout(self) -> None:
        class PendingCall:
            def get(self, timeout: int) -> None:
                self.assert_timeout = timeout
                raise TimeoutError

        pending = {0: PendingCall()}
        completed: list[dict[str, object]] = []
        modal_app._finished_calls(pending, completed)
        self.assertEqual(list(pending), [0])
        self.assertEqual(completed, [])


if __name__ == "__main__":
    unittest.main()
