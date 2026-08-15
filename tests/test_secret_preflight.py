from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from egoverse_modal.secret_preflight import validate


class SecretPreflightTests(unittest.TestCase):
    def test_placeholder_values_are_rejected_without_returning_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "AWS_ACCESS_KEY_ID=replace-with-key\n"
                "AWS_SECRET_ACCESS_KEY=real-looking\n"
                "AWS_DEFAULT_REGION=us-east-1\n"
                "SECRETS_ARN=replace-with-arn\n"
                "R2_ENDPOINT_URL=https://example.invalid\n"
                "R2_ACCESS_KEY_ID=real-looking\n"
                "R2_SECRET_ACCESS_KEY=real-looking\n"
            )
            self.assertEqual(
                validate(path, "egoverse"),
                ["AWS_ACCESS_KEY_ID", "SECRETS_ARN"],
            )


if __name__ == "__main__":
    unittest.main()
