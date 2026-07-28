import os
import unittest
from unittest.mock import patch

from scripts.container_start import env_flag, validated_port


class ContainerStartupTests(unittest.TestCase):
    def test_env_flag_uses_default_and_accepts_true_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(env_flag("GRAPHMIND_BOOTSTRAP_DEMO", default=True))
            self.assertFalse(env_flag("GRAPHMIND_BOOTSTRAP_DEMO"))

        for value in ("1", "true", "YES", "On"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"GRAPHMIND_BOOTSTRAP_DEMO": value}, clear=True
            ):
                self.assertTrue(env_flag("GRAPHMIND_BOOTSTRAP_DEMO"))

    def test_validated_port_accepts_valid_port(self):
        with patch.dict(os.environ, {"PORT": "8501"}, clear=True):
            self.assertEqual(validated_port(), 8501)

    def test_validated_port_rejects_invalid_values(self):
        for value in ("invalid", "0", "65536"):
            with self.subTest(value=value), patch.dict(os.environ, {"PORT": value}, clear=True):
                with self.assertRaises(RuntimeError):
                    validated_port()


if __name__ == "__main__":
    unittest.main()
