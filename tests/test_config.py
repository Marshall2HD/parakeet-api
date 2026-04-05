from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parakeet_api.config import get_settings


class ConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        get_settings.cache_clear()

    def test_local_attention_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            settings = get_settings()
        self.assertTrue(settings.use_local_attention)
        self.assertEqual(settings.attention_model, "rel_pos_local_attn")
        self.assertEqual(settings.att_context_size, (128, 128))

    def test_local_attention_override(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PARAKEET_USE_LOCAL_ATTENTION": "false",
                "PARAKEET_ATT_CONTEXT_SIZE": "[64, 32]",
            },
            clear=True,
        ):
            get_settings.cache_clear()
            settings = get_settings()
        self.assertFalse(settings.use_local_attention)
        self.assertEqual(settings.att_context_size, (64, 32))


if __name__ == "__main__":
    unittest.main()
