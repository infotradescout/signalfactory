import unittest
from datetime import datetime, timezone

import pandas as pd

from src.pipeline import SignalPipelineEngine
from src.signals.schema import NormalizedSignal
from src.signals.scorer import score_signal


class SignalPipelineTests(unittest.TestCase):
    def test_raw_to_packet_trace_exists(self):
        wb_df = pd.DataFrame(
            {
                "country": ["US", "US"],
                "year": [2024, 2025],
                "inflation": [3.1, 2.7],
            }
        )
        cfg = {
            "id": "inflation_forecast",
            "lane": "macro",
            "signal_kind": "forecast_support",
            "output_unit": "%",
            "countries": ["US"],
            "target_column": "inflation",
        }
        engine = SignalPipelineEngine(lane_cfg={"retention_decay_hours": 120})
        out = engine.run(cfg, wb_df=wb_df, fred_df=None, extra_dfs=[])

        self.assertGreater(len(out["raw_events"]), 0)
        self.assertGreater(len(out["normalized_signals"]), 0)
        self.assertEqual(out["packet"]["lane"], "macro")
        self.assertGreaterEqual(len(out["packet"]["signals"]), 1)

    def test_scoring_fields_present(self):
        signal = NormalizedSignal(
            signal_id="s1",
            lane="risk",
            signal_pack="risk_pack",
            entity="US",
            metric="stress",
            value=0.7,
            unit="score",
            event_time=datetime.now(timezone.utc),
            publish_time=datetime.now(timezone.utc),
            source_id="source-1",
            source_type="structured_data",
            source_name="world_bank",
            raw_event_ref="evt-1",
        )
        scored = score_signal(signal, corroboration_count=2, contradiction_count=1)

        self.assertTrue(0.0 <= scored.truth_score <= 1.0)
        self.assertTrue(0.0 <= scored.novelty_score <= 1.0)
        self.assertTrue(0.0 <= scored.recency_score <= 1.0)
        self.assertTrue(0.0 <= scored.relevance_score <= 1.0)
        self.assertTrue(0.0 <= scored.corroboration_score <= 1.0)
        self.assertTrue(0.0 <= scored.contradiction_risk <= 1.0)
        self.assertGreaterEqual(len(scored.confidence_reasons), 1)


if __name__ == "__main__":
    unittest.main()