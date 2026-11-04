import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_FULL_APP_TEST") != "1",
    reason="Set RUN_FULL_APP_TEST=1 when complete prepared/model artifacts are available",
)


def test_single_page_streamlit_ui() -> None:
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.tabs) == 0
    assert next(slider for slider in app.slider if slider.label == "Five-year share of flexible deals").value == 50
    assert next(metric for metric in app.metric if metric.label == "Scenario expected commitment").value
    next(slider for slider in app.slider if slider.label == "Five-year share of flexible deals").set_value(80).run()
    assert not app.exception
