import importlib
import os
from unittest import mock


def test_perception_dry_run():
    os.environ["DRY_RUN"] = "true"
    cfg_mod = importlib.import_module("workspace.config")
    importlib.reload(cfg_mod)
    p = importlib.import_module("workspace.bitcore_research.perception_layer")
    importlib.reload(p)

    res = p.test_connection()
    assert res["success"] is True
    assert res["status_code"] == 200


def test_perception_real_call_mocked():
    # Simulate DRY_RUN false and mock requests.post
    os.environ["DRY_RUN"] = "false"
    cfg_mod = importlib.import_module("workspace.config")
    importlib.reload(cfg_mod)
    p = importlib.import_module("workspace.bitcore_research.perception_layer")
    importlib.reload(p)

    with mock.patch("workspace.bitcore_research.perception_layer.requests.post") as post:
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        post.return_value = mock_resp

        res = p.send_research_request(query="x")
        assert res["success"] is True
        assert res["status_code"] == 200
