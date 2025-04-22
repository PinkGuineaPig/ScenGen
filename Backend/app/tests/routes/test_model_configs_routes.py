# Backend/app/tests/routes/test_model_configs_routes.py

import pytest
from datetime import datetime
from Backend.app.models.run_models import (
    ModelRunConfig,
    ModelRun,
    ModelLossHistory
)

def test_list_model_configs_empty(client):
    """
    When there are no ModelRunConfig rows, GET /api/model_configs should return an empty list.
    """
    resp = client.get("/api/model_configs")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_full_model_with_relations(client, session):
    """
    Create one config, one run, and one loss entry.
    GET /full_model/<id> should return the config plus
    'runs' and 'loss_history' arrays populated.
    """
    # 1) Insert a config
    cfg = ModelRunConfig(model_type="test", parameters={"a": 1})
    session.add(cfg)
    session.flush()                # populate cfg.id

    # 2) Link one run
    run = ModelRun(
        config_id=cfg.id,
        group_ids=[1, 2, 3],
        version=1,
        model_blob=b"bytes"
    )
    session.add(run)

    # 3) Link one loss history entry
    loss = ModelLossHistory(
        config_id=cfg.id,
        epoch=0,
        loss_type="mse",
        value=0.01
    )
    session.add(loss)

    session.commit()

    FULL_ROUTE = "/api/model_configs/full_model/{}"

    # 4) Call the endpoint
    resp = client.get(FULL_ROUTE.format(cfg.id))
    assert resp.status_code == 200

    payload = resp.get_json()
    # Top–level config fields
    assert payload["id"] == cfg.id
    assert payload["model_type"] == "test"
    assert payload["parameters"] == {"a": 1}
    assert "created_at" in payload

    # Runs should be a list with exactly our run, using the frontend serializer
    assert isinstance(payload["runs"], list) and len(payload["runs"]) == 1
    run_obj = payload["runs"][0]
    assert run_obj["id"] == run.id
    assert run_obj["group_ids"] == [1, 2, 3]
    assert run_obj["version"] == 1

    # Loss history likewise
    assert isinstance(payload["loss_history"], list) and len(payload["loss_history"]) == 1
    loss_obj = payload["loss_history"][0]
    assert loss_obj == loss.to_dict()
