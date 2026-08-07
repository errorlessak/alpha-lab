"""The run read surface (§14.1).

Every representation here is immutable, so the caching conventions are part of
the contract rather than an optimisation: `ETag`, `Cache-Control: immutable`,
and `If-None-Match` returning 304.
"""

from __future__ import annotations

import pytest

IMMUTABLE = "immutable"


def test_the_index_lists_the_run_with_its_headline(client, run_id):
    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1

    row = next(item for item in body["runs"] if item["run_id"] == run_id)
    assert row["status"] == "completed"
    assert row["has_tearsheet"] is True
    assert row["selected_model"]
    assert row["oos_net_sharpe"] is not None
    assert row["config_hash"]
    assert row["symbols"] == ["SYNTH"]


def test_the_index_can_be_filtered_by_symbol(client, run_id):
    assert client.get("/api/v1/runs", params={"symbol": "SYNTH"}).json()["total"] >= 1
    assert client.get("/api/v1/runs", params={"symbol": "NOSUCH"}).json()["total"] == 0


def test_the_index_paginates_with_an_opaque_cursor(client):
    # How many runs exist depends on what else in this suite has submitted a
    # job, so the assertion is about the pagination contract rather than a
    # count: a cursor is offered exactly when there is more to fetch, and
    # following it returns different rows.
    first = client.get("/api/v1/runs", params={"limit": 1}).json()
    assert len(first["runs"]) == 1
    assert (first["next_cursor"] is not None) == (first["total"] > 1)

    if first["next_cursor"] is not None:
        second = client.get(
            "/api/v1/runs", params={"limit": 1, "cursor": first["next_cursor"]}
        ).json()
        assert second["runs"][0]["run_id"] != first["runs"][0]["run_id"]
        assert second["total"] == first["total"]


def test_a_cursor_this_service_did_not_issue_is_rejected(client):
    response = client.get("/api/v1/runs", params={"cursor": "not-a-cursor"})
    assert response.status_code == 400
    assert "cursor" in response.json()["detail"].lower()


def test_the_tearsheet_is_served_whole(client, run_id):
    response = client.get(f"/api/v1/runs/{run_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == run_id
    assert body["schema_version"]
    assert body["leaderboard"]
    assert body["dataset"]["periods_per_year"] > 250
    assert body["disclaimer"]


def test_a_run_resource_is_cacheable_forever(client, run_id):
    response = client.get(f"/api/v1/runs/{run_id}")
    assert IMMUTABLE in response.headers["cache-control"]
    assert response.headers["etag"]


def test_if_none_match_gets_a_304(client, run_id):
    etag = client.get(f"/api/v1/runs/{run_id}").headers["etag"]
    response = client.get(f"/api/v1/runs/{run_id}", headers={"If-None-Match": etag})
    assert response.status_code == 304
    assert response.content == b""


def test_a_stale_etag_gets_the_body(client, run_id):
    response = client.get(f"/api/v1/runs/{run_id}", headers={"If-None-Match": '"something-else"'})
    assert response.status_code == 200


def test_the_config_is_served_with_a_diff_against_the_defaults(client, run_id):
    response = client.get(f"/api/v1/runs/{run_id}/config")
    assert response.status_code == 200
    body = response.json()
    assert "triple_barrier" in body["config_yaml"]
    assert body["config_hash"]
    # The smoke config sets its own name and seed, so those must show as
    # differing from the packaged defaults.
    assert "meta.name" in body["diff_vs_defaults"]


def test_the_folds_carry_their_spans_and_purge_counts(client, run_id):
    response = client.get(f"/api/v1/runs/{run_id}/folds")
    assert response.status_code == 200
    body = response.json()
    assert body["scheme"] == "walkforward"
    assert body["folds"]
    first = body["folds"][0]
    assert "train_span" in first
    assert "purged" in first
    assert "embargoed" in first


def test_the_leaderboard_includes_the_shuffled_baseline(client, run_id):
    body = client.get(f"/api/v1/runs/{run_id}/leaderboard").json()
    assert body["selected_model"]
    assert any(row["model"] == "baseline_shuffled" for row in body["rows"])
    assert body["rows"][0]["rank"] == 1


@pytest.mark.parametrize("name", ["equity", "drawdown", "rolling_sharpe", "trades"])
def test_every_documented_series_is_servable(client, run_id, name):
    response = client.get(f"/api/v1/runs/{run_id}/series/{name}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["name"] == name
    assert body["n_source_points"] > 0
    assert len(body["index"]) == body["n_points"]


def test_a_series_is_downsampled_to_the_requested_width(client, run_id):
    body = client.get(f"/api/v1/runs/{run_id}/series/equity", params={"width": 50}).json()
    assert body["n_points"] <= 50
    assert body["downsampled"] is True
    assert body["n_source_points"] > 50


def test_an_unknown_series_names_the_ones_that_exist(client, run_id):
    response = client.get(f"/api/v1/runs/{run_id}/series/nonsense")
    assert response.status_code == 404
    assert "equity" in response.json()["detail"]


def test_a_series_width_outside_the_bounds_is_rejected(client, run_id):
    assert client.get(f"/api/v1/runs/{run_id}/series/equity?width=1").status_code == 422
    assert client.get(f"/api/v1/runs/{run_id}/series/equity?width=999999").status_code == 422


def test_trades_are_paginated(client, run_id):
    first = client.get(f"/api/v1/runs/{run_id}/trades", params={"limit": 5}).json()
    assert len(first["rows"]) <= 5
    assert first["total"] > 0
    if first["next_cursor"]:
        second = client.get(
            f"/api/v1/runs/{run_id}/trades",
            params={"limit": 5, "cursor": first["next_cursor"]},
        ).json()
        assert second["rows"] and second["rows"] != first["rows"]


def test_predictions_can_be_restricted_to_one_fold(client, run_id):
    everything = client.get(f"/api/v1/runs/{run_id}/predictions", params={"limit": 1}).json()
    assert everything["total"] > 0

    one_fold = client.get(
        f"/api/v1/runs/{run_id}/predictions", params={"fold": 0, "limit": 1000}
    ).json()
    assert 0 < one_fold["total"] <= everything["total"]
    assert all(row["fold"] == 0 for row in one_fold["rows"])


def test_a_run_without_shap_says_why_rather_than_404ing_blankly(client, run_id):
    # The smoke config selects a baseline, and §9 only explains tree models.
    response = client.get(f"/api/v1/runs/{run_id}/shap/summary")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "tree model" in detail
    assert response.json()["selected_model"]


def test_an_artifact_can_be_downloaded(client, run_id):
    response = client.get(f"/api/v1/runs/{run_id}/export/tearsheet.json")
    assert response.status_code == 200
    assert run_id in response.headers["content-disposition"]


def test_an_artifact_name_cannot_escape_the_run_directory(client, run_id):
    response = client.get(f"/api/v1/runs/{run_id}/export/../../secrets.json")
    assert response.status_code in {400, 404}


def test_a_run_id_cannot_escape_the_artifact_store(client, api_workspace):
    # Percent-encoded, so the segment is still one path component when the
    # router matches it and only becomes `..` afterwards. The un-encoded form
    # is normalised away by the client and proves nothing.
    #
    # This served the repository's own README until `ArtifactStore.open_run`
    # started checking that a run id names a direct child of the store.
    root, _ = api_workspace
    (root / "outside.md").write_text("# not an artifact\n", encoding="utf-8")

    response = client.get("/api/v1/runs/%2E%2E/export/outside.md")
    assert response.status_code == 404
    assert "not an artifact" not in response.text


def test_an_unexportable_suffix_is_refused(client, run_id):
    assert client.get(f"/api/v1/runs/{run_id}/export/logs.jsonl").status_code == 404


def test_the_research_note_is_served_as_markdown(client, run_id):
    body = client.get(f"/api/v1/runs/{run_id}/research").json()
    assert body["markdown"].startswith("# Research note")
    assert "Not investment advice" in body["markdown"]


def test_comparing_a_run_with_itself_reports_identical_provenance(client, run_id):
    response = client.get("/api/v1/runs/compare", params={"a": run_id, "b": run_id})
    assert response.status_code == 200
    body = response.json()
    assert body["config_identical"] is True
    assert body["data_identical"] is True
    assert body["metric_deltas"]["oos_net_sharpe"]["delta"] == 0.0


def test_comparing_against_a_missing_run_is_a_404(client, run_id):
    response = client.get("/api/v1/runs/compare", params={"a": run_id, "b": "ghost"})
    assert response.status_code == 404
    assert "ghost" in response.json()["detail"]
