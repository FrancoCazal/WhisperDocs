"""The dashboard, alert rules and compose profile in ops/ (audit E13) match
what the API exports: every query names a metric and labels the API has, and
the files point at each other.

The api container mounts app/ and tests/ but not ops/, so there these skip;
CI and a local run read them from the checkout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from prometheus_client import Counter, Histogram

from app import metrics

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
pytestmark = pytest.mark.skipif(not OPS.is_dir(), reason="ops/ is not mounted here")

_SUFFIXES = {Counter: ("_total",), Histogram: ("_bucket", "_count", "_sum")}
_NAME = re.compile(r"\b(whisperdocs_[a-z_]+|up)\b")
_SELECTOR = re.compile(r"\b(whisperdocs_[a-z_]+|up)\{([^}]*)\}")
_MATCHER = re.compile(r"([a-z_]+)\s*(?:=~|!~|!=|=)")
_BY = re.compile(r"\bby\s*\(([^)]*)\)")


def _exported() -> dict[str, set[str]]:
    """Each exported series name with the labels it carries."""
    names: dict[str, set[str]] = {"up": {"job", "instance"}}
    for value in vars(metrics).values():
        for kind, suffixes in _SUFFIXES.items():
            if isinstance(value, kind):
                base = value._name
                labels = set(value._labelnames)
                for suffix in suffixes:
                    names[base + suffix] = labels | ({"le"} if suffix == "_bucket" else set())
    return names


EXPORTED = _exported()


def _check_query(expr: str) -> None:
    used = set(_NAME.findall(expr))
    assert used, f"no metric in {expr!r}"
    unknown = used - EXPORTED.keys()
    assert not unknown, f"{sorted(unknown)} not exported (in {expr!r})"
    labels: set[str] = set()
    for name, matchers in _SELECTOR.findall(expr):
        labels |= set(_MATCHER.findall(matchers))
    for group in _BY.findall(expr):
        labels |= {label.strip() for label in group.split(",") if label.strip()}
    known = set().union(*(EXPORTED[name] for name in used))
    assert labels <= known, f"{sorted(labels - known)} not labels of {sorted(used)}"


def _yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _compose() -> dict:
    return _yaml(ROOT / "docker-compose.yml")


def test_the_dashboard_queries_what_the_api_exports():
    dashboard = json.loads((OPS / "grafana/dashboards/whisperdocs.json").read_text("utf-8"))
    datasource = _yaml(OPS / "grafana/provisioning/datasources/prometheus.yml")["datasources"][0]

    panels = dashboard["panels"]
    assert dashboard["uid"] == "whisperdocs"
    assert len({p["id"] for p in panels}) == len(panels)
    queries = 0
    for panel in panels:
        grid = panel["gridPos"]
        assert grid["x"] + grid["w"] <= 24, panel["title"]
        if panel["type"] == "row":
            continue
        assert panel["datasource"]["uid"] == datasource["uid"], panel["title"]
        assert panel["targets"], panel["title"]
        for target in panel["targets"]:
            _check_query(target["expr"])
            queries += 1
    assert queries >= 10


def test_each_alert_rule_queries_what_the_api_exports():
    groups = _yaml(OPS / "prometheus/alerts.yml")["groups"]
    rules = [rule for group in groups for rule in group["rules"]]

    assert {r["alert"] for r in rules} == {
        "WhisperDocsDown",
        "WhisperDocsHTTPErrors",
        "WhisperDocsModelFailing",
        "WhisperDocsSlowTurns",
    }
    for rule in rules:
        _check_query(rule["expr"])
        assert re.fullmatch(r"\d+m", rule["for"]), rule["alert"]
        assert rule["labels"]["severity"] in {"warning", "critical"}
        assert rule["annotations"]["summary"]


def test_prometheus_scrapes_the_metrics_route_and_loads_the_rules():
    from app.main import API_PREFIX, app

    config = _yaml(OPS / "prometheus/prometheus.yml")
    (job,) = config["scrape_configs"]
    rules = _yaml(OPS / "prometheus/alerts.yml")["groups"][0]["rules"]
    down = next(r for r in rules if r["alert"] == "WhisperDocsDown")

    assert job["metrics_path"] == f"{API_PREFIX}/metrics"
    assert job["metrics_path"] in app.openapi()["paths"]
    assert job["static_configs"][0]["targets"] == ["api:8000"]
    assert f'job="{job["job_name"]}"' in down["expr"]
    # The compose file mounts ops/prometheus at /etc/prometheus.
    for path in config["rule_files"]:
        assert (OPS / "prometheus" / Path(path).relative_to("/etc/prometheus")).is_file()


def test_the_metrics_profile_mounts_what_grafana_and_prometheus_read():
    services = _compose()["services"]
    prometheus, grafana = services["prometheus"], services["grafana"]
    provider = _yaml(OPS / "grafana/provisioning/dashboards/whisperdocs.yml")["providers"][0]
    datasource = _yaml(OPS / "grafana/provisioning/datasources/prometheus.yml")["datasources"][0]

    assert prometheus["profiles"] == grafana["profiles"] == ["metrics"]
    mounts = {
        target: ROOT / source
        for volume in prometheus["volumes"] + grafana["volumes"]
        if volume.startswith("./")
        for source, target in [volume.split(":")[:2]]
    }
    assert mounts["/etc/prometheus"] == ROOT / "./ops/prometheus"
    assert mounts["/etc/grafana/provisioning"] == ROOT / "./ops/grafana/provisioning"
    dashboards = mounts[provider["options"]["path"]]
    assert (dashboards / "whisperdocs.json").is_file()
    home = grafana["environment"]["GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH"]
    assert home == f"{provider['options']['path']}/whisperdocs.json"
    assert datasource["url"] == "http://prometheus:9090"
    # Published on the host's loopback only, like Langfuse.
    for service in (prometheus, grafana):
        assert all(port.startswith("127.0.0.1:") for port in service["ports"])
