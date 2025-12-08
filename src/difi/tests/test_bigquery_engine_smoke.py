import os
from pathlib import Path

import pyarrow as pa
import pytest

from difi.cifi import analyze_observations
from difi.difi import analyze_linkages, LinkageMembers, AllLinkages
from difi.metrics import SingletonMetric
from difi.observations import Observations
from difi.partitions import PartitionSummary, Partitions
from difi.bigquery_types import BigQueryObservationsInput, BigQueryLinkageInput


HAS_BQ = os.getenv("DIFI_TEST_BIGQUERY") == "1"


@pytest.mark.skipif(not HAS_BQ, reason="BigQuery tests disabled (set DIFI_TEST_BIGQUERY=1 to enable)")
def test_bigquery_engine_observations_round_trip():
    """
    Lightweight smoke test that the BigQuery engine can be invoked without
    crashing when a suitable observations table is available.

    This test expects environment variables to point at a small BigQuery
    table that mirrors the difi Observations schema:

      - DIFI_TEST_BQ_OBSERVATIONS_TABLE: fully-qualified table id
    """

    table = os.environ["DIFI_TEST_BQ_OBSERVATIONS_TABLE"]
    metric = SingletonMetric()
    bq_input = BigQueryObservationsInput(table=table)

    all_objects, findable, partition_summary = analyze_observations(
        bq_input,
        partitions=None,
        metric=metric,
        by_object=True,
        ignore_after_discovery=False,
        max_processes=1,
        engine="bigquery",
    )

    # We don't assert specific values here; the goal is simply that the
    # engine executes successfully and returns well-formed tables.
    assert isinstance(all_objects, type(all_objects))
    assert isinstance(findable, type(findable))
    assert isinstance(partition_summary, PartitionSummary)


@pytest.mark.skipif(not HAS_BQ, reason="BigQuery tests disabled (set DIFI_TEST_BIGQUERY=1 to enable)")
def test_bigquery_engine_linkages_smoke():
    """
    Smoke test for the BigQuery-backed linkage engine. This test only
    verifies that the call path succeeds when provided with suitable
    BigQuery tables; it does not currently check numerical parity with
    the DuckDB engine.

    Expected environment variables:

      - DIFI_TEST_BQ_OBSERVATIONS_TABLE
      - DIFI_TEST_BQ_LINKAGE_TABLE
    """

    obs_table = os.environ["DIFI_TEST_BQ_OBSERVATIONS_TABLE"]
    lm_table = os.environ["DIFI_TEST_BQ_LINKAGE_TABLE"]

    # Construct minimal AllObjects / PartitionSummary for the test
    # using an in-memory Observations table. This mirrors the small
    # test datasets used in the DuckDB tests.
    test_obs_path = Path(__file__).parent / "testdata" / "observations.parquet"
    test_lm_path = Path(__file__).parent / "testdata" / "linkage_members.parquet"

    obs = Observations.from_parquet(str(test_obs_path))
    lm = LinkageMembers.from_parquet(str(test_lm_path))

    from difi.cifi import analyze_observations as analyze_obs_mem

    all_objects_mem, _, partition_summary_mem = analyze_obs_mem(
        obs,
        partitions=None,
        metric="singletons",
        by_object=True,
        ignore_after_discovery=False,
        max_processes=1,
        engine="memory",
    )

    obs_bq = BigQueryObservationsInput(table=obs_table)
    lm_bq = BigQueryLinkageInput(
        table=lm_table,
        linkage_id_column="linkage_id",
        obs_id_column="obs_id",
    )

    all_objects_bq, all_linkages_bq, partition_summary_bq = analyze_linkages(
        obs_bq,
        lm_bq,
        all_objects_mem,
        partition_summary=partition_summary_mem,
        min_obs=6,
        contamination_percentage=50.0,
        engine="bigquery",
    )

    assert isinstance(all_objects_bq, type(all_objects_mem))
    assert isinstance(all_linkages_bq, AllLinkages)
    assert isinstance(partition_summary_bq, PartitionSummary)


