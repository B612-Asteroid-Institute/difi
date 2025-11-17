import os
from pathlib import Path

import pytest

from ..cifi import analyze_observations
from ..difi import DuckDBLinkageInput, LinkageMembers, analyze_linkages
from ..observations import Observations

TESTDATA_DIR = Path(__file__).parent / "testdata"
OBSERVATIONS_PATH = TESTDATA_DIR / "observations.parquet"
LINKAGE_MEMBERS_PATH = TESTDATA_DIR / "linkage_members.parquet"

try:
    import duckdb  # type: ignore[import]

    HAS_DUCKDB = True
except ImportError:  # pragma: no cover - exercised via environment
    HAS_DUCKDB = False


@pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")
def test_analyze_observations_duckdb_matches_memory(test_observations, tmp_path):
    # In-memory engine
    all_objects_mem, findable_mem, summary_mem = analyze_observations(
        test_observations,
        partitions=None,
        metric="singletons",
        by_object=True,
        ignore_after_discovery=False,
        max_processes=1,
        engine="memory",
    )

    # Write observations to Parquet and run via duckdb engine using path input
    obs_path = tmp_path / "observations.parquet"
    test_observations.to_parquet(obs_path)

    all_objects_ddb, findable_ddb, summary_ddb = analyze_observations(
        str(obs_path),
        partitions=None,
        metric="singletons",
        by_object=True,
        ignore_after_discovery=False,
        max_processes=1,
        engine="duckdb",
    )

    assert len(all_objects_mem) == len(all_objects_ddb)
    assert all_objects_mem.object_id.to_pylist() == all_objects_ddb.object_id.to_pylist()
    assert all_objects_mem.partition_id.to_pylist() == all_objects_ddb.partition_id.to_pylist()
    assert summary_mem.observations[0].as_py() == summary_ddb.observations[0].as_py()


@pytest.mark.skipif(not HAS_DUCKDB or os.getenv("DIFI_SKIP_DUCKDB_TESTS") == "1", reason="DuckDB tests skipped by environment")
def test_analyze_linkages_duckdb_matches_memory(tmp_path):
    # Load small on-disk test data provided with the package
    obs = Observations.from_parquet(str(OBSERVATIONS_PATH))
    lm = LinkageMembers.from_parquet(str(LINKAGE_MEMBERS_PATH))

    # Memory engine baseline
    all_objects_mem, _, partition_summary_mem = analyze_observations(
        obs,
        partitions=None,
        metric="singletons",
        by_object=True,
        ignore_after_discovery=False,
        max_processes=1,
        engine="memory",
    )

    all_objects_mem2, all_linkages_mem, summary_mem = analyze_linkages(
        obs,
        lm,
        all_objects_mem,
        partition_summary=partition_summary_mem,
        min_obs=6,
        contamination_percentage=50.0,
        engine="memory",
    )

    # DuckDB engine using the same underlying Parquet files (path inputs)
    all_objects_ddb, all_linkages_ddb, summary_ddb = analyze_linkages(
        str(OBSERVATIONS_PATH),
        str(LINKAGE_MEMBERS_PATH),
        all_objects_mem,
        partition_summary=partition_summary_mem,
        min_obs=6,
        contamination_percentage=50.0,
        engine="duckdb",
    )

    assert len(all_linkages_mem) == len(all_linkages_ddb)
    assert set(all_linkages_mem.linkage_id.to_pylist()) == set(all_linkages_ddb.linkage_id.to_pylist())

    # Compare aggregate counts in AllObjects
    assert len(all_objects_mem2) == len(all_objects_ddb)
    assert set(all_objects_mem2.object_id.to_pylist()) == set(all_objects_ddb.object_id.to_pylist())

    # Partition summaries should agree on basic statistics
    assert len(summary_mem) == len(summary_ddb)
    assert summary_mem.observations[0].as_py() == summary_ddb.observations[0].as_py()


@pytest.mark.skipif(not HAS_DUCKDB or os.getenv("DIFI_SKIP_DUCKDB_TESTS") == "1", reason="DuckDB tests skipped by environment")
def test_analyze_linkages_duckdb_with_linkage_input(tmp_path):
    """
    Verify that DuckDBLinkageInput can be used to read linkage members from
    one or more Parquet files and produces the same result as the in-memory
    LinkageMembers table.
    """

    obs = Observations.from_parquet(str(OBSERVATIONS_PATH))
    lm = LinkageMembers.from_parquet(str(LINKAGE_MEMBERS_PATH))

    # Baseline in-memory pipeline
    all_objects_mem, _, partition_summary_mem = analyze_observations(
        obs,
        partitions=None,
        metric="singletons",
        by_object=True,
        ignore_after_discovery=False,
        max_processes=1,
        engine="memory",
    )

    all_objects_mem2, all_linkages_mem, summary_mem = analyze_linkages(
        obs,
        lm,
        all_objects_mem,
        partition_summary=partition_summary_mem,
        min_obs=6,
        contamination_percentage=50.0,
        engine="memory",
    )

    # Use DuckDBLinkageInput pointing at the same Parquet file and explicitly
    # specifying the logical linkage/obs_id columns.
    linkage_input = DuckDBLinkageInput(
        path=str(LINKAGE_MEMBERS_PATH),
        linkage_id_column="linkage_id",
        obs_id_column="obs_id",
    )

    all_objects_ddb, all_linkages_ddb, summary_ddb = analyze_linkages(
        str(OBSERVATIONS_PATH),
        linkage_input,
        all_objects_mem,
        partition_summary=partition_summary_mem,
        min_obs=6,
        contamination_percentage=50.0,
        engine="duckdb",
    )

    assert len(all_linkages_mem) == len(all_linkages_ddb)
    assert set(all_linkages_mem.linkage_id.to_pylist()) == set(all_linkages_ddb.linkage_id.to_pylist())
    assert len(all_objects_mem2) == len(all_objects_ddb)
    assert set(all_objects_mem2.object_id.to_pylist()) == set(all_objects_ddb.object_id.to_pylist())
    assert len(summary_mem) == len(summary_ddb)
    assert summary_mem.observations[0].as_py() == summary_ddb.observations[0].as_py()


@pytest.mark.skipif(not HAS_DUCKDB or os.getenv("DIFI_SKIP_DUCKDB_TESTS") == "1", reason="DuckDB tests skipped by environment")
def test_analyze_linkages_duckdb_with_linkage_input_glob(tmp_path):
    """
    Verify that DuckDBLinkageInput works when the path is a glob pattern
    matching multiple Parquet files. We split the linkage_members parquet
    into two shards and ensure the DuckDB result matches the baseline.
    """

    # Baseline data
    obs = Observations.from_parquet(str(OBSERVATIONS_PATH))
    lm_full = LinkageMembers.from_parquet(str(LINKAGE_MEMBERS_PATH))

    all_objects_mem, _, partition_summary_mem = analyze_observations(
        obs,
        partitions=None,
        metric="singletons",
        by_object=True,
        ignore_after_discovery=False,
        max_processes=1,
        engine="memory",
    )

    all_objects_mem2, all_linkages_mem, summary_mem = analyze_linkages(
        obs,
        lm_full,
        all_objects_mem,
        partition_summary=partition_summary_mem,
        min_obs=6,
        contamination_percentage=50.0,
        engine="memory",
    )

    # Split linkage_members into two Parquet shards
    mid = len(lm_full) // 2
    lm_shard1 = lm_full[:mid]
    lm_shard2 = lm_full[mid:]

    shard1_path = tmp_path / "lm_shard1.parquet"
    shard2_path = tmp_path / "lm_shard2.parquet"
    lm_shard1.to_parquet(shard1_path)
    lm_shard2.to_parquet(shard2_path)

    # Use a glob pattern over both shards
    glob_path = str(tmp_path / "lm_shard*.parquet")
    linkage_input = DuckDBLinkageInput(
        path=glob_path,
        linkage_id_column="linkage_id",
        obs_id_column="obs_id",
    )

    all_objects_ddb, all_linkages_ddb, summary_ddb = analyze_linkages(
        str(OBSERVATIONS_PATH),
        linkage_input,
        all_objects_mem,
        partition_summary=partition_summary_mem,
        min_obs=6,
        contamination_percentage=50.0,
        engine="duckdb",
    )

    assert len(all_linkages_mem) == len(all_linkages_ddb)
    assert set(all_linkages_mem.linkage_id.to_pylist()) == set(all_linkages_ddb.linkage_id.to_pylist())
    assert len(all_objects_mem2) == len(all_objects_ddb)
    assert set(all_objects_mem2.object_id.to_pylist()) == set(all_objects_ddb.object_id.to_pylist())
    assert len(summary_mem) == len(summary_ddb)
    assert summary_mem.observations[0].as_py() == summary_ddb.observations[0].as_py()


@pytest.mark.skipif(not HAS_DUCKDB or os.getenv("DIFI_SKIP_DUCKDB_TESTS") == "1", reason="DuckDB tests skipped by environment")
def test_analyze_linkages_duckdb_with_linkage_input_directory(tmp_path):
    """
    Verify that DuckDBLinkageInput works when the path is a directory
    containing multiple Parquet files.
    """

    obs = Observations.from_parquet(str(OBSERVATIONS_PATH))
    lm_full = LinkageMembers.from_parquet(str(LINKAGE_MEMBERS_PATH))

    all_objects_mem, _, partition_summary_mem = analyze_observations(
        obs,
        partitions=None,
        metric="singletons",
        by_object=True,
        ignore_after_discovery=False,
        max_processes=1,
        engine="memory",
    )

    all_objects_mem2, all_linkages_mem, summary_mem = analyze_linkages(
        obs,
        lm_full,
        all_objects_mem,
        partition_summary=partition_summary_mem,
        min_obs=6,
        contamination_percentage=50.0,
        engine="memory",
    )

    # Write two shards into the tmp directory
    mid = len(lm_full) // 2
    lm_shard1 = lm_full[:mid]
    lm_shard2 = lm_full[mid:]

    shard1_path = tmp_path / "lm_shard1.parquet"
    shard2_path = tmp_path / "lm_shard2.parquet"
    lm_shard1.to_parquet(shard1_path)
    lm_shard2.to_parquet(shard2_path)

    # Point DuckDBLinkageInput at the directory; DuckDB's read_parquet can
    # treat a directory of Parquet files as a dataset.
    linkage_input = DuckDBLinkageInput(
        path=str(tmp_path),
        linkage_id_column="linkage_id",
        obs_id_column="obs_id",
    )

    all_objects_ddb, all_linkages_ddb, summary_ddb = analyze_linkages(
        str(OBSERVATIONS_PATH),
        linkage_input,
        all_objects_mem,
        partition_summary=partition_summary_mem,
        min_obs=6,
        contamination_percentage=50.0,
        engine="duckdb",
    )

    assert len(all_linkages_mem) == len(all_linkages_ddb)
    assert set(all_linkages_mem.linkage_id.to_pylist()) == set(all_linkages_ddb.linkage_id.to_pylist())
    assert len(all_objects_mem2) == len(all_objects_ddb)
    assert set(all_objects_mem2.object_id.to_pylist()) == set(all_objects_ddb.object_id.to_pylist())
    assert len(summary_mem) == len(summary_ddb)
    assert summary_mem.observations[0].as_py() == summary_ddb.observations[0].as_py()


