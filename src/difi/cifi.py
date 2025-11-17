from pathlib import Path
from typing import Optional, Tuple, TypeVar, Union

import logging
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from .metrics import (
    FindabilityMetric,
    FindableObservations,
    SingletonMetric,
    TrackletMetric,
)
from .observations import Observations
from .partitions import Partitions, PartitionSummary

__all__ = ["analyze_observations", "AllObjects"]


logger = logging.getLogger(__name__)


Metrics = TypeVar("Metrics", bound=FindabilityMetric)


class AllObjects(qv.Table):
    object_id = qv.LargeStringColumn()
    partition_id = qv.LargeStringColumn()
    mjd_min = qv.Float64Column()
    mjd_max = qv.Float64Column()
    arc_length = qv.Float64Column()
    num_obs = qv.Int64Column()
    num_observatories = qv.Int64Column()
    findable = qv.BooleanColumn(nullable=True)
    found_pure = qv.Int64Column(nullable=True)
    found_contaminated = qv.Int64Column(nullable=True)
    pure = qv.Int64Column(nullable=True)
    pure_complete = qv.Int64Column(nullable=True)
    contaminated = qv.Int64Column(nullable=True)
    contaminant = qv.Int64Column(nullable=True)
    mixed = qv.Int64Column(nullable=True)
    obs_in_pure = qv.Int64Column(nullable=True)
    obs_in_pure_complete = qv.Int64Column(nullable=True)
    obs_in_contaminated = qv.Int64Column(nullable=True)
    obs_as_contaminant = qv.Int64Column(nullable=True)
    obs_in_mixed = qv.Int64Column(nullable=True)

    @classmethod
    def create(
        cls,
        observations: Observations,
        findable: FindableObservations,
        partition_summary: PartitionSummary,
    ) -> "AllObjects":
        """
        Create a summary of all objects in the observations subdivided by partitions.
        For each unique object in a partition, the number of observations, number of observatories,
        and the arc length of the observations are calculated.


        Parameters
        ----------
        observations : Observations
            Table of observations.
        findable : FindableObservations
            Table of findable observations per partition.
        partition_summary : PartitionSummary
            Table of partition summaries that defines the start and end nights of the partitions.

        Returns
        -------
        all_objects : AllObjects
            Summary of all objects in the observations.
        """
        all_objects = cls.empty()

        for partition in partition_summary:
            partition_id = partition.id[0].as_py()

            # Get the findable objects for this window
            findable_i = findable.select("partition_id", partition_id)

            observations_in_window = observations.filter_partition(partition)
            observations_in_window_table = observations_in_window.flattened_table().append_column(
                "mjd_utc", observations_in_window.time.rescale("utc").mjd()
            )

            num_obs_per_object = observations_in_window_table.group_by(
                ["object_id"], use_threads=False
            ).aggregate(
                [
                    ("object_id", "count", pc.CountOptions(mode="all")),
                    ("mjd_utc", "max", pc.CountOptions(mode="all")),
                    ("mjd_utc", "min", pc.CountOptions(mode="all")),
                    ("observatory_code", "count_distinct", pc.CountOptions(mode="all")),
                ]
            )

            all_objects_i = cls.from_kwargs(
                partition_id=pa.repeat(partition_id, len(num_obs_per_object)),
                object_id=num_obs_per_object["object_id"],
                num_obs=num_obs_per_object["object_id_count"],
                num_observatories=num_obs_per_object["observatory_code_count_distinct"],
                mjd_min=num_obs_per_object["mjd_utc_min"],
                mjd_max=num_obs_per_object["mjd_utc_max"],
                arc_length=pc.subtract(num_obs_per_object["mjd_utc_max"], num_obs_per_object["mjd_utc_min"]),
                findable=pc.is_in(num_obs_per_object["object_id"], findable_i.object_id.unique()),
            )

            all_objects = qv.concatenate([all_objects, all_objects_i])

        return all_objects.sort_by([("partition_id", "ascending"), ("object_id", "ascending")])


def analyze_observations(
    observations: Union[Observations, str, Path],
    partitions: Optional[Partitions] = None,
    metric: Union[str, Metrics] = "singletons",
    by_object: bool = False,
    ignore_after_discovery: bool = False,
    max_processes: Optional[int] = 1,
    engine: str = "auto",
    **metric_kwargs,
) -> Tuple[AllObjects, FindableObservations, PartitionSummary]:
    """
    Can I Find It?

    Parameters
    ----------
    observations : Observations or path-like
        Table of observations, or a path to a Parquet file/dataset containing them.
    partitions : Partitions, optional
        Table of partitions defining the start and end nights (both inclusive) of the partitions.
        If None, a single partition is created from the unique nights in the observations.
    metric : Union[str, Metrics], optional
        Metric to use to determine findability. If a string, the metric is looked up in the metric mapper.
        If a FindabilityMetric, the metric is used directly.
    by_object : bool, optional
        Whether to calculate findability by object.
    ignore_after_discovery : bool, optional
        Whether to ignore observation that follow a discovery. Only used if by_object is True.
    max_processes : int, optional
        The maximum number of processes to use for parallelization.
    engine : {"auto", "memory", "duckdb"}, optional
        Execution engine to use. "memory" retains the original in-memory behavior.
        "duckdb" will load observations from disk using DuckDB; "auto" selects
        based on input type.

    Returns
    -------
    all_objects : AllObjects
        Summary of all objects in the observations.
    findable_observations : FindableObservations
        Table of findable observations per partition.
    partition_summary : PartitionSummary
        Summary of the observations within each partition and details
        about the numbers of objects that are findable.
    """

    # Normalize engine
    if engine not in {"auto", "memory", "duckdb"}:
        raise ValueError(f"Unknown engine '{engine}', expected 'auto', 'memory', or 'duckdb'.")

    # Determine if we were given a path or an in-memory table
    obs_is_table = isinstance(observations, Observations)
    obs_is_path = isinstance(observations, (str, Path))

    if engine == "auto":
        engine = "memory" if obs_is_table else "duckdb" if obs_is_path else "memory"

    # Resolve metric instance
    metric_func_mapper = {
        "singletons": SingletonMetric,
        "tracklets": TrackletMetric,
    }
    if isinstance(metric, str):
        if metric not in metric_func_mapper:
            raise ValueError(f"Unknown metric {metric}")
        metric_func = metric_func_mapper[metric]
        metric_ = metric_func(**metric_kwargs)
    elif isinstance(metric, FindabilityMetric):
        metric_ = metric
    else:
        raise ValueError("metric must be a string or a FindabilityMetric")

    # DuckDB engine: support a streaming, on-disk implementation for the
    # SingletonMetric when given a Parquet path and no explicit partitions.
    if engine == "duckdb" and obs_is_path and isinstance(metric_, SingletonMetric) and partitions is None:
        return _analyze_observations_singletons_duckdb(
            str(observations),
            metric_,
            max_processes=max_processes,
        )

    # Fall back to the original in-memory implementation
    if engine == "duckdb" and not obs_is_table:
        # observations is a path-like pointing to Parquet data
        observations = Observations.from_parquet(str(observations))
        obs_is_table = True

    if not obs_is_table:
        raise TypeError(
            "analyze_observations expected an Observations table or a path-like to Parquet data."
        )

    observations_table: Observations = observations

    if len(observations_table) == 0:
        return AllObjects.empty(), FindableObservations.empty(), PartitionSummary.empty()

    if partitions is None:
        partitions = Partitions.create_single(observations_table.night)

    # Create the partition summary table
    partition_summary = PartitionSummary.create(observations_table, partitions)

    if not pc.all(pc.is_null(observations_table.object_id)).as_py():

        findable_observations = metric_.run(
            observations_table,
            partitions,
            by_object=by_object,
            ignore_after_discovery=ignore_after_discovery,
            max_processes=max_processes,
        )
        partition_summary = partition_summary.update_findable(findable_observations)

        # Create the AllObjects table
        all_objects = AllObjects.create(observations_table, findable_observations, partition_summary)

        return all_objects, findable_observations, partition_summary

    else:
        return AllObjects.empty(), FindableObservations.empty(), partition_summary


def _analyze_observations_singletons_duckdb(
    observations_path: str,
    metric: SingletonMetric,
    max_processes: Optional[int] = None,
) -> Tuple[AllObjects, FindableObservations, PartitionSummary]:
    """
    DuckDB-backed implementation of analyze_observations for the SingletonMetric.

    This implementation operates directly on Parquet observations and computes
    AllObjects and PartitionSummary without loading the full table into memory.
    """

    try:
        import duckdb  # type: ignore[import]
    except ImportError as e:  # pragma: no cover - exercised via tests
        raise ImportError(
            "DuckDB engine requested, but 'duckdb' is not installed. "
            "Install the optional extra with 'pip install difi[duckdb]'."
        ) from e

    # Resolve to an absolute path so DuckDB can always find the Parquet file
    obs_path = str(Path(observations_path).resolve())
    logger.info("Singletons DuckDB: will read observations from %s", obs_path)
    if not Path(obs_path).exists():
        logger.error("Singletons DuckDB: resolved observations path does not exist: %s", obs_path)
        raise FileNotFoundError(f"Observations parquet not found at '{obs_path}'")

    logger.info("Singletons DuckDB: current working directory: %s", Path.cwd())
    conn = duckdb.connect()

    # Configure threads if requested
    if max_processes is not None:
        threads = max(1, max_processes)
        conn.execute("PRAGMA threads = ?", [threads])
        logger.info("Singletons DuckDB: configured PRAGMA threads=%d", threads)
    else:
        threads = conn.execute("PRAGMA threads").fetchone()[0]
        logger.info("Singletons DuckDB: using default PRAGMA threads=%s", threads)

    # Register observations as a DuckDB view over Parquet
    logger.info("Singletons DuckDB: creating view over Parquet")
    conn.execute(f"CREATE VIEW obs AS SELECT * FROM read_parquet('{obs_path}')")

    # Basic stats and single partition spanning all nights with observations
    logger.info("Singletons DuckDB: computing global night range and total obs")
    min_night, max_night, total_obs = conn.execute(
        "SELECT MIN(night), MAX(night), COUNT(*) FROM obs"
    ).fetchone()
    logger.info(
        "Singletons DuckDB: night range=[%s, %s], total_obs=%s",
        min_night,
        max_night,
        total_obs,
    )

    if total_obs == 0:
        logger.info("Singletons DuckDB: no observations, returning empty results")
        return AllObjects.empty(), FindableObservations.empty(), PartitionSummary.empty()

    # Per-object stats and night counts
    min_obs = metric.min_obs
    min_nights = metric.min_nights
    min_nightly = metric.min_nightly_obs_in_min_nights

    logger.info("Singletons DuckDB: computing per-object statistics")
    all_objects_arrow = conn.execute(
        """
        WITH per_obj AS (
            SELECT
                object_id,
                COUNT(*) AS num_obs,
                COUNT(DISTINCT observatory_code) AS num_observatories,
                MIN(night) AS night_min,
                MAX(night) AS night_max
            FROM obs
            WHERE object_id IS NOT NULL
            GROUP BY object_id
        ),
        per_night AS (
            SELECT
                object_id,
                night,
                COUNT(*) AS cnt
            FROM obs
            WHERE object_id IS NOT NULL
            GROUP BY object_id, night
        ),
        night_stats AS (
            SELECT
                object_id,
                COUNT(*) AS total_nights,
                SUM(CASE WHEN cnt >= ? THEN 1 ELSE 0 END) AS nights_ge_threshold
            FROM per_night
            GROUP BY object_id
        )
        SELECT
            po.object_id,
            po.num_obs,
            po.num_observatories,
            po.night_min,
            po.night_max,
            ns.total_nights,
            ns.nights_ge_threshold,
            CASE
                WHEN ns.total_nights < ? THEN FALSE
                WHEN ns.total_nights = ?
                     AND ns.nights_ge_threshold = ?
                     AND po.num_obs >= ? THEN TRUE
                WHEN ns.total_nights > ?
                     AND po.num_obs >= ? THEN TRUE
                ELSE FALSE
            END AS findable
        FROM per_obj AS po
        JOIN night_stats AS ns USING (object_id)
        ORDER BY po.object_id
        """,
        [
            min_nightly,
            min_nights,
            min_nights,
            min_nights,
            min_obs,
            min_nights,
            min_obs,
        ],
    ).arrow().read_all()
    logger.info("Singletons DuckDB: per-object stats rows=%d", len(all_objects_arrow))

    if len(all_objects_arrow) == 0:
        # No objects with non-null IDs
        logger.info("Singletons DuckDB: no objects with non-null IDs, building empty AllObjects")
        partition_summary = PartitionSummary.from_kwargs(
            id=["0"],
            start_night=[min_night],
            end_night=[max_night],
            observations=[total_obs],
            findable=[0],
            found=[None],
            completeness=[None],
            pure_known=[None],
            pure_unknown=[None],
            contaminated=[None],
            mixed=[None],
        )
        return AllObjects.empty(), FindableObservations.empty(), partition_summary

    mjd_min = all_objects_arrow["night_min"].cast(pa.float64())
    mjd_max = all_objects_arrow["night_max"].cast(pa.float64())
    arc_length = pc.subtract(mjd_max, mjd_min)

    logger.info("Singletons DuckDB: building AllObjects table")
    all_objects = AllObjects.from_kwargs(
        object_id=all_objects_arrow["object_id"],
        partition_id=pa.repeat("0", len(all_objects_arrow)),
        mjd_min=mjd_min,
        mjd_max=mjd_max,
        arc_length=arc_length,
        num_obs=all_objects_arrow["num_obs"],
        num_observatories=all_objects_arrow["num_observatories"],
        findable=all_objects_arrow["findable"],
        found_pure=pa.array([0] * len(all_objects_arrow), type=pa.int64()),
        found_contaminated=pa.array([0] * len(all_objects_arrow), type=pa.int64()),
        pure=pa.array([0] * len(all_objects_arrow), type=pa.int64()),
        pure_complete=pa.array([0] * len(all_objects_arrow), type=pa.int64()),
        contaminated=pa.array([0] * len(all_objects_arrow), type=pa.int64()),
        contaminant=pa.array([0] * len(all_objects_arrow), type=pa.int64()),
        mixed=pa.array([0] * len(all_objects_arrow), type=pa.int64()),
        obs_in_pure=pa.array([0] * len(all_objects_arrow), type=pa.int64()),
        obs_in_pure_complete=pa.array([0] * len(all_objects_arrow), type=pa.int64()),
        obs_in_contaminated=pa.array([0] * len(all_objects_arrow), type=pa.int64()),
        obs_as_contaminant=pa.array([0] * len(all_objects_arrow), type=pa.int64()),
        obs_in_mixed=pa.array([0] * len(all_objects_arrow), type=pa.int64()),
    )

    # Partition summary: single partition "0"
    findable_count = int(
        pc.sum(all_objects.findable.cast(pa.int64())).as_py()
    )

    logger.info(
        "Singletons DuckDB: building PartitionSummary, findable_count=%d", findable_count
    )
    partition_summary = PartitionSummary.from_kwargs(
        id=["0"],
        start_night=[min_night],
        end_night=[max_night],
        observations=[total_obs],
        findable=[findable_count],
        found=[None],
        completeness=[None],
        pure_known=[None],
        pure_unknown=[None],
        contaminated=[None],
        mixed=[None],
    )

    # We do not currently materialize the full FindableObservations table in the
    # DuckDB engine; for difi usage the AllObjects.findable flag and
    # PartitionSummary.findable count are sufficient.
    return all_objects, FindableObservations.empty(), partition_summary
