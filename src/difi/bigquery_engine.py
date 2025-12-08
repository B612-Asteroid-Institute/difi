from __future__ import annotations

from typing import Optional, Tuple

import pyarrow as pa
import pyarrow.compute as pc

from .cifi import AllObjects
from .metrics import SingletonMetric, FindableObservations
from .partitions import PartitionSummary
from .bigquery_types import BigQueryObservationsInput, BigQueryLinkageInput


def _get_bq_clients():
    """
    Lazily import and construct BigQuery and BigQuery Storage clients.

    This helper keeps the core difi imports lightweight and only pulls
    in the Google client libraries when the BigQuery engine is used.
    """

    try:  # pragma: no cover - import exercised in integration tests
        from google.cloud import bigquery
        from google.cloud import bigquery_storage_v1
    except Exception as exc:  # pragma: no cover - exercised via tests without extras
        raise ImportError(
            "BigQuery engine requested, but google-cloud-bigquery and "
            "google-cloud-bigquery-storage are not installed. "
            "Install the optional extra with 'pip install difi[bigquery]'."
        ) from exc

    bq_client = bigquery.Client()
    bqstorage_client = bigquery_storage_v1.BigQueryReadClient()
    return bq_client, bqstorage_client


def _run_query_to_arrow(bq_client, bqstorage_client, query: str, *, params: Optional[dict] = None) -> pa.Table:
    """
    Execute a BigQuery SQL query and return the results as a single Arrow table.
    """

    job_config = None
    if params:
        from google.cloud import bigquery as _bq  # type: ignore[import]

        job_config = _bq.QueryJobConfig(
            query_parameters=[
                _bq.ScalarQueryParameter(name, "INT64" if isinstance(value, int) else "FLOAT64", value)
                for name, value in params.items()
            ]
        )

    job = bq_client.query(query, job_config=job_config)
    result = job.result()
    return result.to_arrow(bqstorage_client=bqstorage_client)


def analyze_observations_singletons_bigquery(
    observations_input: BigQueryObservationsInput,
    metric: SingletonMetric,
    max_processes: Optional[int] = None,
) -> Tuple[AllObjects, FindableObservations, PartitionSummary]:
    """
    BigQuery-backed implementation of the SingletonMetric cifi analysis.

    This mirrors the behaviour of ``_analyze_observations_singletons_duckdb``
    but operates directly on a BigQuery table or view instead of Parquet
    data on disk.
    """

    bq_client, bqstorage_client = _get_bq_clients()
    table = observations_input.table

    # Global statistics over the observations table.
    stats_query = f"""
        SELECT
            MIN(night) AS min_night,
            MAX(night) AS max_night,
            COUNT(*) AS total_obs
        FROM `{table}`
        WHERE object_id IS NOT NULL
    """
    stats_tbl = _run_query_to_arrow(bq_client, bqstorage_client, stats_query)
    if len(stats_tbl) == 0 or stats_tbl["total_obs"][0].as_py() == 0:
        return AllObjects.empty(), FindableObservations.empty(), PartitionSummary.empty()

    min_night = stats_tbl["min_night"][0].as_py()
    max_night = stats_tbl["max_night"][0].as_py()
    total_obs = stats_tbl["total_obs"][0].as_py()

    # Per-object statistics implementing the SingletonMetric logic.
    min_obs = metric.min_obs
    min_nights = metric.min_nights
    min_nightly = metric.min_nightly_obs_in_min_nights

    per_object_query = f"""
        WITH per_obj AS (
            SELECT
                object_id,
                COUNT(*) AS num_obs,
                COUNT(DISTINCT observatory_code) AS num_observatories,
                MIN(night) AS night_min,
                MAX(night) AS night_max
            FROM `{table}`
            WHERE object_id IS NOT NULL
            GROUP BY object_id
        ),
        per_night AS (
            SELECT
                object_id,
                night,
                COUNT(*) AS cnt
            FROM `{table}`
            WHERE object_id IS NOT NULL
            GROUP BY object_id, night
        ),
        night_stats AS (
            SELECT
                object_id,
                COUNT(*) AS total_nights,
                SUM(CASE WHEN cnt >= @min_nightly THEN 1 ELSE 0 END) AS nights_ge_threshold
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
                WHEN ns.total_nights < @min_nights THEN FALSE
                WHEN ns.total_nights = @min_nights
                     AND ns.nights_ge_threshold = @min_nights
                     AND po.num_obs >= @min_obs THEN TRUE
                WHEN ns.total_nights > @min_nights
                     AND po.num_obs >= @min_obs THEN TRUE
                ELSE FALSE
            END AS findable
        FROM per_obj AS po
        JOIN night_stats AS ns USING (object_id)
        ORDER BY po.object_id
    """

    params = {
        "min_nightly": int(min_nightly),
        "min_nights": int(min_nights),
        "min_obs": int(min_obs),
    }
    per_obj_tbl = _run_query_to_arrow(bq_client, bqstorage_client, per_object_query, params=params)
    if len(per_obj_tbl) == 0:
        # No objects with non-null IDs.
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

    night_min = per_obj_tbl["night_min"].cast(pa.float64())
    night_max = per_obj_tbl["night_max"].cast(pa.float64())
    # ChunkedArray subtraction needs to go through pyarrow.compute
    arc_length = pc.subtract(night_max, night_min)

    all_objects = AllObjects.from_kwargs(
        object_id=per_obj_tbl["object_id"],
        partition_id=pa.repeat("0", len(per_obj_tbl)),
        mjd_min=night_min,
        mjd_max=night_max,
        arc_length=arc_length,
        num_obs=per_obj_tbl["num_obs"],
        num_observatories=per_obj_tbl["num_observatories"],
        findable=per_obj_tbl["findable"],
        found_pure=pa.array([0] * len(per_obj_tbl), type=pa.int64()),
        found_contaminated=pa.array([0] * len(per_obj_tbl), type=pa.int64()),
        pure=pa.array([0] * len(per_obj_tbl), type=pa.int64()),
        pure_complete=pa.array([0] * len(per_obj_tbl), type=pa.int64()),
        contaminated=pa.array([0] * len(per_obj_tbl), type=pa.int64()),
        contaminant=pa.array([0] * len(per_obj_tbl), type=pa.int64()),
        mixed=pa.array([0] * len(per_obj_tbl), type=pa.int64()),
        obs_in_pure=pa.array([0] * len(per_obj_tbl), type=pa.int64()),
        obs_in_pure_complete=pa.array([0] * len(per_obj_tbl), type=pa.int64()),
        obs_in_contaminated=pa.array([0] * len(per_obj_tbl), type=pa.int64()),
        obs_as_contaminant=pa.array([0] * len(per_obj_tbl), type=pa.int64()),
        obs_in_mixed=pa.array([0] * len(per_obj_tbl), type=pa.int64()),
    )

    findable_count = int(per_obj_tbl["findable"].cast(pa.int64()).sum().as_py())
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

    return all_objects, FindableObservations.empty(), partition_summary


def analyze_linkages_bigquery(
    observations_input: BigQueryObservationsInput,
    linkage_input: BigQueryLinkageInput,
    all_objects: AllObjects,
    partition_summary: Optional[PartitionSummary],
    *,
    min_obs: int,
    contamination_percentage: float,
) -> Tuple[AllObjects, "AllLinkages", PartitionSummary]:  # type: ignore[name-defined]
    """
    BigQuery-backed implementation of analyze_linkages.

    This mirrors the structure of ``_analyze_linkages_duckdb`` but operates
    directly on BigQuery tables for both linkage classification and the
    per-object difi metrics, returning Arrow-backed Quivr tables.
    """

    from .difi import AllLinkages  # local import to avoid circular imports

    bq_client, bqstorage_client = _get_bq_clients()
    obs_table = observations_input.table
    lm_table = linkage_input.table
    lm_linkage_col = linkage_input.linkage_id_column
    lm_obs_col = linkage_input.obs_id_column

    # Derive partition bounds if not supplied. For now we mirror the
    # single-partition assumption used in the DuckDB engine.
    if partition_summary is None or len(partition_summary) == 0:
        stats_query = f"""
            SELECT
                MIN(night) AS min_night,
                MAX(night) AS max_night
            FROM `{obs_table}`
        """
        stats_tbl = _run_query_to_arrow(bq_client, bqstorage_client, stats_query)
        if len(stats_tbl) == 0:
            # No observations: nothing to classify.
            empty_ps = PartitionSummary.empty()
            return AllObjects.empty(), AllLinkages.empty(), empty_ps
        start_night = int(stats_tbl["min_night"][0].as_py())
        end_night = int(stats_tbl["max_night"][0].as_py())
        partition_id = "0"
        partition_summary = PartitionSummary.from_kwargs(
            id=[partition_id],
            start_night=[start_night],
            end_night=[end_night],
            observations=[0],
        )
    else:
        if len(partition_summary) != 1:
            raise ValueError(
                "analyze_linkages_bigquery currently supports exactly one partition."
            )
        part = partition_summary[0]
        start_night = int(part.start_night[0].as_py())
        end_night = int(part.end_night[0].as_py())
        partition_id = str(part.id[0].as_py())

    # Enforce that all linkage members are present in the observations.
    missing_query = f"""
        WITH lm AS (
            SELECT
                {lm_linkage_col} AS linkage_id,
                {lm_obs_col} AS obs_id
            FROM `{lm_table}`
        )
        SELECT
            COUNT(*) AS missing
        FROM (
            SELECT DISTINCT obs_id FROM lm
            EXCEPT DISTINCT
            SELECT DISTINCT id FROM `{obs_table}`
        )
    """
    missing_tbl = _run_query_to_arrow(bq_client, bqstorage_client, missing_query)
    if len(missing_tbl) and missing_tbl["missing"][0].as_py() > 0:
        raise ValueError("All linkage members must be in the observations.")

    # === Build AllLinkages via BigQuery ===
    all_linkages_query = f"""
        WITH lm AS (
            SELECT
                {lm_linkage_col} AS linkage_id,
                {lm_obs_col} AS obs_id
            FROM `{lm_table}`
        ),
        lma AS (
            SELECT
                lm.linkage_id,
                lm.obs_id,
                obs.object_id,
                obs.night,
                (obs.night < @start_night OR obs.night > @end_night) AS outside_partition
            FROM lm
            JOIN `{obs_table}` AS obs
              ON lm.obs_id = obs.id
        ),
        unique_object_members AS (
            SELECT
                linkage_id,
                COUNT(DISTINCT obs_id) AS num_obs,
                COUNT(DISTINCT object_id) AS num_members,
                SUM(CASE WHEN outside_partition THEN 1 ELSE 0 END) AS num_obs_outside_partition
            FROM lma
            GROUP BY linkage_id
        ),
        linkage_object_counts AS (
            SELECT
                linkage_id,
                object_id,
                COUNT(*) AS object_id_counts
            FROM lma
            GROUP BY linkage_id, object_id
        ),
        linkage_best_object AS (
            SELECT
                linkage_id,
                object_id AS object_id_first,
                object_id_counts AS object_id_counts_first,
                num_obs,
                num_members,
                num_obs_outside_partition,
                SAFE_DIVIDE(CAST(object_id_counts AS FLOAT64),
                            CAST(num_obs AS FLOAT64)) AS percentage_in_linkage_first,
                ROW_NUMBER() OVER (
                    PARTITION BY linkage_id
                    ORDER BY SAFE_DIVIDE(CAST(object_id_counts AS FLOAT64),
                                         CAST(num_obs AS FLOAT64)) DESC
                ) AS rn
            FROM linkage_object_counts
            JOIN unique_object_members USING (linkage_id)
        ),
        all_linkages_core AS (
            SELECT
                linkage_id,
                object_id_first,
                num_obs,
                num_members,
                num_obs_outside_partition,
                object_id_counts_first,
                percentage_in_linkage_first,
                ROUND((1.0 - percentage_in_linkage_first) * 100.0, 10) AS contamination
            FROM linkage_best_object
            WHERE rn = 1
        ),
        all_linkages_flags AS (
            SELECT
                linkage_id,
                object_id_first,
                num_obs,
                num_members,
                num_obs_outside_partition,
                object_id_counts_first,
                percentage_in_linkage_first,
                contamination,
                (contamination = 0.0) AS pure,
                (contamination > 0.0 AND contamination <= @contam) AS contaminated,
                (contamination <> 0.0
                 AND NOT (contamination > 0.0 AND contamination <= @contam)) AS mixed,
                CASE
                    WHEN contamination = 0.0
                         OR (contamination > 0.0 AND contamination <= @contam)
                    THEN object_id_first
                    ELSE NULL
                END AS linked_object_id
            FROM all_linkages_core
        ),
        obs_partition_counts AS (
            SELECT
                object_id,
                COUNT(id) AS num_obs_in_partition
            FROM `{obs_table}`
            WHERE night BETWEEN @start_night AND @end_night
            GROUP BY object_id
        ),
        linkage_obs_partition_counts AS (
            SELECT
                linkage_id,
                COUNT(DISTINCT obs_id) AS num_linkage_obs_inside_partition
            FROM lma
            WHERE NOT outside_partition
            GROUP BY linkage_id
        )
        SELECT
            alf.linkage_id,
            alf.linked_object_id,
            alf.num_obs,
            alf.num_obs_outside_partition,
            alf.num_members,
            alf.pure,
            (alf.pure AND
             opc.num_obs_in_partition IS NOT NULL AND
             opc.num_obs_in_partition = lopc.num_linkage_obs_inside_partition) AS pure_complete,
            alf.contaminated,
            alf.contamination,
            alf.mixed,
            (alf.pure AND alf.num_obs >= @min_obs) AS found_pure,
            (alf.contaminated AND alf.object_id_counts_first >= @min_obs) AS found_contaminated
        FROM all_linkages_flags AS alf
        LEFT JOIN obs_partition_counts AS opc
          ON alf.linked_object_id = opc.object_id
        LEFT JOIN linkage_obs_partition_counts AS lopc
          USING (linkage_id)
    """

    params = {
        "start_night": int(start_night),
        "end_night": int(end_night),
        "contam": float(contamination_percentage),
        "min_obs": int(min_obs),
    }
    all_linkages_tbl = _run_query_to_arrow(
        bq_client,
        bqstorage_client,
        all_linkages_query,
        params=params,
    )

    if len(all_linkages_tbl) == 0:
        # No linkages to classify; return empty tables and the original partition summary.
        return all_objects, AllLinkages.empty(), partition_summary  # type: ignore[arg-type]

    all_linkages = AllLinkages.from_kwargs(
        linkage_id=all_linkages_tbl["linkage_id"],
        partition_id=pa.repeat(partition_id, len(all_linkages_tbl)),
        linked_object_id=all_linkages_tbl["linked_object_id"],
        num_obs=all_linkages_tbl["num_obs"],
        num_obs_outside_partition=all_linkages_tbl["num_obs_outside_partition"],
        num_members=all_linkages_tbl["num_members"],
        pure=all_linkages_tbl["pure"],
        pure_complete=all_linkages_tbl["pure_complete"],
        contaminated=all_linkages_tbl["contaminated"],
        contamination=all_linkages_tbl["contamination"],
        mixed=all_linkages_tbl["mixed"],
        found_pure=all_linkages_tbl["found_pure"],
        found_contaminated=all_linkages_tbl["found_contaminated"],
    )

    # === Per-object difi metrics via BigQuery and Python join ===
    per_object_metrics_query = f"""
        WITH lm AS (
            SELECT
                {lm_linkage_col} AS linkage_id,
                {lm_obs_col} AS obs_id
            FROM `{lm_table}`
        ),
        lma AS (
            SELECT
                lm.linkage_id,
                lm.obs_id,
                obs.object_id,
                obs.night
            FROM lm
            JOIN `{obs_table}` AS obs
              ON lm.obs_id = obs.id
        ),
        unique_object_members_full AS (
            SELECT
                lma.linkage_id,
                lma.object_id,
                COUNT(*) AS object_id_counts,
                al.linked_object_id,
                al.pure,
                al.pure_complete,
                al.contaminated,
                al.mixed
            FROM lma
            JOIN (
                {all_linkages_query}
            ) AS al
            USING (linkage_id)
            GROUP BY
                lma.linkage_id,
                lma.object_id,
                al.linked_object_id,
                al.pure,
                al.pure_complete,
                al.contaminated,
                al.mixed
        )
        SELECT
            object_id,
            SUM(CASE WHEN pure AND object_id = linked_object_id THEN 1 ELSE 0 END) AS pure,
            SUM(CASE WHEN pure_complete AND object_id = linked_object_id THEN 1 ELSE 0 END) AS pure_complete,
            SUM(CASE WHEN contaminated AND object_id = linked_object_id THEN 1 ELSE 0 END) AS contaminated,
            SUM(CASE WHEN contaminated AND object_id <> linked_object_id THEN 1 ELSE 0 END) AS contaminant,
            SUM(CASE WHEN mixed THEN 1 ELSE 0 END) AS mixed,
            SUM(CASE WHEN pure THEN object_id_counts ELSE 0 END) AS obs_in_pure,
            SUM(CASE WHEN pure_complete THEN object_id_counts ELSE 0 END) AS obs_in_pure_complete,
            SUM(
                CASE
                    WHEN contaminated AND object_id = linked_object_id
                    THEN object_id_counts
                    ELSE 0
                END
            ) AS obs_in_contaminated,
            SUM(
                CASE
                    WHEN contaminated AND object_id <> linked_object_id
                    THEN object_id_counts
                    ELSE 0
                END
            ) AS obs_as_contaminant,
            SUM(CASE WHEN mixed THEN object_id_counts ELSE 0 END) AS obs_in_mixed,
            SUM(
                CASE
                    WHEN pure AND object_id_counts >= @min_obs THEN 1 ELSE 0 END
            ) AS found_pure,
            SUM(
                CASE
                    WHEN contaminated AND object_id_counts >= @min_obs THEN 1 ELSE 0 END
            ) AS found_contaminated
        FROM unique_object_members_full
        GROUP BY object_id
    """

    metrics_tbl = _run_query_to_arrow(
        bq_client,
        bqstorage_client,
        per_object_metrics_query,
        params=params,
    )

    # Join metrics onto the existing AllObjects table in Python (Arrow join).
    ao_tbl = all_objects.table
    if len(metrics_tbl) > 0:
        ao_updated_tbl = ao_tbl.join(metrics_tbl, "object_id", "object_id")
        def _col_or_zero(name: str) -> pa.Array:
            arr = ao_updated_tbl[name] if name in ao_updated_tbl.column_names else pa.array(
                [0] * len(ao_updated_tbl), type=pa.int64()
            )
            return pa.compute.fill_null(arr, 0).cast(pa.int64())  # type: ignore[attr-defined]

        all_objects_updated = AllObjects.from_kwargs(
            object_id=ao_updated_tbl["object_id"],
            partition_id=ao_updated_tbl["partition_id"],
            mjd_min=ao_updated_tbl["mjd_min"],
            mjd_max=ao_updated_tbl["mjd_max"],
            arc_length=ao_updated_tbl["arc_length"],
            num_obs=ao_updated_tbl["num_obs"],
            num_observatories=ao_updated_tbl["num_observatories"],
            findable=ao_updated_tbl["findable"],
            found_pure=_col_or_zero("found_pure"),
            found_contaminated=_col_or_zero("found_contaminated"),
            pure=_col_or_zero("pure"),
            pure_complete=_col_or_zero("pure_complete"),
            contaminated=_col_or_zero("contaminated"),
            contaminant=_col_or_zero("contaminant"),
            mixed=_col_or_zero("mixed"),
            obs_in_pure=_col_or_zero("obs_in_pure"),
            obs_in_pure_complete=_col_or_zero("obs_in_pure_complete"),
            obs_in_contaminated=_col_or_zero("obs_in_contaminated"),
            obs_as_contaminant=_col_or_zero("obs_as_contaminant"),
            obs_in_mixed=_col_or_zero("obs_in_mixed"),
        )
    else:
        all_objects_updated = all_objects

    # === Update PartitionSummary in Python, mirroring the DuckDB engine ===
    partition = partition_summary[0]

    pure_known = len(
        all_linkages.apply_mask(
            pc.and_(
                pc.equal(all_linkages.pure, True),
                pc.invert(pc.is_null(all_linkages.linked_object_id)),
            )
        )
    )
    pure_unknown = len(
        all_linkages.apply_mask(
            pc.and_(
                pc.equal(all_linkages.pure, True),
                pc.is_null(all_linkages.linked_object_id),
            )
        )
    )
    contaminated_count = len(all_linkages.select("contaminated", True))
    mixed_count = len(all_linkages.select("mixed", True))

    found = len(
        all_linkages.apply_mask(
            pc.and_(
                pc.equal(all_linkages.pure, True),
                pc.invert(pc.is_null(all_linkages.linked_object_id)),
            )
        ).linked_object_id.unique()
    )

    findable_val = (
        partition.findable[0].as_py() if not pc.is_null(partition.findable[0]).as_py() else 0
    )
    completeness = found / findable_val if findable_val and findable_val > 0 else float(found)
    completeness *= 100

    partition_summary_updated = PartitionSummary.from_kwargs(
        id=partition.id,
        start_night=partition.start_night,
        end_night=partition.end_night,
        observations=partition.observations,
        findable=partition.findable,
        found=[found],
        completeness=[completeness],
        pure_known=[pure_known],
        pure_unknown=[pure_unknown],
        contaminated=[contaminated_count],
        mixed=[mixed_count],
    )

    return all_objects_updated, all_linkages, partition_summary_updated
