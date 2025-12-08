from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BigQueryObservationsInput:
    """
    Configuration for providing observations to the BigQuery engine.

    This is a lightweight descriptor that tells difi which BigQuery
    table or view to use as the canonical observations source. The
    referenced table is expected to expose at least the columns used
    by the DuckDB engine:

      - id (STRING)
      - night (INT64)
      - object_id (STRING, nullable)
      - observatory_code (STRING)

    Additional columns (e.g. time, ra/dec, photometry) may be present
    but are not required for the current BigQuery engine, which only
    reproduces the Singletons-based cifi logic.
    """

    table: str
    """Fully-qualified BigQuery table or view name."""

    project: Optional[str] = None
    dataset: Optional[str] = None


@dataclass
class BigQueryLinkageInput:
    """
    Configuration for providing linkage member data to the BigQuery engine.

    This mirrors :class:`difi.difi.DuckDBLinkageInput` but targets a
    BigQuery table or view instead of Parquet on disk. The referenced
    table is expected to contain at least two columns that can be
    mapped into the logical difi linkage schema:

      - linkage_id_column  -> linkage identifier
      - obs_id_column      -> observation identifier
    """

    table: str
    """Fully-qualified BigQuery table or view containing linkage members."""

    linkage_id_column: str
    """Column name holding linkage identifiers."""

    obs_id_column: str
    """Column name holding observation identifiers."""


