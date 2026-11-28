"""Canonical Arrow physical-schema validation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

_CONTRACT_PATH = Path(__file__).with_name("arrow_schema_contract.json")


@lru_cache(maxsize=1)
def _contract_tables() -> Mapping[str, Any]:
    """Load the packaged Arrow schema contract."""

    contract = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
    tables = contract.get("tables") if isinstance(contract, Mapping) else None
    if not isinstance(tables, Mapping):
        raise ValueError(f"Arrow schema contract is missing tables: {_CONTRACT_PATH}")
    return tables


def _type_matches(data_type: Any, datatype: str) -> bool:
    """Return whether a PyArrow type matches one contract datatype."""

    import pyarrow as pa

    match datatype:
        case "string":
            return bool(pa.types.is_string(data_type))
        case "int64":
            return bool(pa.types.is_int64(data_type))
        case "bool":
            return bool(pa.types.is_boolean(data_type))
        case "float64":
            return bool(pa.types.is_float64(data_type))
        case "list<string>":
            return bool(pa.types.is_list(data_type) and pa.types.is_string(data_type.value_type))
        case "fixed_size_list<float32>":
            return bool(pa.types.is_fixed_size_list(data_type) and pa.types.is_float32(data_type.value_type))
        case _:
            raise ValueError(f"Arrow schema contract contains unsupported datatype {datatype!r}")


def validate_arrow_schema(
    schema: Any,
    *,
    table_name: str,
    columns: Iterable[str] | None = None,
) -> None:
    """Validate a full contract table or a requested column subset.

    Full-table validation requires every contract column marked ``required``
    and validates every optional column that is present. Subset validation
    requires and validates exactly the requested contract columns.
    """

    raw_specs = _contract_tables().get(table_name)
    if not isinstance(raw_specs, list):
        raise ValueError(f"Arrow schema contract is missing table {table_name!r}")
    specs: dict[str, Mapping[str, Any]] = {}
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Arrow schema contract has invalid column metadata for {table_name!r}")
        column_name = str(raw_spec["name"])
        if column_name in specs:
            raise ValueError(f"Arrow schema contract table {table_name!r} repeats column {column_name!r}")
        specs[column_name] = raw_spec

    requested = None if columns is None else tuple(dict.fromkeys(str(column) for column in columns))
    if requested is not None:
        unknown = sorted(set(requested).difference(specs))
        if unknown:
            raise ValueError(f"Arrow schema contract table {table_name!r} has no columns {unknown}")
        names_to_validate = requested
    else:
        names_to_validate = tuple(specs)

    for column_name in names_to_validate:
        spec = specs[column_name]
        field_index = schema.get_field_index(column_name)
        required = requested is not None or bool(spec.get("required"))
        if field_index < 0:
            if required:
                raise ValueError(f"Arrow {table_name} table is missing required column {column_name!r}")
            continue
        datatype = str(spec["datatype"])
        actual_type = schema.field(field_index).type
        if not _type_matches(actual_type, datatype):
            raise ValueError(f"Arrow {table_name} column {column_name!r} expected {datatype}, got {actual_type}")


def validate_arrow_file_schema(
    path: str | Path,
    *,
    table_name: str,
    columns: Iterable[str] | None = None,
) -> None:
    """Validate an Arrow IPC file schema without materializing its rows."""

    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        validate_arrow_schema(reader.schema, table_name=table_name, columns=columns)
