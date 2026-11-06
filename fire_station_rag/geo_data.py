"""Dependency-free ingestion and validated queries for every workspace GeoJSON."""
import json
import math
import os
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
DB = DATA / "geo_features.sqlite3"
CATALOG = DATA / "layer_catalog.json"


def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def ingest():
    paths = sorted(p for p in ROOT.rglob("*.geojson") if "fire_station_rag" not in p.parts)
    signature = [[p.relative_to(ROOT).as_posix(), p.stat().st_mtime_ns, p.stat().st_size] for p in paths]
    if CATALOG.exists() and DB.exists():
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        if catalog.get("signature") == signature:
            return catalog
    DATA.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB)
    layers = []
    try:
        with conn:
            conn.execute("CREATE TABLE IF NOT EXISTS features (id TEXT PRIMARY KEY, layer TEXT, properties TEXT, geometry TEXT)")
            conn.execute("CREATE INDEX IF NOT EXISTS feature_layer ON features(layer)")
            conn.execute("DELETE FROM features")
            for path in paths:
                source = path.relative_to(ROOT).as_posix()
                collection = json.loads(path.read_text(encoding="utf-8-sig"))
                features = collection.get("features", [])
                fields, numeric, types, labels = set(), set(), set(), set()
                for index, feature in enumerate(features):
                    props = feature.get("properties") or {}
                    fields.update(props)
                    numeric.update(k for k, v in props.items() if isinstance(v, (int, float))
                                   and not isinstance(v, bool) and math.isfinite(v))
                    geometry = feature.get("geometry")
                    if geometry:
                        types.add(geometry.get("type", "unknown"))
                    for key in ("Layer", "Sub_Class"):
                        if props.get(key):
                            labels.add(str(props[key]))
                    conn.execute("INSERT INTO features VALUES (?,?,?,?)", (
                        f"{source}#{index}", source, json.dumps(props), json.dumps(geometry)))
                layers.append({"source": source, "name": path.stem.replace("_", " "),
                               "category": path.parent.name, "feature_count": len(features),
                               "fields": sorted(fields), "numeric_fields": sorted(numeric),
                               "geometry_types": sorted(types), "labels": sorted(labels)})
    finally:
        conn.close()
    catalog = {"signature": signature, "layers": layers,
               "feature_count": sum(layer["feature_count"] for layer in layers)}
    CATALOG.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog


def resolve_layer(value, catalog):
    query = normalize(value.removesuffix(".geojson"))
    matches = []
    for layer in catalog["layers"]:
        aliases = {normalize(layer["name"]), normalize(layer["source"].removesuffix(".geojson"))}
        aliases.update(normalize(label) for label in layer["labels"])
        aliases |= {a + "s" for a in list(aliases) if not a.endswith("s")}
        if query in aliases:
            matches.append(layer)
    if len(matches) != 1:
        raise ValueError("Layer is unknown or ambiguous; specify its exact source path from /layers")
    return matches[0]


def explicit_count(question, catalog):
    text = normalize(question)
    match = re.fullmatch(r"(?:how many|count|total number of|number of) (.+?)(?: are there| exist| do we have)?", text)
    if not match:
        return None
    subject = match.group(1)
    area_type = None
    for area in ("airside", "cityside"):
        prefix = re.fullmatch(area + r" (.+)", subject)
        suffix = re.fullmatch(r"(.+?) (?:are|are in|are on|in|on|are there in) " + area, subject)
        candidate = prefix.group(1) if prefix else suffix.group(1) if suffix else None
        if candidate:
            # Only treat this as a filter when the remaining layer is recognized.
            try:
                resolve_layer(candidate, catalog)
            except ValueError:
                continue
            subject, area_type = candidate, area.capitalize()
            break
    try:
        layer = resolve_layer(subject, catalog)
    except ValueError:
        return None
    return {"operation": "count", "layer": layer["source"], "area_type": area_type}


def explicit_measurement(question, catalog):
    """Common whole-layer aggregates do not need an LLM inference round trip."""
    text = normalize(question)
    match = re.fullmatch(
        r"(?:what is (?:the )?)?(?:the )?(total|sum|average|minimum|maximum) "
        r"(?:recorded )?(area|perimeter|ground elevation) (?:of|for) (.+)", text)
    if not match:
        return None
    aggregate, measurement, subject = match.groups()
    area_type = None
    for area in ("airside", "cityside"):
        if subject.startswith(area + " "):
            area_type, subject = area.capitalize(), subject[len(area) + 1:]
            break
    try:
        layer = resolve_layer(subject, catalog)
    except ValueError:
        return None
    candidates = {"area": ["Area", "Shape_Area"], "perimeter": ["Perimeter"],
                  "ground elevation": ["Ground_Elv"]}[measurement]
    field = next((field for field in candidates if field in layer["numeric_fields"]), None)
    if not field:
        return None
    return {"operation": {"total": "sum", "sum": "sum", "average": "average",
                          "minimum": "min", "maximum": "max"}[aggregate],
            "layer": layer["source"], "field": field, "area_type": area_type}


def query(catalog, layer, operation, field=None, name=None, area_type=None,
          minimum=None, maximum=None, fields=None):
    info = resolve_layer(layer, catalog)
    if operation not in {"list", "count", "sum", "average", "min", "max", "argmin", "argmax"}:
        raise ValueError("Unsupported operation")
    if field is not None and field not in info["numeric_fields"]:
        raise ValueError(f"Field {field!r} is not a recorded numeric field for {info['name']}")
    if fields is not None:
        if operation != "list":
            raise ValueError("Display columns are supported only for list queries")
        if not fields or any(column not in info["fields"] for column in fields):
            raise ValueError("Display columns must be recorded fields in the selected layer")
    if (operation not in {"list", "count"} or minimum is not None or maximum is not None) and field is None:
        raise ValueError("Select a verified recorded numeric field")
    if name and "Name" not in info["fields"]:
        raise ValueError("This layer has no Name field")
    if area_type and "Area_Type" not in info["fields"]:
        raise ValueError("This layer has no Area_Type field; cannot apply that filter")
    if os.getenv("DATABASE_URL"):
        from postgres_store import records
        raw = records(info["source"])
    else:
        conn = sqlite3.connect(DB)
        try:
            raw = conn.execute("SELECT id,properties FROM features WHERE layer=?", (info["source"],)).fetchall()
        finally:
            conn.close()
    records = []
    missing = 0
    for record_id, props_json in raw:
        props = json.loads(props_json)
        if name and str(props.get("Name", "")).casefold() != name.casefold():
            continue
        if area_type and str(props.get("Area_Type", "")).casefold() != area_type.casefold():
            continue
        value = props.get(field)
        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        if field and not valid:
            missing += 1
        if minimum is not None and (not valid or value < minimum):
            continue
        if maximum is not None and (not valid or value > maximum):
            continue
        records.append({**props, "id": record_id})
    numeric = [r for r in records if isinstance(r.get(field), (int, float))
               and not isinstance(r.get(field), bool) and math.isfinite(r[field])] if field else []
    result = {"source": info["source"], "layer": info["name"], "matched_count": len(records),
              "missing_measurements": missing, "units": "unverified"}
    if operation == "count":
        result["value"] = len(records)
    elif operation == "list":
        visible = records[:50]
        if fields:
            visible = [{"id": r["id"], **{column: r.get(column) for column in fields}} for r in visible]
        result.update(records=visible, truncated=len(records) > 50, display_fields=fields)
    elif operation in {"argmin", "argmax"}:
        target = (min if operation == "argmin" else max)([r[field] for r in numeric], default=None)
        winners = [r for r in numeric if r[field] == target]
        result.update(value=target, records=winners[:50], truncated=len(winners) > 50,
                      measurements_used=len(numeric))
    else:
        values = [r[field] for r in numeric]
        result["measurements_used"] = len(values)
        result["value"] = None if not values else {
            "sum": lambda: math.fsum(values), "average": lambda: math.fsum(values) / len(values),
            "min": lambda: min(values), "max": lambda: max(values),
        }[operation]()
    return result


if __name__ == "__main__":
    catalog = ingest()
    print(f"Indexed {len(catalog['layers'])} layers and {catalog['feature_count']} features into {DB}")
