"""PathSnap evaluator API. Run: python -m uvicorn backend.main:app --reload"""
from __future__ import annotations

import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.data_access import (
    dataset_root, partition_name, list_trip_ids, trip_folder,
    has_required_inputs, read_gps_preview,
)
from backend.pipeline_adapter import predict_trip

log = logging.getLogger("pathsnap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="PathSnap API", description="GPS-to-road route reconstruction", version="0.1.0")


class PredictRequest(BaseModel):
    trip_id: str = Field(..., pattern=r"^[0-9]{8}$")


@app.get("/health")
def health():
    return {"status": "ok", "data_root_exists": dataset_root().is_dir(),
            "partition": partition_name(), "available_trips": len(list_trip_ids())}


@app.get("/trips")
def trips():
    items = list_trip_ids()
    return {"partition": partition_name(), "count": len(items), "trips": items}


@app.get("/preview/{trip_id}")
def preview(trip_id: str):
    try:
        return read_gps_preview(trip_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404 if isinstance(exc, FileNotFoundError) else 400,
                            detail=str(exc)) from exc


@app.post("/predict")
def predict(request: PredictRequest):
    trip_id = request.trip_id
    folder = trip_folder(trip_id)
    if not has_required_inputs(folder, trip_id):
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")
    start = time.perf_counter()
    log.info("Prediction requested: partition=%s trip=%s", partition_name(), trip_id)
    try:
        result = predict_trip(folder, trip_id)
    except NotImplementedError as exc:
        log.warning("Pipeline not integrated: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        log.exception("Prediction failed for trip %s", trip_id)
        raise HTTPException(status_code=500, detail="Prediction failed; consult backend server logs")
    if not isinstance(result, dict) or not {"route_coordinates", "predicted_arc_ids"} <= result.keys():
        raise HTTPException(status_code=500, detail="Pipeline returned an invalid response schema")
    result["trip_id"] = trip_id
    result["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
    log.info("Prediction finished: trip=%s duration_ms=%s", trip_id, result["duration_ms"])
    return result
