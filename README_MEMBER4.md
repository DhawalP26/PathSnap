# PathSnap — Member 4 FastAPI + Streamlit Starter

This is a **development scaffold**, not a finished ML submission. It loads real GPS
tracks and displays a real GPS preview, but `POST /predict` deliberately returns
HTTP 503 until Members 1–3 provide actual inference functions. **No fake ML
results or hardcoded routes are included.**

## Copy into team repository

Copy `backend/`, `frontend/`, `tests/`, `requirements-member4.txt`, and `.gitignore`
into the root of the `PathSnap` Git repository. Merge rather than overwrite if
teammates have created the same files.

## Laptop 1 — API

Open PowerShell in your cloned repository:

```powershell
python -m pip install -r requirements-member4.txt
$env:PATHSNAP_DATA_DIR="$HOME\Downloads\Student folder"
$env:PATHSNAP_PARTITION="student_test"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Modify the data path to match **your** computer. This expects the `student_test`
subdirectory and its numeric trip folders. To serve organizer-reserved evaluator
trips later, point PATHSNAP_DATA_DIR at their parent folder and set
PATHSNAP_PARTITION to the supplied partition name. The evaluator data is *not*
currently present in the supplied student ZIP.

Check in a browser on Laptop 1:

- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/trips
- http://127.0.0.1:8000/docs

The development-trip list should report ten student-test trips.

## Laptop 2 — UI

Both laptops must connect to the same reachable local network. On Laptop 1
use `ipconfig` to obtain its IPv4 address (e.g., `192.168.1.25`).
On Laptop 2, in a clone of the repository:

```powershell
python -m pip install -r requirements-member4.txt
$env:PATHSNAP_API_URL="http://192.168.1.25:8000"
python -m streamlit run frontend/app.py
```

Replace the example IP with Laptop 1's actual local IPv4 address. Open the URL
printed by Streamlit. The map displays actual raw GPS coordinates from API.
The Predict button will return 503 until the real model is connected.

**If Laptop 2 cannot access the API:** verify both devices are on the same
network, the API binds to `0.0.0.0`, and Windows Firewall permits inbound TCP
port 8000 on your private network. Diagnose with
`http://LAPTOP1_IP:8000/health` in Laptop 2's browser.

## API contract

- `GET /health` → process and configured data status.
- `GET /trips` → `{"partition":"student_test","count":10,"trips":["00000000",...]}`
- `GET /preview/{trip_id}` → sampled raw GPS only (`[latitude,longitude,time]`).
- `POST /predict` ← `{"trip_id":"00000000"}`

Expected response from the *real* pipeline:

```json
{
  "trip_id": "00000000",
  "predicted_arc_ids": [10, 11, 12],
  "route_coordinates": [[56.2, 54.9], [56.21, 54.91]],
  "metrics": {"route_continuity": 0.95},
  "duration_ms": 152.3
}
```

The response above is only a **schema illustration**, not a measured
prediction. Coordinates sent to Folium must be `[latitude, longitude]` even
though `.track` and `.nodes` contain `(longitude, latitude)`.

## Integrate when Members 1, 2 and 3 finish

Edit `backend/pipeline_adapter.py`. This function should:

1. Call Member 1's loader and candidate feature generator for the selected trip.
2. Call Member 2's saved model/pipeline to score candidates.
3. Call Member 3's HMM/Viterbi reconstruction to get predicted arc IDs and road
   geometry.
4. **Only after inference:** evaluate predictions on the server, when permitted
   reference labels are available.
5. Return JSON matching the contract above. Do not return reference labels,
   reference route geometry, or any ground truth to the UI. No hardcoded routes.

Ask teammates to provide function signatures, feature schema, artifact path and
expected output coordinates early.

## Test

```powershell
python -m pytest tests -q
```

Run these before your demo:

- Verify `/trips` shows all available evaluation trips dynamically.
- Verify the UI displays every trip; no fixed list.
- Predict more than one unseen trip and inspect real server logs.
- Test Laptop 2 to Laptop 1 connection on the evaluation network.
- Capture UI screenshots and **backend CMD/terminal API log** screenshots.
- Confirm model reference labels are not exposed to UI/prediction.
- Include API/UI run instructions in root README.
- Commit your changes to `feature/member4-api-ui`, open a PR, and merge with team.

## Git workflow

```powershell
git add backend frontend tests requirements-member4.txt .gitignore README_MEMBER4.md
git commit -m "Add Member 4 FastAPI and Streamlit scaffold"
git push -u origin feature/member4-api-ui
```

Do **not** commit passwords, private tokens, evaluator ground truth, or large
private datasets. Disclose AI assistance if required by event policy, and make
sure you can explain the submitted implementation in a viva.
