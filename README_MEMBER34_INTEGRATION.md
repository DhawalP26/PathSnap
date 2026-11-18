# PathSnap Members 3 + 4: live route reconstruction integration

This is an **overlay** for the current PathSnap `main` branch. It adds the Member 4
FastAPI + Streamlit application and connects it to the merged Member 1, 2 and 3
inference APIs. It does **not** include a trained production classifier, private
dataset, teacher labels, or any fabricated predictions.

## What is integrated

1. `preprocessing.prepare_trip(...)` loads only `.track`, `.nodes`, `.arcs`.
2. `models.final_review_integration.load_final_model(...)` verifies the saved
   **40-feature `review1_extended_lattice`** artifact.
3. `models.score_candidates.score_review1(...)` scores the full prepared trip.
4. `reconstruction.route_reconstruction.reconstruct_prepared_trip(...)` runs
   directed Viterbi and returns separate geometry for disconnected routes.
5. `backend.main` serves real inference via POST `/predict`.
6. `frontend.app` renders blue GPS and each green road polyline separately.

There is no inference-time `.route` access or user-supplied model file path.
Candidate coverage and route-break counts are **diagnostics, not accuracy**.

## Apply to an up-to-date PathSnap clone

From your existing clone on the computer containing the model and dataset:

```powershell
git checkout main
git pull origin main
git checkout -b feature/member34-live-integration
```

Copy **the contents** of this ZIP (not its containing folder) into your
repository root, merging the `backend`, `frontend`, `tests`, and `scripts` folders.
Do not overwrite any Member 1/2/3 model or reconstruction sources. Then:

```powershell
python -m pip install -r requirements-member1.txt -r requirements_member2.txt -r requirements.txt -r requirements-member4.txt
python -m pytest -q tests/test_member34_integration.py
python -m pytest -q tests preprocessing/tests
```

The four team modules must be importable from the repo root. `PATHSNAP_MODEL_PATH`
must point to a trustworthy, actual 40-feature production artifact. The old
8-feature `models/best_model.pkl` does **not** satisfy this contract.

## Start backend (Windows PowerShell)

```powershell
$env:PATHSNAP_DATA_DIR = 'C:\Users\YOUR_USER\Downloads\Student folder'
$env:PATHSNAP_PARTITION = 'student_test'
$env:PATHSNAP_MODEL_PATH = "$PWD\results\member2\final_review\final_model.pkl"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Replace the dataset path with your machine's path. Verify:

- `http://127.0.0.1:8000/health`: `model_ready: true`
- `http://127.0.0.1:8000/trips`: ten student-test IDs
- `http://127.0.0.1:8000/docs`: interactive endpoint documentation

If `model_ready` is false, request the finished **all-80-trip final_model.pkl**
from Member 2. The API returns a transparent HTTP 503 when it is missing.

## Start UI

```powershell
$env:PATHSNAP_API_URL = 'http://127.0.0.1:8000'
python -m streamlit run frontend/app.py
```

On a second laptop, set `PATHSNAP_API_URL` to the backend laptop's private LAN
IPv4 URL. Keep both computers on the same network; open port 8000 only on a
trusted network. The UI draws each Viterbi route fragment separately to avoid
falsely joining disconnected streets.

## Run all ten student-test HTTP smoke tests

```powershell
python -m scripts.smoke_student_api --api http://127.0.0.1:8000
```

Results go into `results/member4/student_test_api_smoke.json`. This is a real,
reference-free inference smoke test. It intentionally does not calculate
accuracy from teacher-test references.

## Git delivery

```powershell
git add backend frontend tests/test_member34_integration.py scripts requirements-member4.txt README_MEMBER34_INTEGRATION.md
git commit -m "Integrate real 40-feature ML with Viterbi and FastAPI Streamlit UI"
git push -u origin feature/member34-live-integration
```

Open a pull request to `main`, review team changes, and merge only after CI and
integration testing succeed. This overlay is an implementation package, not a
claim that GitHub was updated automatically. Inference is still blocked until
Member 2's production artifact is available and tested.

## Remaining Member 3 items

- Run real per-trip decoder tests on the training-trip reference routes **offline**.
- Check unexpected restarts, wrong-way arcs, disconnected topology and GPS gaps.
- Separate **decoded route accuracy** from **conditional candidate top-1**.
- Do not expose a `.route` file to live API requests.
