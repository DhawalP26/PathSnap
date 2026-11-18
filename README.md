# PathSnap — Distributed Map Matching & Route Reconstruction

**Team 43 | Machine Learning Hackathon Project**

**GitHub Repository:** https://github.com/DhawalP26/PathSnap

## 1. Project Overview

PathSnap is an intelligent GPS-based map-matching and route-reconstruction system designed to convert noisy GPS observations into accurate and geographically plausible road routes.

The system integrates machine learning, geospatial feature engineering, graph algorithms, and interactive visualization to predict which road segments a vehicle has traveled.

Our solution combines:

- Data preprocessing and GPS feature engineering
- Road-segment candidate generation
- Random Forest classification using 40 engineered features
- Bayesian hyperparameter optimization using Optuna
- Hidden Markov Model (HMM) and Viterbi decoding
- Directed road-network route reconstruction
- FastAPI backend
- Streamlit Route Studio frontend
- Interactive OpenStreetMap visualization

The project's final reported Random Forest candidate-level accuracy is **93.74%**, achieved after fine-tuning with **20 Optuna trials**.

## 2. Team Details

**Team Number: 43**

| SRN | Team Member |
|---|---|
| PES1UG24CS150 | DHAROOR SHIVA SAI VARUN |
| PES1UG24CS151 | DHAWAL PATHAK |
| PES1UG24CS168 | GANESH M |
| PES1UG24CS169 | GAUTAM PRASANTH NAIR |

## 3. Problem Statement

GPS measurements often contain errors due to signal noise, buildings, road intersections, parallel roads, and irregular sampling intervals.

Simply assigning every GPS observation to the nearest road segment may produce incorrect or disconnected routes.

PathSnap addresses this problem by:

1. Identifying nearby candidate road segments for each GPS observation.
2. Extracting spatial, temporal, geometric, and contextual features.
3. Predicting the most likely road segments using machine learning.
4. Using HMM/Viterbi decoding to reconstruct a plausible route.
5. Displaying the reconstructed route through an interactive web interface.

## 4. Final Model Accuracy

The Random Forest classifier was selected for its strong performance on the engineered GPS and road-network features.

### Random Forest Performance

| Parameter | Result |
|---|---|
| Final model | Random Forest Classifier |
| Number of features | 40 |
| Original accuracy | 89.9485% |
| Accuracy after fine-tuning | **93.74%** |
| Improvement | **+3.79 percentage points** |
| Optimization framework | Optuna |
| Optimization technique | Bayesian optimization using TPE |
| Number of trials | 20 |
| Training trips | 80 |
| Development trips | 60 |
| Internal holdout trips | 20 |
| Cross-validation | 3-fold GroupKFold |
| Optimization objective | Candidate top-1 accuracy |

### Accuracy Improvement

**Before fine-tuning: 89.9485%**

**After fine-tuning: 93.74%**

**Overall improvement: +3.79 percentage points**

The 93.74% result represents the team's final reported candidate-level map-matching accuracy. It should not be interpreted as independently measured end-to-end Viterbi reconstructed-route accuracy.

## 5. Machine Learning Models Evaluated

Five machine-learning algorithms were implemented and compared.

| Model | Initial 40-feature candidate top-1 accuracy |
|---|---:|
| Random Forest | **89.9485%** |
| HistGradientBoosting | 89.5762% |
| Logistic Regression | 88.6630% |
| Multilayer Perceptron | 88.3864% |
| Decision Tree | 87.7642% |

Random Forest achieved the highest performance in the original five-model comparison and was subsequently selected for advanced hyperparameter tuning.

The final fine-tuning experiment used 20 Optuna trials and produced the reported accuracy of **93.74%**.

## 6. Dataset Description

The project uses GPS trip observations and corresponding directed road-network data.

| Dataset Partition | Number of Trips |
|---|---:|
| Training | 80 |
| Student Test | 10 |
| Teacher Test | 10 |
| Total | 100 |

### Dataset Files

- `.track` — GPS coordinates and timestamps
- `.nodes` — Road-network nodes
- `.arcs` — Directed road segments
- `.route` — Reference route information for permitted training and evaluation

The reference `.route` files are not used during live inference.

### Feature Engineering

The project uses **40 engineered features**, including:

- GPS-to-road distance
- Heading difference
- GPS speed and movement
- Timestamp differences
- Road-segment geometry
- Candidate ranking
- Relative candidate information
- Temporal movement context
- Road connectivity
- Directed network transition information
- Turn and road-position features

These features help the classifier distinguish between nearby candidate roads.

## 7. System Architecture

```text
                  GPS Track Data
                       |
                       v
              Data Preprocessing
                       |
                       v
           Candidate Road Generation
                       |
                       v
             40-Feature Engineering
                       |
                       v
              Random Forest Model
                       |
                       v
          Road Candidate Probabilities
                       |
                       v
           Hidden Markov Model (HMM)
                       |
                       v
              Viterbi Algorithm
                       |
                       v
            Route Reconstruction
                       |
                       v
                FastAPI Backend
                       |
                       v
             Streamlit Route Studio
                       |
                       v
           Interactive OpenStreetMap
```

### Member 1 — Data Preprocessing

Responsible for:

- Loading and validating GPS datasets
- Exploratory data analysis
- Road-segment candidate generation
- Spatial feature engineering
- Preparing supervised training features

### Member 2 — Machine Learning

Responsible for:

- Training five ML classifiers
- Comparing model performance
- Engineering and evaluating extended features
- Random Forest fine-tuning
- Optuna Bayesian optimization
- Exporting the trained model for inference

### Member 3 — Route Reconstruction

Responsible for:

- Directed road-graph construction
- HMM emission and transition scoring
- Viterbi route decoding
- Shortest-path reconstruction
- Handling disconnected route segments

### Member 4 — Backend and Frontend

Responsible for:

- FastAPI integration
- Streamlit Route Studio development
- Model and Viterbi integration
- Interactive GPS and predicted-route visualization
- API testing and application deployment
- GitHub integration and technical documentation

## 8. Technologies Used

| Category | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Hyperparameter Optimization | Optuna |
| Data Processing | NumPy, Pandas |
| Geospatial Processing | PyProj and spatial-processing utilities |
| Route Reconstruction | HMM, Viterbi, directed graph algorithms |
| Backend | FastAPI |
| Frontend | Streamlit |
| Map Visualization | Folium, OpenStreetMap |
| Model Serialization | Joblib |
| Testing | Pytest |
| Version Control | Git and GitHub |

## 9. Project Structure

```text
PathSnap/
|
|-- preprocessing/
|   |-- Data loading
|   |-- Candidate generation
|   |-- Feature engineering
|
|-- models/
|   |-- Model training
|   |-- Random Forest optimization
|   |-- Candidate scoring
|
|-- reconstruction/
|   |-- Directed road graph
|   |-- HMM and Viterbi
|   |-- Route reconstruction
|
|-- backend/
|   |-- main.py
|   |-- pipeline_adapter.py
|   |-- data_access.py
|
|-- frontend/
|   |-- app.py
|   |-- route_utils.py
|
|-- scripts/
|   |-- smoke_student_api.py
|
|-- requirements.txt
|-- requirements-member1.txt
|-- requirements_member2.txt
|-- requirements-member4.txt
|-- README.md
```

## 10. Installation Instructions

### Step 1 — Clone the Repository

```powershell
git clone https://github.com/DhawalP26/PathSnap.git
cd PathSnap
```

### Step 2 — Install Dependencies

```powershell
python -m pip install -r requirements-member1.txt -r requirements_member2.txt -r requirements.txt -r requirements-member4.txt
```

### Step 3 — Configure Dataset and Model

Obtain the dataset from the authorized project source.

The trained 40-feature production model should be available at:

```text
results/member2/final_review/final_model.pkl
```

Use a trusted model artifact compatible with the application's 40-feature inference pipeline.

The model file and private datasets are not distributed through the public GitHub repository.

## 11. Running the Application

### Terminal 1 — Start FastAPI

Open PowerShell inside the project directory.

```powershell
$env:PATHSNAP_DATA_DIR = "$HOME\Downloads\Student folder"

$env:PATHSNAP_PARTITION = "student_test"

$env:PATHSNAP_MODEL_PATH = "$PWD\results\member2\final_review\final_model.pkl"

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The API will be available at:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

### Terminal 2 — Start Streamlit

Open another PowerShell terminal.

```powershell
cd PathSnap

$env:PATHSNAP_API_URL = "http://127.0.0.1:8000"

python -m streamlit run frontend/app.py
```

Streamlit will display the PathSnap Route Studio application.

### Using Route Studio

1. Select a GPS trip from the dropdown.
2. Preview the GPS track on OpenStreetMap.
3. Click **Predict Route**.
4. The backend generates features and scores road candidates using Random Forest.
5. The Viterbi algorithm reconstructs the route.
6. The reconstructed route appears on the map.

**Map colors:**

- Blue — Original GPS observations
- Green — Predicted road route

Disconnected Viterbi route segments are displayed independently to avoid incorrectly connecting unrelated roads.

## 12. API Endpoints

| HTTP Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check backend and model readiness |
| GET | `/trips` | List available trips |
| GET | `/preview/{trip_id}` | Preview GPS observations |
| POST | `/predict` | Generate reconstructed route |

### Example Prediction Request

```json
{
  "trip_id": "00000000"
}
```

The prediction response contains the predicted road arc IDs, route coordinates, route breaks, execution time, and diagnostic information.

## 13. Testing and Validation

The project's recorded automated testing results include:

- **10 Member 3/4 integration tests passed**
- **64 project tests passed**
- Streamlit prediction smoke test passed
- The production model was successfully loaded and used for a student-trip prediction

### Run Integration Tests

```powershell
python -m pytest -q tests/test_member34_integration.py
```

### Run Project Tests

```powershell
python -m pytest -q tests preprocessing/tests
```

### Run Student-Test API Smoke Tests

Start the backend first, then run:

```powershell
python -m scripts.smoke_student_api --api http://127.0.0.1:8000
```

The results are stored in:

```text
results/member4/student_test_api_smoke.json
```

These smoke tests check whether predictions execute successfully. They do not calculate accuracy without reference route labels.

## 14. Two-Laptop Deployment

PathSnap supports a distributed application setup.

**Laptop 1:** Runs FastAPI, the trained machine-learning model, and route reconstruction.

**Laptop 2:** Runs the Streamlit Route Studio interface.

Both laptops should be connected to the same trusted local network.

On Laptop 1, start the server:

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

On Laptop 2, configure the backend IP address:

```powershell
$env:PATHSNAP_API_URL = "http://SERVER_LAN_IP:8000"

python -m streamlit run frontend/app.py
```

Replace `SERVER_LAN_IP` with Laptop 1's actual local network IP address.

Verify connectivity using the `/health` endpoint before running predictions.

## 15. Challenges and Solutions

### GPS Noise

GPS positions do not always align directly with road segments.

**Solution:** Spatial candidate generation and engineered geometric features.

### Road Ambiguity

Multiple roads may exist close to an observation.

**Solution:** Random Forest candidate scoring using spatial, temporal, and contextual features.

### Disconnected Routes

Independent road predictions may not form physically plausible continuous routes.

**Solution:** Directed HMM transition modeling and Viterbi route reconstruction.

### Model Performance

The original Random Forest achieved approximately 89.95% candidate-level accuracy.

**Solution:** Optuna-based Random Forest fine-tuning using 20 trials, producing the team's final reported accuracy of 93.74%.

### Application Integration

Different team modules must operate together.

**Solution:** A complete FastAPI prediction pipeline and Streamlit interface integrating preprocessing, ML, and reconstruction.

## 16. Limitations and Future Improvements

Potential future improvements include:

- Better handling of highly noisy GPS observations
- Improved matching around parallel roads and complex intersections
- More robust transition-probability calibration
- Further feature engineering and model evaluation
- Reduced model inference latency
- Additional route-level evaluation
- Improved deployment and scalability

The 93.74% candidate-level Random Forest accuracy and the accuracy of the fully reconstructed Viterbi route are separate measures.

## 17. Conclusion

PathSnap demonstrates an integrated machine-learning and graph-based approach to GPS map matching and route reconstruction.

The project combines 40-feature Random Forest candidate scoring with directed HMM/Viterbi decoding and a complete FastAPI–Streamlit application.

The team's final reported Random Forest accuracy reached **93.74% after 20 Optuna trials**, improving upon the original baseline of **89.9485%** by approximately **3.79 percentage points**.

PathSnap provides an interactive interface for selecting GPS trips and visualizing reconstructed road routes using OpenStreetMap.

The project demonstrates how supervised machine learning and sequence-based route reconstruction can be combined into a practical geospatial application.

## 18. Project Repository

**GitHub:** https://github.com/DhawalP26/PathSnap

**Team:** 43

**Project:** PathSnap — Distributed Map Matching & Route Reconstruction

**Year:** 2026

---

**Developed by Team 43 | Machine Learning Hackathon 2026**
