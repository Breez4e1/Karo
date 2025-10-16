# Data and Code

---

## I. Raw Data

**Directory:** `kmls`  
Stores the original trajectory files collected from the 2bulu platform using **“Mount Kailash”** as the keyword.

---

## II. Code and Usage

1. **`extract_tracks.py`**  
   Checks each trajectory file in the `kmls` directory, filters out those with complete GPS coordinate tracks and total distances between **30 km and 60 km**, and generates the corresponding files in the `tracks` directory.

2. **`find_continuous_tracks.py`**  
   Examines all data files in the `tracks` directory and identifies tracks that meet the following criteria:  
   - The starting point is within **2 km** of the Darchen center.  
   - The route passes within **1 km** of the Gangjia/Zhire Temple center.  
   - There are **no stops longer than 2 hours** between these two points.  
   - The total duration is between **5 hours and 3 days**.  
   - The activity occurred between **May and October**.  
   - The departure time is between **6:00 AM and 2:00 PM**.  
   All qualifying tracks are recorded in the `continuous_tracks.csv` file.

3. **`filter_and_trim_tracks.py`**  
   Based on the records in `continuous_tracks.csv`, processes each corresponding file in the `tracks` directory, removes data points beyond the Gangjia/Zhire Temple section, and stores all processed data files in the `filtered_tracks` directory.

4. **`flexible_time_predictor.py`**  
   Uses the data in `filtered_tracks` for **XGBoost-based** data preparation and model training.

5. **`visualize_model_results.py`**  
   Graphically displays the model and results obtained in Step 4.

6. **`visualize_track_predictions.py`**  
   Randomly selects a trajectory, generates predictions from each point to various destinations, calculates prediction errors, and creates visual charts.

7. **`shap_analysis.py`**  
   Conducts **SHAP analysis** on the model obtained in Step 5 and generates related analytical plots.

8. **`display_track_map.py`**  
   Creates a schematic diagram of the **Mount Kailash Kora** route and its segment points.

9. **`lstm_minimal.py`**  
   Trains a **simple feedforward neural network** using the data obtained in Step 4 and produces performance metrics to compare the **XGBoost** model with a basic neural network model.

---

## III. Paper Directory

**Source file:** `Karo_English copy.tex`  
Located in the `paper` directory, containing the manuscript of the research paper.


