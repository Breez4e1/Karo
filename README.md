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
 # 数据与代码

---

## 一、原始数据

**目录：** `kmls`  
保存了从 2bulu 平台上以 “冈仁波齐” 为关键字抓取到的轨迹原始文件。

---

## 二、代码及用途

1. **extract_tracks.py**  
   检查 `kmls` 目录中的每个轨迹文件，过滤具有完整 GPS 坐标轨迹以及总路程在 30KM 到 60KM 之间的轨迹文件，生成对应的 `tracks` 目录。

2. **find_continuous_tracks.py**  
   检查 `tracks` 目录中的所有数据文件，找出所有起点在塔钦中心 2KM 范围内，且经过岗加/芝热寺中心 1KM 范围内，而且在这两个点之间的轨迹没有超过 2 小时的逗留时间，总时长在 5 小时到 3 天，发生在 5 月到 10 月之间，出发时间在早上 6 点以后、下午 2 点之前的轨迹，将找到的轨迹记录在 `continuous_tracks.csv` 文件中。

3. **filter_and_trim_tracks.py**  
   根据 `continuous_tracks.csv` 中的记录，将 `tracks` 目录中对应的文件逐个处理，去除岗加/芝热寺之后的数据点，将所有加工后的数据文件存储在 `filtered_tracks` 目录中。

4. **flexible_time_predictor.py**  
   使用 `filtered_tracks` 中的数据进行 XGBoost 数据准备及模型训练。

5. **visualize_model_results.py**  
   以图形化形式展示步骤 4 所训练得到的模型及结果。

6. **visualize_track_predictions.py**  
   随机选取一个轨迹，为轨迹上的每个点生成到各个目的地的预测，计算预测误差并创建可视化图表。

7. **shap_analysis.py**  
   对步骤 5 所得到的模型进行 SHAP 分析，生成相关分析图表。

8. **display_track_map.py**  
   创建转山路线及分段点示意图。

9. **lstm_minimal.py**  
   使用一个简单的前向神经网络对步骤 4 所得到的数据进行训练并得到相关性能数据，用于对比分析 XGBoost 模型和简单神经网络模型之间的性能差异。

---

## 三、文章目录

**源文件：** `Karo_English copy.tex`  
位于 `paper` 目录下，包含论文正文文件。
---

## III. Paper Directory

**Source file:** `Karo_English copy.tex`  
Located in the `paper` directory, containing the manuscript of the research paper.


