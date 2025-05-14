# YOLOv8 物件偵測模型訓練與 TF.js 導出

本專案展示如何使用 Ultralytics YOLOv8 訓練一個自訂物件偵測模型，並將其導出為 TensorFlow.js (TF.js) 格式，以便在網頁應用程式中部署。

## 功能

*   使用 YOLOv8 進行模型訓練。
*   將訓練好的模型導出為多種格式，主要目標為 TF.js。
*   透過編輯 Python 腳本頂部的變數進行配置。

## 先決條件

在開始之前，請確保您的系統已安裝以下軟體：

1.  **Python** (建議版本 3.11 或更高版本)
2.  **pip** (Python 套件安裝程式)

## 設定步驟

1.  **複製 (或下載) 專案檔案**:
    如果您是從 Git 儲存庫取得此專案：
    ```bash
    git clone https://github.com/JimmyHon1984/AI-Train-yolov8-roboflow.git
    cd AI-Train-yolov8-roboflow
    ```
    否則，請確保您已將 `train_export_model.py` 和 `requirements.txt` 檔案放置在您的工作目錄中。

2.  **安裝 Python 依賴套件**:
    在您的專案目錄中開啟終端機，然後執行：
    ```bash
    pip install -r requirements.txt
    ```

    ***如果 traino_export_model 轉換模型中出現問題***:
    ```
    如果您在導出為 TF.js 格式時遇到問題，您可能需要明確安裝 `tensorflowjs`：
    ```bash
    pip install tensorflowjs
    ```

## 工作流程

### 1. 下載數據集

本專案直接從 Roboflow Universe 下載預先準備好的數據集。

*   **執行以下命令以下載並解壓縮數據集：**
    
    ### 執行 Python 腳本下載花卉數據集

    此選項將下載一個花卉數據集，其中包含約 3000 張圖片，涵蓋 13 個類別。

    執行 `get_data.py` 腳本以下載數據集：

    ```bash
    python get_data.py
    ```


### 2. 準備自定義數據集樣本

如果您想從原始數據集中創建一個較小的樣本以加速訓練或測試，可以使用 dataset_sampler.py 腳本：

*   **執行以下命令以調整數據集：**
    ```bash
    python dataset_sampler.py \
        --src original_dataset \
        --total 500 \
        --train-ratio 8 \
        --test-ratio 1 \
        --valid-ratio 1 \
        --output sampled_dataset
    ```

**參數說明：**

*   **`--src 或 -s`**: **(必需)** 原始數據集的路徑
*   **`--total 或 -t`**: **(必需)** 要採樣的圖像總數
*   **`--train-ratio 或 -tr`**: (可選) 訓練集比例，默認值為 7
*   **`--test-ratio 或 -te`**: (可選) 測試集比例，默認值為 1
*   **`--valid-ratio 或 -v`**: (可選) 驗證集比例，默認值為 2
*   **`--output 或 -o`**: **(必需)** 採樣數據集的輸出目錄

腳本將：

1. 從原始數據集中隨機選取指定數量的圖像和對應的標籤文件
2. 按指定比例分配到訓練、測試和驗證集
3. 在輸出目錄中創建相同的目錄結構
4. 創建更新後的 data.yaml 文件以供訓練使用

### 3. 執行訓練與導出

完成後，開始執行訓練模型與導出，在終端機中執行以下命令：

在第一次運行, ultralytics 會檢查並安裝缺失的 dependency

*   **執行以下命令以訓練模型與導出：**
    ```bash
    python train_export_model.py \
        --data_yaml /content/AI-Train-yolov8-roboflow/sampler/data.yaml \  
        --base_model yolov8m.pt \
        --epochs 2 \
        --img_size 640 \
        --project_dir my_training \
        --experiment_name fruit_detector \
        --export_format tfjs
    ```

**參數說明：**

*   **`--data_yaml`**: **(必需)** 將此變數的值更新為您在上一步數據集中找到的 `data.yaml` 檔案的完整路徑。
*   **`--base_model`**: (可選) 預設為 `"yolov8m.pt"`。您可以更改為其他 YOLOv8 模型，如 `"yolov8s.pt"` 或 `"yolov8l.pt"`。
*   **`--epochs`**: (可選) 訓練的週期數，預設為 `20`。
*   **`--img_size`**: (可選) 訓練和導出時的圖像大小，預設為 `640`。
*   **`--project_dir my_training`**: (可選) 儲存訓練結果的父目錄名稱，預設為 `"training_runs"`。
*   **`--experiment_name`**: (可選) 特定訓練運行的子目錄名稱，預設為 `"orange_detection_exp"`。
*   **`--export_format`**: (可選) 導出的模型格式，預設為 `"tfjs"`。其他選項包括 `'onnx'`, `'torchscript'`, `'tflite'` 等。