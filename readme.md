# YOLOv8 物件偵測模型訓練與 TF.js 導出

本專案展示如何使用 Ultralytics YOLOv8 訓練一個自訂物件偵測模型，並將其導出為 TensorFlow.js (TF.js) 格式，以便在網頁應用程式中部署。

## 功能

*   使用 YOLOv8 進行模型訓練。
*   將訓練好的模型導出為多種格式，主要目標為 TF.js。
*   透過編輯 Python 腳本頂部的變數進行配置。

## 先決條件

在開始之前，請確保您的系統已安裝以下軟體：

1.  **Python** (建議版本 3.8 或更高版本)
2.  **pip** (Python 套件安裝程式)
3.  **curl** (用於從 URL 下載檔案的命令列工具)
4.  **unzip** (用於解壓縮 ZIP 檔案的命令列工具)

    *   在 Debian/Ubuntu 上: `sudo apt update && sudo apt install curl unzip`
    *   在 macOS 上 (通常已內建，或使用 Homebrew): `brew install curl unzip`
    *   在 Windows 上: 可以使用 Git Bash (通常包含 curl 和 unzip)，或 Windows Subsystem for Linux (WSL)。

## 設定步驟

1.  **複製 (或下載) 專案檔案**:
    如果您是從 Git 儲存庫取得此專案：
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
    否則，請確保您已將 `train_export_model.py` 和 `requirements.txt` 檔案放置在您的工作目錄中。

2.  **安裝 Python 依賴套件**:
    在您的專案目錄中開啟終端機，然後執行：
    ```bash
    pip install -r requirements.txt
    ```
    如果您在導出為 TF.js 格式時遇到問題，您可能需要明確安裝 `tensorflowjs`：
    ```bash
    pip install tensorflowjs
    ```

## 工作流程

### 1. 下載數據集

本專案使用 `curl` 命令直接從 Roboflow Universe 下載預先準備好的數據集。

*   **執行以下命令以下載並解壓縮數據集：**
    ```bash
    curl -L "https://universe.roboflow.com/ds/Az7Nm6mc97?key=Gcijbs26Gz" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
    ```
    *   **注意**: 此命令會下載一個名為 `roboflow.zip` 的檔案，然後解壓縮它，最後刪除 `roboflow.zip`。
    *   解壓縮後，通常會產生一個包含數據集檔案的資料夾 (例如，資料夾名稱可能為 `test-orange-2` 或類似名稱，這取決於 Roboflow 上的專案設定)。
    *   **重要**: 進入解壓縮後的資料夾，找到 `data.yaml` 檔案。您需要**複製此 `data.yaml` 檔案的完整路徑**，以用於下一步的配置。

    例如，如果命令在 `/home/user/my_project` 中執行，並且 `roboflow.zip` 解壓縮到一個名為 `test-orange-2` 的資料夾，則 `data.yaml` 的路徑可能是 `/home/user/my_project/test-orange-2/data.yaml`。

### 2. 配置訓練腳本

打開 `train_export_model.py` 檔案，並修改頂部的「用戶配置」部分：

*   **`DATA_YAML_PATH`**: **(必需)** 將此變數的值更新為您在上一步中找到的 `data.yaml` 檔案的完整路徑。
    ```python
    # 範例 (Linux/MacOS):
    DATA_YAML_PATH = "/path/to/your/dataset_folder/data.yaml"
    # 範例 (Windows):
    # DATA_YAML_PATH = r"C:\path\to\your\dataset_folder\data.yaml"
    ```

*   **`BASE_MODEL_PT`**: (可選) 預設為 `"yolov8m.pt"`。您可以更改為其他 YOLOv8 模型，如 `"yolov8s.pt"` 或 `"yolov8l.pt"`。
*   **`EPOCHS`**: (可選) 訓練的週期數，預設為 `20`。
*   **`IMG_SIZE`**: (可選) 訓練和導出時的圖像大小，預設為 `640`。
*   **`PROJECT_DIR_NAME`**: (可選) 儲存訓練結果的父目錄名稱，預設為 `"training_runs"`。
*   **`EXPERIMENT_NAME`**: (可選) 特定訓練運行的子目錄名稱，預設為 `"orange_detection_exp"`。
*   **`EXPORT_FORMAT`**: (可選) 導出的模型格式，預設為 `"tfjs"`。其他選項包括 `'onnx'`, `'torchscript'`, `'tflite'` 等。

### 3. 執行訓練與導出

配置完成後，在終端機中執行以下命令：

```bash
python train_export_model.py
