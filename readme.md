# YOLOv8 物件偵測模型訓練與匯出指南 (新版腳本流程)

本指南將引導您使用提供的腳本來訓練 YOLOv8 物件偵測模型，並將其轉換為 TensorFlow.js (TFJS) 格式供前端使用。

**重要提示：** 本 README 專注於 `experiments` 目錄下的腳本化實驗流程。請務必先閱讀位於專案根目錄 (`AI-Train-yolov8-roboflow/`) 下的 **主 `README.md` 文件**，以完成環境設定、依賴套件安裝 (`pip install -r requirements.txt`) 以及其他重要的初始配置。

## 目錄結構概覽

    ```
    AI-Train-yolov8-roboflow/
    ├── experiments/
    │   ├── collect_data.sh                       # 下載完整 Roboflow 數據集腳本
    │   ├── 640_20_100/                           # 實驗參數目錄 (圖片640, 20週期, 100樣本)
    │   │   ├── prepare_shared_dataset_in_exp_100.sh # 準備此實驗的抽樣數據腳本
    │   │   └── run_8_trains_in_exp.sh            # 執行此實驗的多個訓練任務腳本
    │   ├── 640_50_1000/                          # 實驗參數目錄 (圖片640, 50週期, 1000樣本)
    │   │   ├── prepare_shared_dataset_in_exp_1000.sh # 準備此實驗的抽樣數據腳本
    │   │   └── run_single_training_640_50_1000.sh  # 執行此實驗的單一訓練任務腳本 (範例名稱)
    │   ├── ...                                   # 其他實驗參數目錄
    │   └── dataset_roboflow_downloaded/          # (範例) 完整數據集下載位置
    ├── src/                                      # 核心 Python 原始碼 (例如 train_export_model.py)
    ├── yolov8n.pt                              # 基礎模型
    ├── yolov8s.pt                              # 基礎模型
    ├── yolov8m.pt                              # 基礎模型
    ├── README.md                               # 專案主 README
    └── requirements.txt                        # Python 依賴套件
    ```


## 初始設定

1.  **複製儲存庫 (Clone the Repository)**：
    ```bash
    git clone https://github.com/JimmyHon1984/AI-Train-yolov8-roboflow.git
    cd AI-Train-yolov8-roboflow
    ```

2.  **閱讀主 README.md 並安裝依賴**：
    *   請務必詳細閱讀位於專案根目錄下的 `README.md` 文件，並依照指示完成環境設定和依賴套件安裝。
    *   通常包括執行：
        ```bash
        pip install -r requirements.txt
        ```

## 數據準備流程

1.  **下載完整數據集 (使用 `collect_data.sh`)**：
    *   請確保在 `AI-Train-yolov8-roboflow` 專案根目錄：
        ```bash
        cd AI-Train-yolov8-roboflow
        ```
    *   執行 `collect_data.sh` 腳本以下載完整的訓練數據集。此腳本將處理原主 README 中「第 1 部分 下載數據集」的步驟。
        ```bash
        ./collect_data.sh
        ```
    *   **注意**：請確保數據集成功下載到預期位置 (例如 `AI-Train-yolov8-roboflow/annot-1/`)。下載的資料夾中應包含 `dataset.yaml` 文件。

2.  **為特定實驗準備抽樣數據集**：
    *   對於您計劃執行的每一個實驗組合（例如 `640_20_100`，`640_50_1000` 等），您需要進入其對應的子目錄並準備該實驗所需的抽樣數據。
    *   例如，若要準備 `640_20_100` 實驗的數據：
        ```bash
        cd experiments/640_20_100  
        ```
    *   執行該目錄下的數據準備腳本 (例如 `prepare_shared_dataset_in_exp_100.sh` 或類似名稱的腳本)：
        ```bash
        ./prepare_shared_dataset_in_exp_100.sh # 請替換為實際的腳本名稱
        ```
    *   此腳本會從步驟 1 下載的完整數據集中，按照該實驗設定的數量（例如 100 個樣本）和固定的 7:1:2 比例（訓練:驗證:測試集）複製或連結圖片和標籤到一個該實驗專用的數據集位置 (例如該實驗目錄下的 `shared_dataset_output/` 子目錄內，並生成對應的 `data.yaml`)。
    *   對每個實驗參數組合目錄（`640_20_500`、`640_50_100` 等）重複此步驟。返回上一層目錄可使用 `cd ..`。

## 模型訓練與匯出流程

1.  **執行訓練與匯出腳本**：
    *   進入您已準備好數據的特定實驗目錄。例如，繼續以 `640_20_100` 為例：
        ```bash
        # 假設您已在 experiments/640_20_100 目錄中
        # 如果不在，請 cd experiments/640_20_100
        ```
    *   執行該目錄下的訓練腳本 (例如 `run_8_trains_in_exp.sh` 或針對 `640_50_1000` 的單一訓練腳本)。

        *   **對於多個並行訓練 (例如 `run_8_trains_in_exp.sh`)**：
            此類型腳本通常設計為同時針對不同的基礎模型 (n, s, m) 或其他變體進行訓練。您可能需要編輯此腳本或通過參數傳遞來指定：
            *   **基礎模型 (Base_model)**：n, s, m。腳本應能處理這些不同模型的訓練。
            *   **實驗名稱 (Experiment_name)**：根據測試表格提供 (測試案例編號)。這通常會作為 YOLO 訓練指令中 `name` 參數的一部分。
            *   **圖片尺寸 (Img_size)** 和 **週期 (Epochs)** 通常由目錄名稱隱含（例如 `640_20_100` 表示 img_size=640, epochs=20）。腳本應能從其所在路徑或配置中獲取這些值。
            ```bash
            ./run_8_trains_in_exp.sh # 根據腳本設計，可能需要傳遞參數
            ```

        *   **對於單一訓練過程 (例如 `640_50_1000` 目錄下的腳本)**：
            如 `run_single_training_640_50_1000.sh` (範例名稱)。
            *   **修改腳本**：確保腳本內的 `BASE_MODEL_NAME` (例如 `yolov8m.pt`, `yolov8n.pt`, `yolov8s.pt`)、`EPOCHS` (50)、`IMG_SIZE` (640)、`YOUR_DATASET_YAML_PATH` (應指向此實驗目錄下由 `prepare_...` 腳本生成的 `data.yaml`) 和 `UNIQUE_RUN_NAME` (測試案例編號) 已正確設定。
            *   基礎模型 (如 `yolov8m.pt`) 應位於專案根目錄 (`AI-Train-yolov8-roboflow/yolov8m.pt`)。對於 `n` 和 `s` 模型，您需要確保對應的 `.pt` 文件也存在於該位置，並在腳本中指定正確的基礎模型文件名。
            ```bash
            ./run_single_training_640_50_1000.sh # 假設您將該腳本放置於此並配置好
            ```
    *   訓練腳本內部應調用 `src/train_export_model.py`（或類似的核心訓練與匯出邏輯），並自動處理模型的訓練、評估以及匯出為 `.pt`、`.onnx` 和 TFJS (`_web_model`) 格式。

2.  **記錄與收集結果**：
    *   訓練完成後，相關的輸出（包括模型檔案和效能指標）會自動儲存。
    *   **儲存位置**：通常在執行訓練腳本的實驗目錄下，YOLOv8 會創建一個類似 `training_runs_output/` 的輸出資料夾。
    *   **模型檔案**：
        *   TensorFlow.js 格式：`YOUR_EXPERIMENT_NAME_web_model` 資料夾。
        *   PyTorch 格式：`weights/best.pt` (或 `last.pt`)。
        *   ONNX 格式：`best.onnx` (或 `last.onnx`)。
    *   **效能指標**：
        *   mAP (mean Average Precision) 等指標通常記錄在輸出資料夾內的 `results.csv` 或其他日誌文件中。
        *   訓練時間通常可以從腳本的控制台輸出或日誌中獲取。
    *   請務必為每個測試案例記錄這些重要數據。

## 測試參數組合

請依照以下參數組合，重複上述「為特定實驗準備抽樣數據集」和「執行訓練與匯出腳本」的步驟：

*   **週期 (Epochs)**：20, 50
*   **圖片尺寸 (Img_size)**：640 (固定)
*   **基礎模型 (Base_model)**：yolov8m
*   **數據集樣本數**：根據您的實驗目錄設定（例如 100, 500, 1000）。

## 範例測試流程

以測試案例：Epochs=20, Img_size=640, Base_model=yolov8n, Dataset_samples=100 為例。

1.  確保您在 `AI-Train-yolov8-roboflow/` 目錄下。
2.  `./collect_data.sh` 
3.  `cd experiments`
4.  `cd 640_20_100`
5.  `./prepare_shared_dataset_in_exp_100.sh` 
7.  `./run_8_trains_in_exp.sh` 
8.  訓練完成後，檢查`training_runs_output/` 中模型訓練結果資料夾是否創建，並保存資料夾。
9.  對所有其他參數組合重複步驟 4-8 (根據需要調整目錄和腳本)。

## 重要提示

*   **輸出管理**：由於您會運行多次實驗，請確保每個實驗的輸出都保存在唯一的實驗名稱下，以便區分和比較結果。
*   **`640_50_1000` 特例**：此實驗（或類似的大型單一實驗）是單獨運行的，請使用為其設計的單一訓練腳本。
*   **腳本權限**：如果遇到 `Permission denied` 錯誤，請確保您的 `.sh` 腳本具有執行權限：`chmod +x your_script_name.sh`。

