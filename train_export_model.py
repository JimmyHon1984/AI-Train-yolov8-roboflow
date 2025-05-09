# train_export_model.py
import os
from ultralytics import YOLO
from pathlib import Path # 使用 pathlib 處理路徑更方便

# --- 用戶配置 ---
# 請在此處修改以下參數以適合您的需求。
# ==============================================================================
# 1. 數據集 YAML 檔案的路徑:
#    !! 重要 !! 
#    執行 download_data.py 後，該腳本會輸出 data.yaml 的完整路徑。
#    請將該路徑複製並粘貼到下面的 DATA_YAML_PATH 變數中。
#    確保路徑是正確的，並且檔案存在。
#    範例 (Windows): DATA_YAML_PATH = r"C:\Users\YourUser\project\datasets\test-orange-2\data.yaml"
#    範例 (Linux/MacOS): DATA_YAML_PATH = "/home/YourUser/project/datasets/test-orange-2/data.yaml"
DATA_YAML_PATH = ""  # <--- 請務必更新此路徑！例如: r"/content/datasets/test-orange-2/data.yaml"

# 2. 模型和訓練參數:
BASE_MODEL_PT = "yolov8m.pt"    # 用於開始訓練的預訓練模型 (.pt 檔案)
EPOCHS = 20                     # 訓練的 epoch 數量
IMG_SIZE = 640                  # 訓練和導出時的圖像大小 (像素)

# 3. 輸出目錄和實驗名稱:
#    所有訓練運行將儲存在 'PROJECT_DIR_NAME' 下的 'EXPERIMENT_NAME' 子目錄中。
#    例如，如果 PROJECT_DIR_NAME="training_runs" 且 EXPERIMENT_NAME="orange_v1",
#    則結果會儲存在 "training_runs/orange_v1/"
PROJECT_DIR_NAME = "training_runs"
EXPERIMENT_NAME = "orange_detection_exp" # 您可以修改此名稱以區分不同的訓練運行

# 4. 導出格式:
EXPORT_FORMAT = "tfjs"          # 例如 'tfjs', 'onnx', 'torchscript', 'tflite'
# ==============================================================================
# --- 配置結束 ---

def train_and_export_yolo_model(
    data_yaml_path_str, 
    base_model_name, 
    num_epochs, 
    image_size, 
    project_dir, 
    experiment_name, 
    export_format
):
    """
    訓練 YOLOv8 模型並將其導出為指定格式。
    """
    data_yaml_path = Path(data_yaml_path_str)
    if not data_yaml_path_str or not data_yaml_path.exists():
        print(f"[錯誤] 數據集設定檔 (data.yaml) 未找到或未設定！")
        print(f"  檢查的路徑: {data_yaml_path.resolve() if data_yaml_path_str else '未設定'}")
        print(f"  請確保 'train_export_model.py' 腳本頂部的 'DATA_YAML_PATH' 已正確設定。")
        print(f"  您應該先執行 'download_data.py' 並將其輸出的 data.yaml 路徑複製到此處。")
        return

    # 1. 載入一個基礎 YOLOv8 模型
    print(f"載入基礎模型: {base_model_name}...")
    model = YOLO(base_model_name)

    # 2. 訓練模型
    print(f"開始訓練模型...")
    print(f"  數據集: {data_yaml_path.resolve()}")
    print(f"  Epochs: {num_epochs}, 圖像大小: {image_size}x{image_size}")
    print(f"  訓練結果將儲存在: {Path(project_dir) / experiment_name}")
    
    try:
        results = model.train(
            data=str(data_yaml_path), # model.train 需要字串路徑
            epochs=num_epochs,
            imgsz=image_size,
            project=project_dir,      # 父目錄
            name=experiment_name,     # 特定實驗的子目錄
            exist_ok=True             # 如果實驗目錄已存在，則覆蓋 (YOLO 會處理，如果為 False 則會創建 exp2, exp3 等)
        )
    except Exception as e:
        print(f"[錯誤] 模型訓練過程中發生錯誤: {e}")
        return

    # results.save_dir 是訓練運行儲存的目錄
    run_dir = Path(results.save_dir) 
    print(f"[成功] 模型訓練完成。結果儲存在: {run_dir.resolve()}")

    # 最佳訓練權重的路徑
    best_model_path = run_dir / "weights" / "best.pt"
    
    if not best_model_path.exists():
        print(f"[錯誤] 找不到訓練後的最佳模型權重檔案 '{best_model_path}'。")
        print(f"  請檢查訓練日誌於 '{run_dir.resolve()}' 以了解詳細資訊。")
        return

    print(f"找到最佳模型權重: {best_model_path.resolve()}")

    # 3. 載入訓練好的最佳模型以進行導出
    print(f"載入訓練好的模型 '{best_model_path.resolve()}' 以進行導出...")
    trained_model = YOLO(best_model_path) # 從 .pt 檔案載入模型

    # 4. 導出模型
    print(f"正在將模型導出為 {export_format} 格式...")
    try:
        # model.export() 返回導出的檔案/資料夾的路徑
        exported_path_str = trained_model.export(format=export_format, imgsz=image_size) 
        
        exported_location = Path(exported_path_str)

        print(f"[成功] 模型已成功導出為 {export_format} 格式。")
        print(f"  導出的模型位於: {exported_location.resolve()}")

    except Exception as e:
        print(f"[錯誤] 導出模型時發生錯誤: {e}")
        print(f"  請確保已安裝導出 {export_format} 格式所需的依賴項。")
        if export_format == "tfjs":
            print("  對於 TF.js 導出，您可能需要執行: pip install tensorflowjs")
        elif export_format == "tflite":
            print("  對於 TFLite 導出，您可能需要執行: pip install tensorflow")


if __name__ == "__main__":
    print("="*50)
    print("YOLOv8 模型訓練與導出腳本")
    print("="*50)

    if not DATA_YAML_PATH:
        print("\n[警告] 'DATA_YAML_PATH' 尚未在腳本中設定！")
        print("請打開 'train_export_model.py' 並在頂部的配置區域中設定 'DATA_YAML_PATH'。")
        print("該路徑應為執行 'download_data.py' 後輸出的 data.yaml 檔案的完整路徑。")
        print("腳本無法繼續，直到此路徑被設定。\n")
    else:
        print(f"將使用以下配置進行訓練和導出：")
        print(f"  數據集 YAML: {DATA_YAML_PATH}")
        print(f"  基礎模型: {BASE_MODEL_PT}")
        print(f"  Epochs: {EPOCHS}")
        print(f"  圖像大小: {IMG_SIZE}")
        print(f"  專案目錄: {PROJECT_DIR_NAME}")
        print(f"  實驗名稱: {EXPERIMENT_NAME}")
        print(f"  導出格式: {EXPORT_FORMAT}\n")

        train_and_export_yolo_model(
            DATA_YAML_PATH,
            BASE_MODEL_PT,
            EPOCHS,
            IMG_SIZE,
            PROJECT_DIR_NAME,
            EXPERIMENT_NAME,
            EXPORT_FORMAT
        )
    print("\n訓練和導出流程已執行。")
    print("="*50)
