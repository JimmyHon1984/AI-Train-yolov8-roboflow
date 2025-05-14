# train_export_model.py
import os
import argparse
from ultralytics import YOLO
from pathlib import Path # 使用 pathlib 處理路徑更方便

def parse_arguments():
    """解析命令列參數以設定 YOLOv8 訓練和導出參數。"""
    parser = argparse.ArgumentParser(
        description="訓練並導出自定義 YOLOv8 模型。"
    )
    
    # 參數設定
    parser.add_argument(
        "--data_yaml", 
        help="數據集 YAML 文件的路徑。"
    )
    
    parser.add_argument(
        "--base_model", 
        default="yolov8m.pt",
        help="預訓練模型文件。預設值: yolov8m.pt"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=20,
        help="訓練的 epoch 數量。預設值: 20"
    )
    
    parser.add_argument(
        "--img_size", 
        type=int, 
        default=640,
        help="訓練和導出的圖像大小（像素）。預設值: 640"
    )
    
    parser.add_argument(
        "--project_dir", 
        default="training_runs",
        help="保存訓練結果的目錄。預設值: training_runs"
    )
    
    parser.add_argument(
        "--experiment_name", 
        default="orange_detection_exp",
        help="此次訓練實驗的名稱。預設值: orange_detection_exp"
    )
    
    parser.add_argument(
        "--export_format", 
        default="tfjs",
        help="導出模型的格式。選項: tfjs, onnx, torchscript, tflite。預設值: tfjs"
    )
    
    return parser.parse_args()

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
        print(f"  請確保提供了正確的 data.yaml 檔案路徑。")
        print(f"  您應該先按照 README.md 中的指示使用 curl 命令下載並解壓縮數據集，")
        print(f"  然後將解壓縮後資料夾中 data.yaml 的完整路徑提供給腳本。")
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

    # 解析命令列參數
    args = parse_arguments()
    
    # 如果命令列參數中沒有指定數據集路徑，則檢查原始腳本中的設定
    data_yaml_path = args.data_yaml or ""
    
    if not data_yaml_path:
        print("\n[警告] 數據集 YAML 路徑未設定！")
        print("請使用 --data_yaml 參數提供 data.yaml 檔案的路徑。")
        print("例如: python train_export_model.py --data_yaml /path/to/your/data.yaml")
        print("請參考 README.md 中的數據下載步驟。")
        print("腳本無法繼續，直到此路徑被設定。\n")
    else:
        print(f"將使用以下配置進行訓練和導出：")
        print(f"  數據集 YAML: {data_yaml_path}")
        print(f"  基礎模型: {args.base_model}")
        print(f"  Epochs: {args.epochs}")
        print(f"  圖像大小: {args.img_size}")
        print(f"  專案目錄: {args.project_dir}")
        print(f"  實驗名稱: {args.experiment_name}")
        print(f"  導出格式: {args.export_format}\n")

        train_and_export_yolo_model(
            data_yaml_path,
            args.base_model,
            args.epochs,
            args.img_size,
            args.project_dir,
            args.experiment_name,
            args.export_format
        )
    print("\n訓練和導出流程已執行。")
    print("="*50)