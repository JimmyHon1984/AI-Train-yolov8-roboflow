#!/bin/bash

# --- 訓練配置 ---
BASE_MODEL_NAME="yolov8m.pt"
IMG_SIZE=640
EPOCHS=50
DATASET_ID_TAG="dataset1000" # 用於命名，表示這是針對特定數據集的運行
DEVICE_ID="0" # 指定使用的 GPU ID

# --- 路徑 ---
SCRIPT_DIR_EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
EXPERIMENT_DIR="$SCRIPT_DIR_EXP"
AI_PROJECT_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
BASE_MODEL_PATH="${AI_PROJECT_ROOT}/${BASE_MODEL_NAME}"

# !!! 重要：請將此路徑修改為您特定數據集的 data.yaml 文件路徑 !!!
# 例如：YOUR_DATASET_YAML_PATH="${AI_PROJECT_ROOT}/datasets/my_dataset_1000_samples/data.yaml"
# 或者：YOUR_DATASET_YAML_PATH="${EXPERIMENT_DIR}/specific_dataset_configs/dataset_1000.yaml"
YOUR_DATASET_YAML_PATH="/path/to/your/specific_dataset_for_1000_items/data.yaml" # <--- *** 請務必修改此處 ***

TRAINING_PROJECT_PARENT_DIR="${EXPERIMENT_DIR}/training_runs_output"
# 構建一個唯一的運行名稱
UNIQUE_RUN_NAME="exp_m${BASE_MODEL_NAME%.*}_img${IMG_SIZE}_ep${EPOCHS}_${DATASET_ID_TAG}"
TRAIN_COMMAND_OR_SCRIPT="yolo"

echo "================================================================================"
echo "準備啟動單一訓練任務"
echo "基礎模型: $BASE_MODEL_PATH"
echo "數據集 YAML: $YOUR_DATASET_YAML_PATH"
echo "圖片尺寸: $IMG_SIZE"
echo "訓練週期: $EPOCHS"
echo "設備 ID: $DEVICE_ID"
echo "訓練輸出父目錄: $TRAINING_PROJECT_PARENT_DIR"
echo "訓練運行名稱: $UNIQUE_RUN_NAME"
echo "================================================================================"

# 檢查基礎模型是否存在
if [ ! -f "$BASE_MODEL_PATH" ]; then
    echo "錯誤: 基礎模型 '$BASE_MODEL_PATH' 未找到。"
    exit 1
fi

# 檢查數據集 YAML 文件是否存在
if [ "$YOUR_DATASET_YAML_PATH" == "/path/to/your/specific_dataset_for_1000_items/data.yaml" ]; then
    echo "錯誤: 請修改腳本中的 'YOUR_DATASET_YAML_PATH' 變量，"
    echo "      使其指向您實際的 data.yaml 文件路徑。"
    exit 1
elif [ ! -f "$YOUR_DATASET_YAML_PATH" ]; then
    echo "錯誤: 數據集 YAML 文件 '$YOUR_DATASET_YAML_PATH' 未找到。"
    echo "請確保路徑正確，並且文件存在。"
    exit 1
fi

# 創建訓練輸出父目錄 (如果不存在)
mkdir -p "$TRAINING_PROJECT_PARENT_DIR"

echo "--------------------------------------------------"
echo "啟動訓練: 名稱='$UNIQUE_RUN_NAME', 設備=$DEVICE_ID ..."
echo "--------------------------------------------------"

if [ "$TRAIN_COMMAND_OR_SCRIPT" == "yolo" ]; then
  ( # 使用子 shell 以便 PYTHONPATH 的設置是局部的
    export PYTHONPATH="${AI_PROJECT_ROOT}:${PYTHONPATH}"
    yolo train \
        model="$BASE_MODEL_PATH" \
        data="$YOUR_DATASET_YAML_PATH" \
        epochs="$EPOCHS" \
        imgsz="$IMG_SIZE" \
        project="$TRAINING_PROJECT_PARENT_DIR" \
        name="$UNIQUE_RUN_NAME" \
        device="$DEVICE_ID" \
        exist_ok=True # 如果意外地重新運行同名實驗，允許覆蓋或繼續
        # 您可以在此處添加其他 YOLO 訓練參數，例如：
        # batch=16
        # workers=8
        # patience=10
        # lr0=0.001
  )
  # 檢查上一個命令的退出狀態
  if [ $? -ne 0 ]; then
      echo "錯誤: 訓練任務 '$UNIQUE_RUN_NAME' 失敗。"
      exit 1
  else
      echo "訓練任務 '$UNIQUE_RUN_NAME' 完成。"
  fi
else
  echo "錯誤: 此腳本目前僅配置為使用 'yolo' 命令。"
  exit 1
fi

echo "================================================================================"
echo "單一訓練任務已處理完畢。"
echo "檢查 '${TRAINING_PROJECT_PARENT_DIR}/${UNIQUE_RUN_NAME}' 目錄以獲取結果。"
echo "================================================================================"
