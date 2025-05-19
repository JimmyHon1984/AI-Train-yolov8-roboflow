#!/bin/bash

# --- 訓練配置 ---
BASE_MODEL_NAME="yolov8m.pt"
IMG_SIZE=640
EPOCHS=20
NUM_INSTANCES=8 # 我們仍然可以定義8個不同的運行，但它們會順序執行
# NUM_GPUS=1 # 對於順序執行，這個變量意義不大，但device="0"是固定的

# --- 路徑 ---
SCRIPT_DIR_EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
EXPERIMENT_DIR="$SCRIPT_DIR_EXP"
AI_PROJECT_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
BASE_MODEL_PATH="${AI_PROJECT_ROOT}/${BASE_MODEL_NAME}"
SHARED_DATA_YAML_PATH="${EXPERIMENT_DIR}/shared_dataset_output/data.yaml"
TRAINING_PROJECT_PARENT_DIR="${EXPERIMENT_DIR}/training_runs_output"
BASE_RUN_NAME_PREFIX="exp_m${BASE_MODEL_NAME%.*}_img${IMG_SIZE}_ep${EPOCHS}_shared"
TRAIN_COMMAND_OR_SCRIPT="yolo"

echo "================================================================================"
echo "準備啟動 $NUM_INSTANCES 個訓練任務 (順序執行)"
# ... (其他 echo 保持不變) ...
echo "================================================================================"

if [ ! -f "$BASE_MODEL_PATH" ]; then
    echo "錯誤: 基礎模型 '$BASE_MODEL_PATH' 未找到。"
    exit 1
fi
if [ ! -f "$SHARED_DATA_YAML_PATH" ]; then
    echo "錯誤: 共享數據集 YAML 文件 '$SHARED_DATA_YAML_PATH' 未找到。"
    echo "請先運行數據集準備腳本 (prepare_shared_dataset_in_exp.sh)。"
    exit 1
fi
mkdir -p "$TRAINING_PROJECT_PARENT_DIR"

# 循環啟動訓練任務 (順序執行)
for i in $(seq 1 $NUM_INSTANCES)
do
  UNIQUE_RUN_NAME="${BASE_RUN_NAME_PREFIX}_instance${i}"
  DEVICE_ID="0" # 固定為 GPU 0

  echo "--------------------------------------------------"
  echo "啟動訓練實例 $i (共 $NUM_INSTANCES): 名稱='$UNIQUE_RUN_NAME', 設備=$DEVICE_ID ..."
  echo "--------------------------------------------------"

  if [ "$TRAIN_COMMAND_OR_SCRIPT" == "yolo" ]; then
    ( # 仍然使用子 shell 以便 PYTHONPATH 的設置是局部的
      export PYTHONPATH="${AI_PROJECT_ROOT}:${PYTHONPATH}"
      yolo train \
          model="$BASE_MODEL_PATH" \
          data="$SHARED_DATA_YAML_PATH" \
          epochs="$EPOCHS" \
          imgsz="$IMG_SIZE" \
          project="$TRAINING_PROJECT_PARENT_DIR" \
          name="$UNIQUE_RUN_NAME" \
          device="$DEVICE_ID" \
          exist_ok=True # 考慮是否真的需要，如果每次都是新實例名
          # 其他YOLO參數
    ) # 注意：這裡移除了 '&'
    # 檢查上一個命令的退出狀態
    if [ $? -ne 0 ]; then
        echo "錯誤: 訓練實例 $i ('$UNIQUE_RUN_NAME') 失敗。中止後續任務。"
        exit 1
    fi
  else
    echo "自定義訓練腳本執行邏輯未完全實現。"
    # (自定義腳本邏輯，同樣移除 '&')
  fi
  echo "訓練實例 $i ('$UNIQUE_RUN_NAME') 完成。"
done

echo "================================================================================"
echo "所有 $NUM_INSTANCES 個訓練任務已順序處理完畢。"
echo "檢查 '${TRAINING_PROJECT_PARENT_DIR}' 目錄以獲取結果。"
echo "================================================================================"
