#!/bin/bash

# --- 訓練配置 ---
BASE_MODEL_NAME="yolov8m.pt"
IMG_SIZE=640
EPOCHS=50
NUM_INSTANCES=8
NUM_GPUS_AVAILABLE=1 # <--- 重要：請將此值設置為您系統上可用於訓練的 GPU 數量
                     # 例如，如果您有 2 個 GPU (ID 0 和 1)，則設置為 2。
                     # 如果只有 1 個 GPU，則設置為 1 (所有任務將競爭同一個 GPU)。

# --- 路徑 ---
SCRIPT_DIR_EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
EXPERIMENT_DIR="$SCRIPT_DIR_EXP"
AI_PROJECT_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
BASE_MODEL_PATH="${AI_PROJECT_ROOT}/${BASE_MODEL_NAME}"
SHARED_DATA_YAML_PATH="${EXPERIMENT_DIR}/shared_dataset_output/data.yaml"
TRAINING_PROJECT_PARENT_DIR="${EXPERIMENT_DIR}/training_runs_output"
BASE_RUN_NAME_PREFIX="exp_m${BASE_MODEL_NAME%.*}_img${IMG_SIZE}_ep${EPOCHS}_shared_parallel" # 稍微修改前綴以示區別
TRAIN_COMMAND_OR_SCRIPT="yolo"

echo "================================================================================"
echo "準備啟動 $NUM_INSTANCES 個訓練任務 (並行執行)"
echo "基礎模型: $BASE_MODEL_PATH"
echo "圖片尺寸: $IMG_SIZE"
echo "訓練週期: $EPOCHS"
echo "共享數據集 YAML: $SHARED_DATA_YAML_PATH"
echo "訓練輸出目錄: $TRAINING_PROJECT_PARENT_DIR"
echo "用於分配的可用 GPU 數量: $NUM_GPUS_AVAILABLE"
echo "================================================================================"

if [ "$NUM_GPUS_AVAILABLE" -le 0 ]; then
    echo "警告: NUM_GPUS_AVAILABLE 設置為 $NUM_GPUS_AVAILABLE。所有任務將嘗試使用 device 0。"
    echo "如果這不是預期的，請修改 NUM_GPUS_AVAILABLE 為您系統上可用的 GPU 數量。"
    # 根據需要，可以將 NUM_GPUS_AVAILABLE 強制設為 1，或直接退出
    # NUM_GPUS_AVAILABLE=1
fi

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

# 循環啟動訓練任務 (並行執行)
for i in $(seq 1 $NUM_INSTANCES)
do
  UNIQUE_RUN_NAME="${BASE_RUN_NAME_PREFIX}_instance${i}"

  # 計算要使用的 GPU ID，循環使用可用的 GPU
  # (i-1) 是因為 seq 從 1 開始，而 GPU ID 通常從 0 開始
  # 如果 NUM_GPUS_AVAILABLE 為 1 (或更少，經過上面警告處理後)，DEVICE_ID 將始終為 0
  if [ "$NUM_GPUS_AVAILABLE" -gt 0 ]; then
      DEVICE_ID=$(( (i-1) % NUM_GPUS_AVAILABLE ))
  else
      DEVICE_ID="0" # 如果 NUM_GPUS_AVAILABLE 未正確設置或為0，則回退到 GPU 0
  fi

  echo "--------------------------------------------------"
  echo "啟動訓練實例 $i (共 $NUM_INSTANCES): 名稱='$UNIQUE_RUN_NAME', 分配到設備=$DEVICE_ID ..."
  echo "--------------------------------------------------"

  if [ "$TRAIN_COMMAND_OR_SCRIPT" == "yolo" ]; then
    ( # 使用子 shell 以便 PYTHONPATH 的設置是局部的，並將整個子 shell 背景化
      export PYTHONPATH="${AI_PROJECT_ROOT}:${PYTHONPATH}"
      yolo train \
          model="$BASE_MODEL_PATH" \
          data="$SHARED_DATA_YAML_PATH" \
          epochs="$EPOCHS" \
          imgsz="$IMG_SIZE" \
          project="$TRAINING_PROJECT_PARENT_DIR" \
          name="$UNIQUE_RUN_NAME" \
          device="$DEVICE_ID" \
          exist_ok=True # 確保不會因為目錄已存在而失敗 (YOLO 行為)
          # 其他YOLO參數
    ) & # <--- 將子 shell (以及其中的 yolo 命令) 放到背景執行
  else
    echo "自定義訓練腳本執行邏輯未完全實現 (並行模式)。"
    # (如果使用自定義腳本，請確保也使用 '&' 將其背景化)
  fi

  # 可選：如果您擔心同時啟動太多進程導致瞬間負載過高，可以稍微延遲
  # 例如，每啟動一個任務後等待1秒
  # sleep 1
done

echo "================================================================================"
echo "已啟動所有 $NUM_INSTANCES 個訓練任務。"
echo "等待所有背景訓練任務完成..."
wait # 等待所有由該 shell 啟動的背景子進程完成
echo "================================================================================"
echo "所有 $NUM_INSTANCES 個訓練任務已處理完畢 (無論成功或失敗)。"
echo "檢查 '${TRAINING_PROJECT_PARENT_DIR}' 目錄下每個實例的日誌以獲取結果。"
echo "================================================================================"
