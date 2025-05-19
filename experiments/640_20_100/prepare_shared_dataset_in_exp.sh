#!/bin/bash

# --- 路徑 ---
SCRIPT_DIR_EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
EXPERIMENT_DIR="$SCRIPT_DIR_EXP"
AI_PROJECT_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)" # 這應該解析為 AI-Train-yolov8-roboflow 的絕對路徑

# --- 配置 ---
# 源數據集根目錄 (相對於 AI_PROJECT_ROOT)
# 假設 annot-1 文件夾直接位於 AI_PROJECT_ROOT 下
DOWNLOADED_DATA_ROOT="${AI_PROJECT_ROOT}/annot-1"

# 共享數據集採樣配置
TOTAL_IMAGES_TO_SAMPLE=500 # 您希望新的共享數據集包含的圖片總數
TRAIN_RATIO=7  # 訓練集比例部分
VALID_RATIO=2  # 驗證集比例部分
TEST_RATIO=1   # 測試集比例部分

# 數據採樣器腳本的絕對路徑
SAMPLER_SCRIPT_PATH="${AI_PROJECT_ROOT}/src/dataset_sampler.py"

# 共享數據集的輸出目錄 (在當前實驗目錄下)
SHARED_SAMPLER_OUTPUT_DIR="${EXPERIMENT_DIR}/shared_dataset_output"
SHARED_DATA_YAML_PATH="${SHARED_SAMPLER_OUTPUT_DIR}/data.yaml"

echo "================================================================================"
echo "準備共享數據集 (在實驗目錄內)"
echo "實驗目錄 (EXPERIMENT_DIR): ${EXPERIMENT_DIR}"
echo "AI 項目根目錄 (AI_PROJECT_ROOT): ${AI_PROJECT_ROOT}"
echo "源數據集根目錄 (DOWNLOADED_DATA_ROOT): ${DOWNLOADED_DATA_ROOT}"
echo "採樣器腳本路徑 (SAMPLER_SCRIPT_PATH): ${SAMPLER_SCRIPT_PATH}"
echo "總採樣圖片數: ${TOTAL_IMAGES_TO_SAMPLE}"
echo "採樣比例 (Train/Valid/Test): ${TRAIN_RATIO}/${VALID_RATIO}/${TEST_RATIO}"
echo "採樣輸出目錄 (SHARED_SAMPLER_OUTPUT_DIR): ${SHARED_SAMPLER_OUTPUT_DIR}"
echo "================================================================================"

# 步驟 1: 檢查輸入
# --------------------------------------------------------------------------------
if [ ! -d "$DOWNLOADED_DATA_ROOT" ]; then
    echo "錯誤: 源數據集根目錄 '$DOWNLOADED_DATA_ROOT' 未找到。"
    echo "請確保 DOWNLOADED_DATA_ROOT 變量已正確設置，並且該目錄存在。"
    echo "預期路徑: ${AI_PROJECT_ROOT}/annot-1"
    exit 1
fi

if [ ! -f "${DOWNLOADED_DATA_ROOT}/data.yaml" ]; then
    echo "錯誤: 源數據集根目錄 '$DOWNLOADED_DATA_ROOT' 中未找到 'data.yaml' 文件。"
    echo "請確保源數據集結構正確。"
    exit 1
fi

for split_type in train test valid; do
    if [ ! -d "${DOWNLOADED_DATA_ROOT}/${split_type}/images" ]; then
        echo "錯誤: 源數據集根目錄 '$DOWNLOADED_DATA_ROOT' 中未找到 '${split_type}/images' 子目錄。"
        echo "請確保源數據集結構正確 (包含 train/images, test/images, valid/images)。"
        exit 1
    fi
    if [ ! -d "${DOWNLOADED_DATA_ROOT}/${split_type}/labels" ]; then
        echo "錯誤: 源數據集根目錄 '$DOWNLOADED_DATA_ROOT' 中未找到 '${split_type}/labels' 子目錄。"
        echo "請確保源數據集結構正確 (包含 train/labels, test/labels, valid/labels)。"
        exit 1
    fi
done

if [ ! -f "$SAMPLER_SCRIPT_PATH" ]; then
    echo "錯誤: 數據採樣器腳本未在 '$SAMPLER_SCRIPT_PATH' 找到。"
    exit 1
fi
echo "--------------------------------------------------------------------------------"

# 步驟 2: 運行數據採樣器
# --------------------------------------------------------------------------------
# 清理舊的採樣數據（如果存在）
if [ -d "$SHARED_SAMPLER_OUTPUT_DIR" ]; then
    echo "清理已存在的共享數據集目錄: $SHARED_SAMPLER_OUTPUT_DIR"
    rm -rf "$SHARED_SAMPLER_OUTPUT_DIR"
fi
mkdir -p "$SHARED_SAMPLER_OUTPUT_DIR" # 確保輸出目錄存在

echo "運行採樣器..."
(
  export PYTHONPATH="${AI_PROJECT_ROOT}:${PYTHONPATH}"

  python3 "$SAMPLER_SCRIPT_PATH" \
      --src "$DOWNLOADED_DATA_ROOT" \
      --total "$TOTAL_IMAGES_TO_SAMPLE" \
      --train-ratio "$TRAIN_RATIO" \
      --valid-ratio "$VALID_RATIO" \
      --test-ratio "$TEST_RATIO" \
      --output "$SHARED_SAMPLER_OUTPUT_DIR"
)

# 檢查 data.yaml 是否生成
if [ ! -f "$SHARED_DATA_YAML_PATH" ]; then
    echo "錯誤: data.yaml 未在 $SHARED_SAMPLER_OUTPUT_DIR 中生成。採樣失敗。"
    echo "請檢查採樣器腳本的輸出和錯誤信息。"
    exit 1
else
    echo "採樣器腳本執行成功。"
fi

echo "共享數據集準備完成。data.yaml 位於: $SHARED_DATA_YAML_PATH"
echo "================================================================================"
