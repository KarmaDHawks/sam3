#!/bin/bash
# -*- coding: utf-8 -*-
"""
Master inference script for all datasets.
Runs VOS, TREK-150, EgoExo4D, EgoTracks, and IT3DEgo inference sequentially.
Logs errors and generates a report of failed sequences.
"""

set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE="/home/marco/Desktop/SAM3-exp/sam3"
LOGS_DIR="${WORKSPACE}/inference_logs"
REPORT_FILE="${LOGS_DIR}/inference_report.txt"
ERROR_SUMMARY="${LOGS_DIR}/error_summary.txt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create logs directory
mkdir -p "${LOGS_DIR}"

# Initialize report files
cat > "${REPORT_FILE}" << EOF
=============================================================================
SAM3 INFERENCE EXECUTION REPORT
Generated: $(date)
=============================================================================

EOF

cat > "${ERROR_SUMMARY}" << EOF
=============================================================================
ERROR SUMMARY BY DATASET
Generated: $(date)
=============================================================================

EOF

# Function to log message
log_msg() {
    local level=$1
    shift
    local msg="$@"
    echo -e "[${level}] ${msg}" | tee -a "${REPORT_FILE}"
}

# Function to run inference script with error handling
run_inference() {
    local dataset_name=$1
    local script=$2
    shift 2
    local args=("$@")

    log_msg "${BLUE}→${NC}" "Starting ${dataset_name} inference..."

    local log_file="${LOGS_DIR}/${dataset_name}_${TIMESTAMP}.log"
    local error_log="${LOGS_DIR}/${dataset_name}_errors_${TIMESTAMP}.txt"

    if python "${WORKSPACE}/${script}" "${args[@]}" \
        > >(tee -a "${log_file}") \
        2> >(tee -a "${error_log}" >&2); then
        log_msg "${GREEN}✓${NC}" "${dataset_name} inference completed successfully"
        echo "  Log: ${log_file}" >> "${REPORT_FILE}"
        
        # Check if error log has content (warnings/errors during execution)
        if [ -s "${error_log}" ]; then
            log_msg "${YELLOW}⚠${NC}" "${dataset_name} had some warnings/errors (see ${error_log})"
            echo "" >> "${ERROR_SUMMARY}"
            echo "--- ${dataset_name} WARNINGS/ERRORS ---" >> "${ERROR_SUMMARY}"
            tail -20 "${error_log}" >> "${ERROR_SUMMARY}"
        fi
    else
        local exit_code=$?
        log_msg "${RED}✗${NC}" "${dataset_name} inference FAILED (exit code: ${exit_code})"
        echo "  Error log: ${error_log}" >> "${REPORT_FILE}"
        
        echo "" >> "${ERROR_SUMMARY}"
        echo "--- ${dataset_name} FAILED (exit code: ${exit_code}) ---" >> "${ERROR_SUMMARY}"
        echo "Command: python ${WORKSPACE}/${script} ${args}" >> "${ERROR_SUMMARY}"
        echo "Errors:" >> "${ERROR_SUMMARY}"
        tail -50 "${error_log}" >> "${ERROR_SUMMARY}"
        
        # Continue to next dataset instead of exiting
        log_msg "${YELLOW}!${NC}" "Continuing to next dataset despite error..."
    fi
    
    echo "" >> "${REPORT_FILE}"
}

# ============================================================================
# INFERENCE CONFIGURATIONS
# ============================================================================

# Output directory for all masks
OUTPUT_BASE="/media/TBData/marco/Projects/SAMem/SAM3.1_base/masks"
mkdir -p "${OUTPUT_BASE}"

echo "=============================================================================
Starting SAM3 Inference Pipeline
Workspace: ${WORKSPACE}
Output Base: ${OUTPUT_BASE}
Logs Directory: ${LOGS_DIR}
Timestamp: ${TIMESTAMP}
=============================================================================" | tee -a "${REPORT_FILE}"

# ============================================================================
# 1. TREK-150 INFERENCE
# ============================================================================
log_msg "${BLUE}[1/5]${NC}" "TREK-150 Dataset"

TREK150_DIR="/media/TBDataNAS/Visual Object Tracking/TREK-150-annotations-w-imgs"  # UPDATE THIS
TREK150_OUTPUT="${OUTPUT_BASE}/TREK150v2"

if [ -d "${TREK150_DIR}" ]; then
    run_inference "TREK-150" \
        "tools/trek150_inference.py" \
        --trek150_dir "/media/TBDataNAS/Visual Object Tracking/TREK-150-annotations-w-imgs" \
        --output_mask_dir "/media/TBData/marco/Projects/SAMem/SAM3.1_base/masks/TREK150v2" \
        --offload_video_to_cpu 
        "${TREK150_OUTPUT}"
else
    log_msg "${YELLOW}⊘${NC}" "TREK-150 directory not found: ${TREK150_DIR} (skipping)"
    echo "" >> "${REPORT_FILE}"
fi

# ============================================================================
# 2. EGOEXO4D INFERENCE
# ============================================================================
log_msg "${BLUE}[2/5]${NC}" "EgoExo4D Dataset"

EGOEXO4D_FRAMES="/media/TBData4/data/EgoExo4d/frames/val"  # UPDATE THIS
EGOEXO4D_ANNO="/media/TBDataNAS/Egocentric Vision/EgoExo4D/v2/annotations/vot_ego_exo/sot/val"  # UPDATE THIS
EGOEXO4D_OUTPUT="${OUTPUT_BASE}/VISTA"

if [ -d "${EGOEXO4D_FRAMES}" ] && [ -d "${EGOEXO4D_ANNO}" ]; then
    run_inference "EgoExo4D" \
        "tools/egoexo_inference.py" \
        --frames_dir "${EGOEXO4D_FRAMES}"   \
        --anno_dir "${EGOEXO4D_ANNO}"   \
        --output_dir "${EGOEXO4D_OUTPUT}"   \
        --view "ego" 
        "${EGOEXO4D_OUTPUT}"
else
    log_msg "${YELLOW}⊘${NC}" "EgoExo4D directories not found (skipping)"
    echo "" >> "${REPORT_FILE}"
fi

# ============================================================================
# 3. EGOTRACKS INFERENCE
# ============================================================================
log_msg "${BLUE}[3/5]${NC}" "EgoTracks Dataset"

EGOTRACKS_FRAMES="/media/TBDataNAS/Egocentric Vision/Ego4D/v2/clips_frames_val/frames"
EGOTRACKS_ANNO="/home/zaira/Projects/sam2/TREK-150-toolkit/toolkit/datasets/egotracks-annotations"
EGOTRACKS_OUTPUT="${OUTPUT_BASE}/EgoTracks"

if [ -d "${EGOTRACKS_FRAMES}" ] && [ -d "${EGOTRACKS_ANNO}" ]; then
    run_inference "EgoTracks" \
        "tools/egotracks_inference.py" \
        --frames_dir "${EGOTRACKS_FRAMES}" \
        --anno_dir "${EGOTRACKS_ANNO}"  \
        --output_dir "${EGOTRACKS_OUTPUT}"  \
        --offload_video_to_cpu 
        "${EGOTRACKS_OUTPUT}"
else
    log_msg "${YELLOW}⊘${NC}" "EgoTracks directories not found (skipping)"
    echo "" >> "${REPORT_FILE}"
fi

# ============================================================================
# 4. IT3DEGO INFERENCE
# ============================================================================
log_msg "${BLUE}[4/5]${NC}" "IT3DEgo Dataset"

IT3DEGO_FRAMES="/media/TBDataNAS/Egocentric Vision/IT3DEgo/raw_videos"
IT3DEGO_ANNO="/media/TBDataNAS/Egocentric Vision/IT3DEgo/annotations"
IT3DEGO_OUTPUT="${OUTPUT_BASE}/IT3DEgo"

if [ -d "${IT3DEGO_FRAMES}" ] && [ -d "${IT3DEGO_ANNO}" ]; then
    run_inference "IT3DEgo" \
        "tools/it3dego_inference.py" \
        --frames_root "${IT3DEGO_FRAMES}"  \
        --ann_root "${IT3DEGO_ANNO}"    \
        --output_dir "${IT3DEGO_OUTPUT}"    \
        --offload_video_to_cpu 
        "${IT3DEGO_OUTPUT}"
else
    log_msg "${YELLOW}⊘${NC}" "IT3DEgo directories not found (skipping)"
    echo "" >> "${REPORT_FILE}"
fi

# ============================================================================
# FINAL REPORT
# ============================================================================

echo "" | tee -a "${REPORT_FILE}"
echo "=============================================================================" | tee -a "${REPORT_FILE}"
echo "INFERENCE PIPELINE COMPLETED" | tee -a "${REPORT_FILE}"
echo "Completion time: $(date)" | tee -a "${REPORT_FILE}"
echo "=============================================================================" | tee -a "${REPORT_FILE}"
echo "" | tee -a "${REPORT_FILE}"

echo "Output directory: ${OUTPUT_BASE}" | tee -a "${REPORT_FILE}"
echo "Logs directory: ${LOGS_DIR}" | tee -a "${REPORT_FILE}"
echo "Report: ${REPORT_FILE}" | tee -a "${REPORT_FILE}"
echo "Error Summary: ${ERROR_SUMMARY}" | tee -a "${REPORT_FILE}"
echo "" | tee -a "${REPORT_FILE}"

# Print error summary if there were any
if [ -s "${ERROR_SUMMARY}" ]; then
    echo "" | tee -a "${REPORT_FILE}"
    echo "=============================================================================" | tee -a "${REPORT_FILE}"
    cat "${ERROR_SUMMARY}" | tee -a "${REPORT_FILE}"
    echo "=============================================================================" | tee -a "${REPORT_FILE}"
fi

log_msg "${GREEN}✓${NC}" "Pipeline execution log saved to: ${REPORT_FILE}"