#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Check if conda command is available
if ! type conda > /dev/null; then
  echo "Conda is not available. Please ensure Conda is installed and initialized."
  exit 1
fi

# Check for the minimum number of arguments
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <dataset-type> [radar-config-file]"
  exit 1
fi

# Set dataset type from the first argument
DATASET_TYPE=$1

# Set radar config file path; use default if not provided
RADAR_CONFIG="${2:-$DEFAULT_CONFIG}"

# Choose the appropriate configuration file based on the dataset type
if [ "$DATASET_TYPE" = "mini" ]; then
  CONFIG_FILE="vis4d/zoo/cc_3dt/cc_3dt_pp_kf3d_nusc_mini.py"
  WORK_DIR="vis4d-workspace/cc_3dt_pp_kf3d_nusc_mini"
elif [ "$DATASET_TYPE" = "trainval" ]; then
  CONFIG_FILE="vis4d/zoo/cc_3dt/cc_3dt_pp_kf3d_nusc.py"
  WORK_DIR="vis4d-workspace/cc_3dt_pp_kf3d_nusc"
elif [ "$DATASET_TYPE" = "test" ]; then
  CONFIG_FILE="vis4d/zoo/cc_3dt/cc_3dt_pp_kf3d_nusc_test.py"
  WORK_DIR="vis4d-workspace/cc_3dt_pp_kf3d_nusc_test"
else
  echo "Invalid dataset type. Please choose from 'mini', 'trainval', or 'test'."
  exit 1
fi

# Activate the cc3dt Conda environment
source activate cc3dt

# Execute the vis4d.pl test command and extract the output directory
python -m vis4d.pl test --config "$CONFIG_FILE" --gpus 1 --ckpt qd_models/cc_3dt_frcnn_r101_fpn_24e_nusc_f24f84.pt --config.pure_detection "$RADAR_CONFIG"

# Get the newest directory
cd "$WORK_DIR" || exit
NEWEST_DIR=$(ls -td -- */ | head -n 1 | cut -d'/' -f1)

# Check if the NEWEST_DIR variable is set, if not exit the script
if [ -z "$NEWEST_DIR" ]; then
  echo "Newest directory could not be determined."
  exit 1
fi

# Construct the full path to the newest directory
FULL_PATH="$WORK_DIR/$NEWEST_DIR"

# Navigate back to the initial directory
cd /root/cc3dt || exit

echo $FULL_PATH

# Execute the eval_nusc.sh script with the dynamically determined timestamp and dataset type
bash eval_nusc.sh "${FULL_PATH}" "$DATASET_TYPE"