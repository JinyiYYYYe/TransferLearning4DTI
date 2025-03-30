#!/bin/bash
# Run the simualation for all possible combinations. Jinyi Ye.

# Define variables
SS=500  # Sample size
EPOCH=50  # Number of epochs
SOURCE_DATASETS=(gpcr transporter kinase nuclearreceptor protease ionchannel)
TARGET_DATASETS=(gpcr transporter kinase nuclearreceptor protease ionchannel)

# Step 1: Generate subdatasets
for DATASET in "${SOURCE_DATASETS[@]}"; do
    python  create_small_dataset.py --d "$DATASET" --ss $SS
done

# Step 2: Run experiments for each source-target pair (excluding identical pairs)
for TARGET in "${TARGET_DATASETS[@]}"; do
  echo "Running experiments for $TARGET"
  # Obtain scratch performance results
  python3 baseline_training.py --setting 2 --tlf 0 --td "$TARGET" --ss $SS --en 0 --sf 1
  python3 main_training.py --setting 3 --epoch $EPOCH --ss $SS --en 0 --tlf 0 --sf 1 --td "$TARGET"
  for SOURCE in "${SOURCE_DATASETS[@]}"; do
        echo "Running experiments from $SOURCE to $TARGET"
        if [ "$SOURCE" != "$TARGET" ]; then
            # Extract hidden layer output
            python3 main_training.py --setting 5 --train 0 --epoch $EPOCH --ss $SS --en 0 --el 1 --tlf 1 --sf 1 --sd "$SOURCE" --td "$TARGET"

            # Obtain shallow classifier performance results
            python3 baseline_training.py --setting 2 --tlf 1 --el 1 --sd "$SOURCE" --td "$TARGET" --ss $SS --en 0 --sf 1

            # Obtain full fine-tuning performance results
            python3 main_training.py --setting 3 --epoch $EPOCH --ss $SS --en 0 --tlf 1 --sf 1 --sd "$SOURCE" --td "$TARGET"

            # Obtain fine-tuning with freezing layer 1 performance results
            python3 main_training.py --setting 3 --epoch $EPOCH --ss $SS --en 0 --ff 1 --fl 1 --tlf 1 --sf 1 --sd "$SOURCE" --td "$TARGET"
        fi
    done
done
