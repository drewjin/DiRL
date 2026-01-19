#!/bin/bash

# Set your Hugging Face credentials
export HF_TOKEN="your_token_here"
export HF_USERNAME="your_username_here"
export HF_ENDPOINT=https://hf-mirror.com

# Get the absolute path of the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HFD_PATH="$SCRIPT_DIR/data/hfd.sh"

# Make hfd.sh executable
chmod +x "$HFD_PATH"

# Download evaluation datasets
mkdir -p "$SCRIPT_DIR/data"
cd "$SCRIPT_DIR/data"

"$HFD_PATH" Auraithm/AIME2024 --dataset --tool wget --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv AIME2024/AIME2024.json ./ 2>/dev/null || true
"$HFD_PATH" Auraithm/AIME2025 --dataset --tool wget --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv AIME2025/AIME2025.json ./ 2>/dev/null || true
"$HFD_PATH" Auraithm/MATH500 --dataset --tool wget --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv MATH500/MATH500.json ./ 2>/dev/null || true
"$HFD_PATH" Auraithm/GSM8K --dataset --tool wget --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv GSM8K/GSM8K.json ./ 2>/dev/null || true
"$HFD_PATH" Auraithm/OlympiadBench --dataset --tool wget --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv OlympiadBench/OlympiadBench.json ./ 2>/dev/null || true

# Download training datasets
"$HFD_PATH" Auraithm/GLM4.6-OpenR1Math-SFT --dataset --tool wget --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv GLM4.6-OpenR1Math-SFT/GLM4.6-OpenR1Math-SFT.json ./ 2>/dev/null || true
"$HFD_PATH" Auraithm/BigMath-RL --dataset --tool wget --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv BigMath-RL/BigMath-RL.json ./ 2>/dev/null || true
"$HFD_PATH" Auraithm/Light-OpenR1Math-SFT --dataset --tool wget --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv Light-OpenR1Math-SFT/Light-OpenR1Math-SFT.json ./ 2>/dev/null || true
"$HFD_PATH" Auraithm/Light-MATH-RL --dataset --tool wget --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv Light-MATH-RL/Light-MATH-RL.json ./ 2>/dev/null || true

# Download models
mkdir -p "$SCRIPT_DIR/public"
cd "$SCRIPT_DIR/public"

"$HFD_PATH" JetLM/SDAR-8B-Chat --tool wget --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv JetLM/SDAR-8B-Chat ./ 2>/dev/null || true
rm -rf JetLM
"$HFD_PATH" OpenMOSS-Team/DiRL-8B-Instruct --tool wget --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv OpenMOSS-Team/DiRL-8B-Instruct ./ 2>/dev/null || true
rm -rf OpenMOSS-Team

cd "$SCRIPT_DIR"

echo "All datasets and models downloaded successfully!"
