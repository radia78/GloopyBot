# setup.sh
#!/bin/bash

echo "Updating and upgrading the packages"
sudo apt-get update

echo "Installing pip"
sudo apt install -y python3-pip

echo "Installing Llama-CPP"
sudo pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal

echo "Running setup script"
sudo pip3 install -r requirements.txt

echo "Downloading the LLM model from HuggingFace Hub"
sudo mkdir -p models/llm_model && mkdir -p models/emb_model
sudo huggingface-cli download radia/Qwen1.5-1.8B-Q4_K_M-GGUF --local-dir models/llm_model
sudo huggingface-cli download radia/snowflake-arctic-embed-l-Q4_K_M-GGUF --local-dir models/emb_model