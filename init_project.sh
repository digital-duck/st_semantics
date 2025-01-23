#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Initializing Multilingual Embedding Explorer project...${NC}"

# Create project directory structure
mkdir -p utils models images
touch utils/__init__.py models/__init__.py

# Create empty input files
touch chn.txt enu.txt

# Create .env file
echo "# Your Hugging Face API key
HF_API_KEY=" > .env

# Create .gitignore
echo ".env
__pycache__/
*.pyc
.DS_Store
images/
*.log
.venv/
" > .gitignore

# Function to create file if it doesn't exist
create_file() {
    if [ ! -f "$1" ]; then
        echo -e "${GREEN}Creating $1...${NC}"
        touch "$1"
    else
        echo -e "${BLUE}$1 already exists, skipping...${NC}"
    fi
}

# Create main application files
create_file "app.py"
create_file "config.py"
create_file "utils/error_handling.py"
create_file "models/model_manager.py"
create_file "requirements.txt"
create_file "README.md"

# Set up virtual environment
echo -e "${BLUE}Setting up virtual environment...${NC}"
python -m venv .venv

# Activate virtual environment based on OS
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source .venv/bin/activate
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
fi

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
pip install -r requirements.txt

# Download LASER models
echo -e "${BLUE}Downloading LASER models...${NC}"
python -m laserembeddings download-models

echo -e "${GREEN}Project initialization complete!${NC}"
echo -e "${BLUE}Next steps:${NC}"
echo "1. Add your Hugging Face API key to .env"
echo "2. Install Ollama from https://ollama.ai (if using local models)"
echo "3. Pull Ollama models: ollama pull snowflake-arctic-embed2"
echo "4. Start the app: streamlit run app.py"