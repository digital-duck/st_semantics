#!/bin/bash

# Create multi-page structure
echo -e "${GREEN}Creating multi-page application structure...${NC}"
mkdir -p pages components services
touch pages/__init__.py components/__init__.py services/__init__.py

# Create main pages
echo -e "${GREEN}Creating application pages...${NC}"
touch "pages/1_🔤_semantics.py"
touch "pages/2_🌐_translator.py"
touch "pages/3_🤖_chatbot.py"
touch "pages/4_📝_notes.py"

# Create components
echo -e "${GREEN}Creating reusable components...${NC}"
touch components/embedding_viz.py
touch components/translation_widget.py
touch components/note_editor.py

# Create services
echo -e "${GREEN}Creating service integrations...${NC}"
touch services/google_translate.py
touch services/tts_service.py

# Create main app file
echo -e "${GREEN}Creating Home.py...${NC}"
touch Home.py
