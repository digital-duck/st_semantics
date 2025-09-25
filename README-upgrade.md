# Torch Security Upgrade Plan

**Objective**: Upgrade torch from 2.4.1+cu121 (zinets) and remove 2.5.1 (base) to torch ≥2.6.0 to address CVE-2025-32434 security vulnerability.

## 🚨 Security Context

**Vulnerability**: CVE-2025-32434 - Serious vulnerability in `torch.load` affecting versions < 2.6.0
**Error Message**: "Due to a serious vulnerability issue in torch.load, even with weights_only=True, we now require users to upgrade torch to at least v2.6"
**Reference**: https://nvd.nist.gov/vuln/detail/CVE-2025-32434

## 📋 Pre-Upgrade Assessment

### Current State Analysis:
```bash
# Check current environments
conda env list

# zinets environment (primary workspace)
conda activate zinets
pip show torch
# Expected: torch 2.4.1+cu121

# base environment
conda activate base
pip show torch
# Expected: torch 2.5.1
```

### Critical Dependencies in zinets environment:
- **docling-ibm-models** - May be sensitive to torch changes
- **easyocr** - Generally compatible
- **effdet** - Computer vision, should be fine
- **laserembeddings** - Key for multilingual embeddings, test carefully
- **sentence-transformers** - Core dependency, usually forward-compatible
- **timm** - Computer vision models
- **torchvision** - Will need coordinated upgrade

## 🎯 Upgrade Strategy

### Step 1: Create Backup and New Environment Strategy
```bash
# Create backup of current zinets environment
conda activate zinets
conda env export > /home/papagame/projects/digital-duck/st_semantics/docs/zinets_backup_$(date +%Y%m%d).yml

# Verify backup created
ls -la /home/papagame/projects/digital-duck/st_semantics/docs/zinets_backup_*.yml
```

### Step 2: Remove torch from Base Environment
```bash
# Activate base environment
conda activate base

# Check what depends on torch
pip show torch

# Uninstall torch from base (this may remove accelerate, bitsandbytes)
pip uninstall torch

# If accelerate/bitsandbytes are needed elsewhere, note for later reinstall
# pip install accelerate bitsandbytes
```

### Step 3: Create Fresh Environment (zinets2)
```bash
# Create new environment with Python 3.11 (same as zinets)
conda create -n zinets2 python=3.11 -y

# Activate new environment
conda activate zinets2

# Verify clean state
python -c "import sys; print(f'Python: {sys.version}')"

# Check CUDA compatibility
nvidia-smi
```

### Step 4: Install torch 2.6+ First (Clean Install)
```bash
# Install torch and torchvision first from PyTorch index
pip install torch>=2.6.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Verify torch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Step 5: Install Core Dependencies
```bash
# Install packages individually to avoid conflicts
pip install streamlit
pip install pandas>=2.1.4
pip install numpy>=1.21.0,<2.0.0
pip install plotly>=6.1.1
pip install scikit-learn>=1.0.0
pip install transformers>=4.30.0
pip install umap-learn>=0.5.3
pip install phate>=1.0.7
pip install python-dotenv>=0.19.0
pip install requests>=2.31.0
pip install sentencepiece>=0.1.99
pip install kaleido>=0.2.1
pip install deepl>=1.15.0
pip install pillow>=7.1.0
pip install altair>=4.0.0

# Optional: Install sentence-transformers if needed
pip install sentence-transformers
```

### Step 6: Alternative - Install from Requirements (Modified Approach)
```bash
# If individual installs work, try requirements.txt
cd /home/papagame/projects/digital-duck/st_semantics

# Create temporary requirements without torch lines
grep -v "torch\|torchvision" requirements.txt > temp_requirements.txt

# Install non-torch requirements
pip install -r temp_requirements.txt

# Clean up
rm temp_requirements.txt
```

### Step 7: Verify Core Dependencies
```bash
# Activate zinets2 environment
conda activate zinets2

# Check if key packages still work
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python -c "import sentence_transformers; print('Sentence-Transformers: OK')"

# Test critical imports
python -c "
try:
    from transformers import AutoTokenizer, AutoModel
    print('✅ Transformers: OK')
except Exception as e:
    print(f'❌ Transformers error: {e}')
"
```

### Step 8: Test Core Embedding Models
```bash
# Activate zinets2 environment
conda activate zinets2
cd /home/papagame/projects/digital-duck/st_semantics/src

# Test basic embedding functionality
python -c "
import sys
sys.path.append('.')
from models.model_manager import get_model

try:
    # Test our elite tier models
    models_to_test = ['Sentence-BERT Multilingual', 'BGE-M3 (Ollama)', 'E5-Base-v2']

    for model_name in models_to_test:
        print(f'Testing {model_name}...')
        try:
            model = get_model(model_name)
            embeddings = model.get_embeddings(['test', 'hello'], lang='en')
            if embeddings is not None:
                print(f'✅ {model_name}: OK - shape {embeddings.shape}')
            else:
                print(f'⚠️  {model_name}: Returned None')
        except Exception as e:
            print(f'❌ {model_name}: Error - {e}')

except Exception as e:
    print(f'❌ Model manager error: {e}')
"
```

### Step 9: Test Streamlit Application
```bash
# Activate zinets2 environment
conda activate zinets2
cd /home/papagame/projects/digital-duck/st_semantics/src

# Test main application starts
streamlit run Welcome.py --server.headless true --server.port 8502 &
STREAMLIT_PID=$!

# Wait a moment for startup
sleep 10

# Check if it's running
curl -s http://localhost:8502 > /dev/null && echo "✅ Streamlit app started successfully" || echo "❌ Streamlit app failed to start"

# Kill the test instance
kill $STREAMLIT_PID 2>/dev/null
```

### Step 10: Test PHATE Visualizations
```bash
# Activate zinets2 environment
conda activate zinets2

# Test dimensional reduction functionality
python -c "
import sys
sys.path.append('.')

try:
    from components.embedding_viz import EmbeddingVisualizer
    import pandas as pd

    # Test with simple data
    test_data = pd.DataFrame({
        'text': ['hello', 'world', 'test', 'data'],
        'lang': ['en', 'en', 'en', 'en']
    })

    viz = EmbeddingVisualizer()
    # This should not crash
    print('✅ EmbeddingVisualizer initialized successfully')

except Exception as e:
    print(f'❌ PHATE visualization error: {e}')
"
```

## 🚨 Troubleshooting Guide

### Common Issues and Solutions:

#### Issue 1: CUDA Version Mismatch
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# If mismatch, try different torch index:
# For CUDA 11.8: --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1: --index-url https://download.pytorch.org/whl/cu121
```

#### Issue 2: Dependency Conflicts
```bash
# If packages break, reinstall with updated torch
pip install sentence-transformers --upgrade --force-reinstall
pip install transformers --upgrade
```

#### Issue 3: laserembeddings Compatibility
```bash
# If laserembeddings fails, try:
pip uninstall laserembeddings
pip install laserembeddings --no-deps
pip install torch>=2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Issue 4: Complete Environment Reset (Last Resort)
```bash
# If upgrade fails completely, restore from backup:
conda env remove -n zinets
conda env create -f /home/papagame/projects/digital-duck/st_semantics/zinets_backup_$(date +%Y%m%d).yml

# Then try upgrade again with different approach
```

## ✅ Success Criteria

**Upgrade is successful when:**
1. ✅ `python -c "import torch; print(torch.__version__)"` shows ≥ 2.6.0
2. ✅ BGE-M3, Sentence-BERT Multilingual, E5-Base-v2 models load and generate embeddings
3. ✅ Streamlit application starts without errors
4. ✅ PHATE visualizations render correctly
5. ✅ No more "torch.load vulnerability" error messages

## 📝 Post-Upgrade Validation (zinets2)

### Test Our Elite Models
Run the following validation script after upgrade:

```bash
# Activate zinets2 environment
conda activate zinets2
cd /home/papagame/projects/digital-duck/st_semantics/src
python -c "
import sys
sys.path.append('.')

print('=== Post-Upgrade Validation ===')
print(f'PyTorch version: {__import__('torch').__version__}')

from models.model_manager import get_active_models
print(f'Active models: {len(get_active_models())} available')

# Test elite tier models specifically
elite_models = ['Sentence-BERT Multilingual', 'BGE-M3 (Ollama)', 'E5-Base-v2']
test_texts = ['hello', 'world', '0', '1', 'mother', 'father']

for model_name in elite_models:
    try:
        model = get_model(model_name)
        embeddings = model.get_embeddings(test_texts, lang='en')
        if embeddings is not None and len(embeddings) == len(test_texts):
            print(f'✅ {model_name}: Working - {embeddings.shape}')
        else:
            print(f'⚠️  {model_name}: Partial success')
    except Exception as e:
        print(f'❌ {model_name}: {str(e)[:100]}...')

print('=== Validation Complete ===')
"
```

## 🔄 Recovery Plan

If anything goes wrong during upgrade:

1. **Quick Recovery**: Restore from backup environment
2. **Alternative Strategy**: Create new environment with torch 2.6+ and reinstall requirements
3. **Gradual Migration**: Keep old environment, create new one, migrate gradually

**Backup files location**: `/home/papagame/projects/digital-duck/st_semantics/zinets_backup_*.yml`

---

## 📋 Execution Checklist (Fresh Environment Strategy)

- [ ] Create environment backup (zinets)
- [ ] Remove torch from base environment
- [ ] Create fresh zinets2 environment
- [ ] Install torch 2.6+ first (clean install)
- [ ] Install core dependencies individually
- [ ] Verify core dependencies work
- [ ] Test embedding models (BGE-M3, Sentence-BERT, E5-Base-v2)
- [ ] Test Streamlit application
- [ ] Test PHATE visualizations
- [ ] Run post-upgrade validation script
- [ ] Confirm no more security vulnerability errors
- [ ] Update CLAUDE.md to use zinets2 environment

**Execute manually and stop at any step that shows errors. We can troubleshoot issues as they arise.**

---
*Generated: $(date)*
*Environment: zinets (torch 2.4.1+cu121 → 2.6.0+)*
*Security: CVE-2025-32434 mitigation*