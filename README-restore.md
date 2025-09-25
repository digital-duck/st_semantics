# Restore zinets Environment from Backup

**Objective**: Restore the broken zinets virtual environment from the backup file created before the torch upgrade attempt.

## 🚨 Situation

- **Current State**: zinets environment is broken after failed torch upgrade
- **Solution**: Restore from conda environment export backup
- **Benefit**: Get back to working state with torch 2.4.1+cu121 (functional but vulnerable)
- **Next Step**: Create fresh zinets2 environment separately for torch 2.6+ upgrade

## 🔄 Recovery Steps

### Step 1: Check Available Backup Files
```bash
# Find your backup files
ls -la /home/papagame/projects/digital-duck/st_semantics/docs/zinets_backup_*.yml

# Expected output: zinets_backup_YYYYMMDD.yml file(s)
# zinets_backup_20250924.yml
```

### Step 2: Remove Broken Environment
```bash
# Remove the broken zinets environment completely
conda env remove -n zinets

# Verify it's removed (zinets should no longer appear in list)
conda env list
```

### Step 3: Restore from Backup
```bash
# Restore zinets from the backup file (use the most recent date)
# Replace YYYYMMDD with the actual date from Step 1


# Example (if backup file is zinets_backup_20250924.yml):
conda env create -f /home/papagame/projects/digital-duck/st_semantics/docs/zinets_backup_20250924.yml
```

### Step 4: Verify Restoration Success
```bash
# Check that zinets environment is restored
conda env list

# Activate restored environment
conda activate zinets

# Verify Python and key packages
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import streamlit; print(f'Streamlit: OK')"
python -c "import pandas; print(f'Pandas: OK')"
```

### Step 5: Test Core Functionality
```bash
# Navigate to project directory
cd /home/papagame/projects/digital-duck/st_semantics/src

# Test model manager functionality
python -c "
import sys
sys.path.append('.')

try:
    from models.model_manager import get_active_models
    active_models = get_active_models()
    print(f'✅ Active models: {len(active_models)} available')

    # List first few model names
    model_names = list(active_models.keys())[:5]
    print(f'Sample models: {model_names}')
except Exception as e:
    print(f'❌ Model manager error: {e}')
"
```

### Step 6: Test Elite Tier Models (Optional Verification)
```bash
# Test our key elite models work
python -c "
import sys
sys.path.append('.')

try:
    from models.model_manager import get_model

    # Test models that should work with torch 2.4.1
    test_models = ['Sentence-BERT Multilingual', 'E5-Base-v2']

    for model_name in test_models:
        try:
            model = get_model(model_name)
            print(f'✅ {model_name}: Model loaded successfully')
        except Exception as e:
            print(f'⚠️  {model_name}: {str(e)[:50]}...')

except Exception as e:
    print(f'❌ Model testing error: {e}')
"
```

### Step 7: Quick Streamlit Test
```bash
# Test that Streamlit application can start
cd /home/papagame/projects/digital-duck/st_semantics/src

# Quick startup test (will run in background briefly)
streamlit run Welcome.py --server.headless true --server.port 8503 &
STREAMLIT_PID=$!

# Wait for startup
sleep 5

# Check if it's running
curl -s http://localhost:8503 > /dev/null && echo "✅ Streamlit app started successfully" || echo "❌ Streamlit app failed to start"

# Stop the test instance
kill $STREAMLIT_PID 2>/dev/null

# Wait a moment for cleanup
sleep 2
```

## ✅ Success Criteria

**Restoration is successful when:**
1. ✅ `conda activate zinets` works without errors
2. ✅ `python -c "import torch; print(torch.__version__)"` shows 2.4.1+cu121
3. ✅ Model manager loads active models successfully
4. ✅ Sentence-BERT Multilingual and E5-Base-v2 models load
5. ✅ Streamlit application starts without errors
6. ✅ You can continue your geosemantry research work

## 🚨 Important Notes

### Security Warning
- **Restored environment still has torch 2.4.1+cu121** - vulnerable to CVE-2025-32434
- **Only use for existing research** - don't load untrusted torch files
- **Create zinets2 environment** for secure torch 2.6+ version

### Next Steps After Successful Restore
1. **Resume geosemantry work** in restored zinets environment
2. **Create zinets2 environment** using README-upgrade.md fresh environment strategy
3. **Migrate to zinets2** once torch 2.6+ environment is stable
4. **Keep zinets as backup** - don't attempt torch upgrade on it again

## 🔧 Troubleshooting

### Issue 1: Backup File Not Found
```bash
# Check if backup exists in different location
find /home/papagame/projects/digital-duck -name "*zinets_backup*" -type f

# If found elsewhere, use the correct path in conda env create command
```

### Issue 2: Conda Environment Creation Fails
```bash
# Try with full path to conda
/home/papagame/anaconda3/bin/conda env create -f /path/to/zinets_backup_YYYYMMDD.yml

# Or check conda is in PATH
which conda
conda --version
```

### Issue 3: Package Installation Errors During Restore
```bash
# If restore partially fails, you can manually fix missing packages:
conda activate zinets
pip install [missing-package-name]

# Or try recreating with conda-forge
conda env create -f zinets_backup_YYYYMMDD.yml -c conda-forge
```

### Issue 4: CUDA/GPU Issues After Restore
```bash
# Verify CUDA is still working
nvidia-smi

# Test torch CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# If CUDA not working, may need to reinstall torch
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

## 📋 Recovery Checklist

- [ ] Find backup file (zinets_backup_YYYYMMDD.yml)
- [ ] Remove broken zinets environment
- [ ] Restore from backup file
- [ ] Activate and verify basic functionality
- [ ] Test model manager and key models
- [ ] Test Streamlit application startup
- [ ] Confirm you can resume geosemantry research
- [ ] Plan zinets2 creation for secure torch 2.6+ upgrade

---

## 📝 Recovery Log Template

```bash
# Document your restore process:
Date: $(date)
Backup file used: zinets_backup_YYYYMMDD.yml
Recovery status: [SUCCESS/PARTIAL/FAILED]
Torch version after restore: [version]
Models working: [list]
Issues encountered: [any problems]
Next steps: [plan for zinets2 creation]
```

**Execute each step manually and verify success before proceeding to the next step.**

---
*Generated: 2025-09-24*
*Purpose: Emergency recovery of zinets environment*
*Security Note: Restored environment contains vulnerable torch 2.4.1+cu121*