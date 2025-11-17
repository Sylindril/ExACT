# Setup Instructions for Debugging Wikipedia Tasks

This guide shows you how to set up the debugging environment **without overwriting your existing configurations**.

## File Structure

```
ExACT/
├── debug_env.sh.template           # Template for environment setup (COPY THIS)
├── debug_run.sh.template           # Template for test script (COPY THIS)
├── configs/llms/
│   ├── providers.yaml              # Your existing config (DON'T MODIFY)
│   └── providers.yaml.example      # Example providers (REFERENCE THIS)
├── setup_debug_task.py             # Run this to create test tasks
├── QUICKSTART_DEBUG.md             # Quick start guide
└── DEBUG_RUN_WIKIPEDIA_TASK.md     # Detailed debugging guide
```

## Setup Steps

### 1. Create Your Debug Environment File

**Copy the template:**
```bash
cp debug_env.sh.template debug_env.sh
```

**Edit your copy:**
```bash
vim debug_env.sh
```

**What to change:**
```bash
# Find these lines and update them:
export VLM_ENDPOINT="http://localhost:8000/v1"  # ← Change to your VLM endpoint
export VLM_MODEL_NAME="your-model-name"          # ← Change to your model name
```

**Examples:**
- vLLM: `export VLM_ENDPOINT="http://localhost:8000/v1"`
- SGLang: `export VLM_ENDPOINT="http://localhost:30000/v1"`
- LM Studio: `export VLM_ENDPOINT="http://localhost:1234/v1"`
- Ollama: `export VLM_ENDPOINT="http://localhost:11434/v1"`

### 2. Add Your VLM Provider to providers.yaml (Optional)

**Check the examples:**
```bash
cat configs/llms/providers.yaml.example
```

**Add your provider to providers.yaml:**
```bash
vim configs/llms/providers.yaml
```

**Example addition:**
```yaml
# Add this to your existing providers.yaml
local_vlm:
    provider: openai
    llm_api_base: http://localhost:8000/v1
    llm_api_version: ''
```

### 3. Create Your Debug Run Script

**Copy the template:**
```bash
cp debug_run.sh.template debug_run.sh
```

**Make it executable:**
```bash
chmod +x debug_run.sh
```

**Optional: Edit if you want to change settings:**
```bash
vim debug_run.sh
```

Common changes:
- `max_steps=3` → `max_steps=10` (more exploration)
- `test_idx="999"` → `test_idx="998"` (different test task)
- `--observation_type image_som` → `--observation_type accessibility_tree` (text-only mode)

### 4. Create Test Tasks

```bash
python3 setup_debug_task.py
```

This creates:
- `configs/visualwebarena/test_wikipedia_debug/999.json` - Simple question task
- `configs/visualwebarena/test_wikipedia_debug/998.json` - Search task

### 5. Run the Test

```bash
source debug_env.sh
./debug_run.sh
```

## What NOT to Modify

❌ **DON'T edit these existing files:**
- `configs/llms/providers.yaml` (unless adding your provider)
- Any files in `shells/` directory
- Any files in `configs/visualwebarena/test_*` (except our new test_wikipedia_debug)

✅ **DO edit these template copies:**
- `debug_env.sh` (your copy)
- `debug_run.sh` (your copy)

## Git Ignore Recommendations

Add these to your `.gitignore` to avoid committing your local configs:

```bash
# Add to .gitignore
debug_env.sh
debug_run.sh
data/debug_wikipedia/
configs/visualwebarena/test_wikipedia_debug/
```

To add:
```bash
cat >> .gitignore << 'EOF'

# Local debugging files (created from templates)
debug_env.sh
debug_run.sh
data/debug_wikipedia/
configs/visualwebarena/test_wikipedia_debug/
EOF
```

## Verification Checklist

Before running, verify:

- [ ] Created `debug_env.sh` from template
- [ ] Updated `VLM_ENDPOINT` and `VLM_MODEL_NAME` in `debug_env.sh`
- [ ] Added your provider to `providers.yaml` (or using existing one)
- [ ] Created `debug_run.sh` from template (or using as-is)
- [ ] Made `debug_run.sh` executable
- [ ] Ran `setup_debug_task.py` to create test tasks
- [ ] Your VLM is running and accessible

## Quick Test

Test your VLM endpoint is working:

```bash
# Load your environment
source debug_env.sh

# Test the endpoint
curl $VLM_ENDPOINT/models

# Should return a list of models
```

## Next Steps

Once setup is complete:

1. **Read:** `QUICKSTART_DEBUG.md` for a quick walkthrough
2. **Run:** `./debug_run.sh` to test
3. **Troubleshoot:** See `DEBUG_RUN_WIKIPEDIA_TASK.md` if issues occur

## Template Files Reference

| Template File | Purpose | Your Action |
|---------------|---------|-------------|
| `debug_env.sh.template` | Environment config | Copy to `debug_env.sh` and edit |
| `debug_run.sh.template` | Test runner | Copy to `debug_run.sh` (optional edits) |
| `configs/llms/providers.yaml.example` | Provider examples | Reference when editing `providers.yaml` |
| `setup_debug_task.py` | Task creator | Run as-is, no edits needed |

## Example: Complete Setup from Scratch

```bash
# 1. Copy templates
cp debug_env.sh.template debug_env.sh
cp debug_run.sh.template debug_run.sh
chmod +x debug_run.sh

# 2. Edit your config
vim debug_env.sh
# Update VLM_ENDPOINT and VLM_MODEL_NAME

# 3. Add provider (if needed)
vim configs/llms/providers.yaml
# Add your local_vlm or vllm configuration

# 4. Create test tasks
python3 setup_debug_task.py

# 5. Test your VLM
source debug_env.sh
curl $VLM_ENDPOINT/models

# 6. Run the test
./debug_run.sh

# 7. Check results
firefox data/debug_wikipedia/render_999.html
```

---

**That's it!** Your existing files remain untouched, and you have a clean debugging environment.
