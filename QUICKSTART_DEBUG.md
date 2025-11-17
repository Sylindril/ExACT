# Quick Start: Debug Wikipedia Task with Local VLM

**Goal**: Run a simple Wikipedia task with your local VLM to verify everything works.

> **Note**: This guide uses template files to avoid overwriting your existing configurations.
> See `DEBUG_SETUP_INSTRUCTIONS.md` for detailed setup instructions.

## Prerequisites

1. Local VLM running (vLLM, SGLang, LM Studio, Ollama, etc.)
2. Python environment with ExACT dependencies
3. Playwright browsers installed (`playwright install chromium`)

## Quick Setup (4 Steps)

### 1. Copy template files

```bash
cp debug_env.sh.template debug_env.sh
cp debug_run.sh.template debug_run.sh
chmod +x debug_run.sh
```

### 2. Configure your VLM

Edit `debug_env.sh`:

```bash
vim debug_env.sh
```

**Change these two lines:**
```bash
export VLM_ENDPOINT="http://localhost:8000/v1"  # ← Your VLM endpoint
export VLM_MODEL_NAME="your-model-name"          # ← Your model name
```

**Examples**:
- vLLM: `http://localhost:8000/v1` with model `"llava-hf/llava-v1.6-vicuna-13b-hf"`
- SGLang: `http://localhost:30000/v1` with model `"lmms-lab/llama3-llava-next-8b"`
- LM Studio: `http://localhost:1234/v1` with your loaded model name
- Ollama: `http://localhost:11434/v1` with model `"llava:13b"`

### 3. Create test task and add provider

**Create the test task:**
```bash
python3 setup_debug_task.py
```

**Add your VLM provider to providers.yaml:**
```bash
# See examples first
cat configs/llms/providers.yaml.example

# Edit your providers.yaml
vim configs/llms/providers.yaml
```

Add your provider (example):
```yaml
local_vlm:
    provider: openai
    llm_api_base: http://localhost:8000/v1
    llm_api_version: ''
```

### 4. Run the test

```bash
source debug_env.sh
./debug_run.sh
```

## What Happens

1. Browser opens Wikipedia page about Python programming language
2. Your VLM sees the page and the task: *"What year was Python first released?"*
3. Agent tries to find the answer (1991)
4. Results saved to `data/debug_wikipedia/`

## Check Results

```bash
# View HTML visualization (shows what the agent did)
firefox data/debug_wikipedia/render_999.html

# Check if it passed
cat data/debug_wikipedia/log_files/*.log | grep -i "result"

# View the full log
cat data/debug_wikipedia/log_files/*.log
```

## Troubleshooting

### "Connection refused" or timeout

Your VLM isn't running or wrong endpoint. Test it:
```bash
source debug_env.sh
curl $VLM_ENDPOINT/models
```

Should return your model list.

### "Model not found"

Wrong model name. Check what's available:
```bash
curl $VLM_ENDPOINT/models
```

### Want to use text-only mode?

Edit `debug_run.sh`, change these lines:
```bash
--action_set_tag id_accessibility_tree \
--observation_type accessibility_tree \
```

This avoids needing vision capabilities.

### Still not working?

See the detailed guide: `DEBUG_RUN_WIKIPEDIA_TASK.md`

## What Files Were Modified?

**Created (your local copies from templates):**
- ✅ `debug_env.sh` - Your VLM configuration
- ✅ `debug_run.sh` - Your test script
- ✅ `configs/visualwebarena/test_wikipedia_debug/` - Test tasks

**Modified (you should edit):**
- ✅ `configs/llms/providers.yaml` - Added your provider

**NOT Modified (original templates):**
- ✅ `debug_env.sh.template` - Clean template
- ✅ `debug_run.sh.template` - Clean template
- ✅ `configs/llms/providers.yaml.example` - Examples only

## Next Steps

Once task 999 works:

1. **Try task 998** (requires search):
   ```bash
   # Edit debug_run.sh, change test_idx="999" to test_idx="998"
   vim debug_run.sh
   ./debug_run.sh
   ```

2. **Increase max_steps** for more exploration:
   ```bash
   # In debug_run.sh, change max_steps=3 to max_steps=10
   ```

3. **Try a real VWA task**:
   ```bash
   # See DEBUG_RUN_WIKIPEDIA_TASK.md
   # Section: "Running Real VWA Wikipedia Tasks"
   ```

## Expected Output (Success)

```
=========================================
Running Wikipedia Debug Task
=========================================
Model: llava-v1.6-vicuna-13b-hf
Provider: local_vlm
Test task: 999
Max steps: 3
=========================================

[Intent]: What year was Python first released?
[Config file]: configs/visualwebarena/test_wikipedia_debug/999.json
...
[Action]: click [element_id]
...
[Result] (PASS)

=========================================
Done!
=========================================
Results saved to: data/debug_wikipedia
```

## Keep Your Workspace Clean

Add to `.gitignore`:
```bash
cat >> .gitignore << 'EOF'

# Local debugging files (created from templates)
debug_env.sh
debug_run.sh
data/debug_wikipedia/
configs/visualwebarena/test_wikipedia_debug/
EOF
```

---

**For detailed setup**: See `DEBUG_SETUP_INSTRUCTIONS.md`
**For troubleshooting**: See `DEBUG_RUN_WIKIPEDIA_TASK.md`
**For understanding VWA links**: See `DEBUG_VWA_WIKIPEDIA_LINK.md`
