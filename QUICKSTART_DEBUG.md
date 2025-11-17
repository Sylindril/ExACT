# Quick Start: Debug Wikipedia Task with Local VLM

**Goal**: Run a simple Wikipedia task with your local VLM to verify everything works.

## 3-Step Setup

### 1. Create the test task

```bash
python3 setup_debug_task.py
```

This creates a simple task: "What year was Python first released?" (answer: 1991)

### 2. Configure your VLM

Edit `debug_env.sh`:

```bash
vim debug_env.sh
```

Change these two lines:
```bash
export VLM_ENDPOINT="http://localhost:8000/v1"  # Your VLM endpoint
export VLM_MODEL_NAME="your-model-name"          # Your model name
```

**Examples**:
- vLLM: `http://localhost:8000/v1` with model like `"llava-hf/llava-v1.6-vicuna-13b-hf"`
- SGLang: `http://localhost:30000/v1` with model like `"lmms-lab/llama3-llava-next-8b"`
- LM Studio: `http://localhost:1234/v1` with whatever model name you loaded
- Ollama: `http://localhost:11434/v1` with model like `"llava:13b"`

Then load the environment:
```bash
source debug_env.sh
```

### 3. Run the test

```bash
./debug_run.sh
```

## What Happens

1. Browser opens Wikipedia page about Python
2. Your VLM sees the page and the task
3. Agent tries to find the release year
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
curl http://localhost:8000/v1/models
```

Should return your model list.

### "Model not found"

Wrong model name. Check what's available:
```bash
curl http://localhost:8000/v1/models
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

## Next Steps

Once this works:

1. **Try the search task** (requires interaction):
   ```bash
   # Edit debug_run.sh, change test_idx to 998
   vim debug_run.sh  # Change test_idx="999" to test_idx="998"
   ./debug_run.sh
   ```

2. **Increase max_steps** to let agent explore more:
   ```bash
   # In debug_run.sh, change max_steps=3 to max_steps=10
   ```

3. **Try a real VWA task**:
   ```bash
   # Follow instructions in DEBUG_RUN_WIKIPEDIA_TASK.md
   # Section: "Running Real VWA Wikipedia Tasks"
   ```

## Files Created

- `debug_env.sh` - Environment configuration
- `debug_run.sh` - Test runner script
- `setup_debug_task.py` - Creates test tasks
- `configs/visualwebarena/test_wikipedia_debug/999.json` - Simple test task
- `configs/visualwebarena/test_wikipedia_debug/998.json` - Search test task

## Expected Output (Success)

```
[Intent]: What year was Python first released?
[Config file]: configs/visualwebarena/test_wikipedia_debug/999.json
...
[Action]: click [element_id]
...
[Action]: type [element_id] [text]
...
[Result] (PASS)
```

That's it! Once you see `(PASS)`, your pipeline is working.

---

**For more details**, see:
- `DEBUG_RUN_WIKIPEDIA_TASK.md` - Complete debugging guide
- `DEBUG_VWA_WIKIPEDIA_LINK.md` - Understanding the link system
