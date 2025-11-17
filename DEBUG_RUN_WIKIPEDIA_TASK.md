# Quick Debug: Running a Wikipedia Task with Local VLM

This guide walks you through running a simple Wikipedia task using a real Wikipedia URL with your local VLM to test if everything is working.

## Prerequisites

1. **Local VLM running** (e.g., using vLLM, SGLang, Ollama, etc.)
   - Example: `http://localhost:8000/v1` (OpenAI-compatible API)
2. **Browser environment** - At minimum, you need the Reddit environment running (or can skip and just test Wikipedia standalone)

## Step 1: Configure Your Local VLM Provider

Edit `configs/llms/providers.yaml` to add your local VLM:

```bash
vim configs/llms/providers.yaml
```

Add your local VLM configuration:

```yaml
openai:
    provider: openai
    llm_api_base: https://api.openai.com/v1
    llm_api_version: ''

# Add your local VLM here
local_vlm:
    provider: sglang  # or 'openai' if using vLLM with OpenAI API
    llm_api_base: http://localhost:8000/v1  # Change to your VLM endpoint
    llm_api_version: ''
```

**Note**: If your local VLM uses OpenAI-compatible API (vLLM, SGLang, LM Studio, etc.), use `provider: openai`.

## Step 2: Set Up Environment Variables

Create a file `debug_env.sh`:

```bash
#!/bin/bash
# debug_env.sh

# Python path
export PYTHONPATH=$(pwd)

# Dataset
export DATASET=visualwebarena

# For debugging, we'll use real Wikipedia
# Reddit can be a placeholder since we're focusing on Wikipedia
export REDDIT="http://localhost:9999"  # or your actual Reddit instance
export WIKIPEDIA="https://en.wikipedia.org"  # Real Wikipedia!
export CLASSIFIEDS="http://localhost:9980"
export SHOPPING="http://localhost:7770"

# Local VLM API configuration
export PROVIDER="local_vlm"  # matches the name in providers.yaml
export AGENT_LLM_API_BASE="http://localhost:8000/v1"  # Your VLM endpoint
export AGENT_LLM_API_KEY="not-needed"  # Many local VLMs don't need a key

# Optional: if you want to use OpenAI for embeddings
export OPENAI_API_KEY="sk-your-key-here"  # Only if using OpenAI embeddings
export EMBEDDING_MODEL_PROVIDER="openai"

# Debug mode
export DEBUG=True
```

Load it:
```bash
source debug_env.sh
```

## Step 3: Create a Simple Wikipedia Task Config

First, let's modify the raw config to have a simple Wikipedia task:

```bash
python3 << 'PYTHON'
import json

# Create a minimal Wikipedia-only task for testing
simple_task = {
    "sites": ["wikipedia"],
    "task_id": 999,
    "require_login": False,
    "storage_state": None,
    "start_url": "__WIKIPEDIA__/wiki/Python_(programming_language)",
    "geolocation": None,
    "intent": "What year was Python first released? Answer with just the year.",
    "image": [],
    "instantiation_dict": {},
    "require_reset": False,
    "eval": {
        "eval_types": ["string_match"],
        "reference_answers": {
            "exact_match": "1991"
        }
    }
}

# Save as a test config directory
import os
os.makedirs("configs/visualwebarena/test_wikipedia_debug", exist_ok=True)

with open("configs/visualwebarena/test_wikipedia_debug/999.json", "w") as f:
    json.dump(simple_task, f, indent=2)

print("Created simple Wikipedia task: configs/visualwebarena/test_wikipedia_debug/999.json")
PYTHON
```

Now replace the `__WIKIPEDIA__` placeholder:

```bash
# This will replace __WIKIPEDIA__ with https://en.wikipedia.org
python3 << 'PYTHON'
import json

with open("configs/visualwebarena/test_wikipedia_debug/999.json", "r") as f:
    config = json.load(f)

# Replace the placeholder
config["start_url"] = config["start_url"].replace("__WIKIPEDIA__", "https://en.wikipedia.org")

with open("configs/visualwebarena/test_wikipedia_debug/999.json", "w") as f:
    json.dump(config, f, indent=2)

print("Updated start_url to: " + config["start_url"])
PYTHON
```

## Step 4: Create a Simple Test Script

Create `debug_run.sh`:

```bash
#!/bin/bash
# debug_run.sh

# Load environment
source debug_env.sh

# Model configuration
model="your-model-name"  # e.g., "llava-v1.6-34b", "qwen2-vl-7b", etc.
instruction_path="src/prompts/vwa/jsons/p_som_cot_id_actree_3s_final.json"
test_config_dir="configs/visualwebarena/test_wikipedia_debug"
test_idx="999"  # Our simple test task

# Simple agent (not MCTS for faster debugging)
agent="prompt"
max_steps=3  # Just a few steps for quick test
prompt_constructor_type="MCoTPolicyPConstructor"

# Save directory
SAVE_ROOT_DIR="data/debug_wikipedia"
mkdir -p $SAVE_ROOT_DIR

# Run evaluation
python runners/eval/eval_vwa_agent.py \
--instruction_path $instruction_path \
--test_idx $test_idx \
--model $model \
--provider $PROVIDER \
--agent_type $agent \
--prompt_constructor_type $prompt_constructor_type \
--result_dir $SAVE_ROOT_DIR \
--test_config_base_dir $test_config_dir \
--repeating_action_failure_th 5 \
--viewport_height 2048 \
--max_obs_length 3840 \
--action_set_tag som \
--observation_type image_som \
--temperature 0.7 \
--top_p 0.9 \
--max_steps $max_steps

echo ""
echo "Results saved to: $SAVE_ROOT_DIR"
echo "Check the HTML output: $SAVE_ROOT_DIR/render_999.html"
```

Make it executable:
```bash
chmod +x debug_run.sh
```

## Step 5: Run the Test

```bash
./debug_run.sh
```

## Step 6: Check the Results

### What to Look For:

1. **Console Output**:
   ```
   [Intent]: What year was Python first released?
   [Config file]: configs/visualwebarena/test_wikipedia_debug/999.json
   ...
   [Result] (PASS/FAIL)
   ```

2. **HTML Visualization**:
   ```bash
   # Open in browser to see the agent's actions step-by-step
   firefox data/debug_wikipedia/render_999.html
   # or
   chromium data/debug_wikipedia/render_999.html
   ```

3. **Log Files**:
   ```bash
   ls data/debug_wikipedia/log_files/
   cat data/debug_wikipedia/log_files/*.log
   ```

4. **Trajectory**:
   ```bash
   ls data/debug_wikipedia/trajectories/
   ```

## Troubleshooting

### Issue 1: VLM Not Responding

**Symptoms**: Timeout errors, connection refused

**Check**:
```bash
# Test your VLM endpoint
curl http://localhost:8000/v1/models
```

**Fix**: Make sure your VLM server is running and the port is correct.

### Issue 2: Image/SoM Not Working

**Symptoms**: Errors about image processing

**Try**: Use text-only mode instead:
```bash
# In debug_run.sh, change these lines:
--action_set_tag id_accessibility_tree \
--observation_type accessibility_tree \
```

### Issue 3: Browser/Playwright Issues

**Symptoms**: Browser won't launch, page errors

**Fix**:
```bash
# Install playwright browsers
playwright install chromium

# Or use headless mode
# (This is already default if render=False in the script)
```

### Issue 4: Model Not Understanding Task

**Symptoms**: Agent takes random actions, doesn't answer correctly

**Check**:
1. Your model supports vision (for `image_som` mode)
2. The model name is correct
3. Try with a simpler task first

**Debug with minimal model**:
```bash
# Test with OpenAI first to isolate VLM issues
export PROVIDER="openai"
export OPENAI_API_KEY="sk-your-key"
# Change model in debug_run.sh:
model="gpt-4o-mini"
```

## Alternative: Even Simpler Test (No Browser)

If you just want to test if your VLM API is working:

```bash
python3 << 'PYTHON'
import requests
import json

# Test your VLM endpoint
url = "http://localhost:8000/v1/chat/completions"
payload = {
    "model": "your-model-name",
    "messages": [
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ],
    "max_tokens": 10
}

response = requests.post(url, json=payload)
print("Response:", response.json())
PYTHON
```

## Running Real VWA Wikipedia Tasks

Once basic debugging works, try a real VWA Wikipedia task:

```bash
# Use task 45 (Reddit + Wikipedia)
# First, edit the config to use real Wikipedia
python3 << 'PYTHON'
import json
import os

# Read the raw config
with open("configs/visualwebarena/test_reddit_v2.raw.json", "r") as f:
    data = json.load(f)

# Find task 45
task_45 = [t for t in data if t["task_id"] == 45][0]

# Modify the Wikipedia URL to use a real article about endangered species
task_45["start_url"] = "__REDDIT__/f/dataisbeautiful/60156 |AND| __WIKIPEDIA__/wiki/Endangered_species_in_the_United_States"

# Save just this task
os.makedirs("configs/visualwebarena/test_wiki_real", exist_ok=True)
task_45_copy = task_45.copy()
task_45_copy["start_url"] = task_45_copy["start_url"].replace("__REDDIT__", os.environ.get("REDDIT", "http://localhost:9999"))
task_45_copy["start_url"] = task_45_copy["start_url"].replace("__WIKIPEDIA__", "https://en.wikipedia.org")

with open("configs/visualwebarena/test_wiki_real/45.json", "w") as f:
    json.dump(task_45_copy, f, indent=2)

print("Created real Wikipedia task 45")
PYTHON

# Run it
python runners/eval/eval_vwa_agent.py \
--instruction_path src/prompts/vwa/jsons/p_som_cot_id_actree_3s_final.json \
--test_idx 45 \
--model your-model-name \
--provider local_vlm \
--agent_type prompt \
--prompt_constructor_type MCoTPolicyPConstructor \
--result_dir data/debug_task45 \
--test_config_base_dir configs/visualwebarena/test_wiki_real \
--max_steps 5 \
--action_set_tag som \
--observation_type image_som
```

## Expected Workflow

Here's what happens when you run the script:

1. **Browser launches** (headless by default)
2. **Navigates to Wikipedia**: `https://en.wikipedia.org/wiki/Python_(programming_language)`
3. **VLM receives**:
   - Screenshot of the page with SoM (Set-of-Mark) overlays
   - Task instruction
   - Accessible elements
4. **VLM generates actions**:
   - Example: `click [element_id]`, `type [element_id] [text]`, `scroll up`, etc.
5. **Browser executes actions**
6. **Repeat** until task complete or max steps reached
7. **Evaluation**: Checks if the answer matches "1991"

## Key Files Reference

| File | Purpose |
|------|---------|
| `configs/llms/providers.yaml` | VLM endpoint configuration |
| `debug_env.sh` | Environment variables for your setup |
| `debug_run.sh` | Main test script |
| `configs/visualwebarena/test_wikipedia_debug/999.json` | Simple test task |
| `data/debug_wikipedia/render_999.html` | Visual output of agent actions |
| `data/debug_wikipedia/log_files/*.log` | Detailed execution logs |

## Success Indicators

You'll know it's working when:

✅ Browser successfully opens Wikipedia page
✅ VLM receives and processes the page screenshot
✅ Agent generates valid actions (click, type, etc.)
✅ Actions are executed on the page
✅ Evaluation shows PASS or at least completes without errors

## Next Steps

Once this works:

1. **Try different Wikipedia articles**
2. **Increase max_steps** to let the agent explore more
3. **Try more complex tasks** (multi-tab, requiring search, etc.)
4. **Switch to MCTS agents** for better performance (rmcts_mad, etc.)
5. **Run on full VWA benchmark**

---

Good luck! If you run into issues, check the log files first - they're very detailed and will show exactly where things are breaking.
