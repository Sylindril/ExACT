# Debugging VWA Wikipedia Tasks - Quick Reference

This README provides a quick reference for the debugging resources in this repository.

## 📁 File Structure

```
ExACT/
├── README_DEBUGGING.md                 # This file - overview of debugging resources
├── DEBUG_SETUP_INSTRUCTIONS.md         # Detailed setup guide (START HERE)
├── QUICKSTART_DEBUG.md                 # Quick 4-step guide
├── DEBUG_RUN_WIKIPEDIA_TASK.md         # Comprehensive debugging guide
├── DEBUG_VWA_WIKIPEDIA_LINK.md         # Understanding VWA link system
│
├── debug_env.sh.template               # Template: Copy and edit with your VLM config
├── debug_run.sh.template               # Template: Copy and customize test script
├── setup_debug_task.py                 # Script: Creates simple test tasks
│
├── configs/llms/
│   ├── providers.yaml                  # Your config (add your VLM provider here)
│   └── providers.yaml.example          # Examples of various VLM providers
│
└── .gitignore                          # Excludes your local copies from git
```

## 🚀 Quick Start

**Want to run a Wikipedia task with your local VLM? Follow these steps:**

1. **Copy the templates:**
   ```bash
   cp debug_env.sh.template debug_env.sh
   cp debug_run.sh.template debug_run.sh
   chmod +x debug_run.sh
   ```

2. **Configure your VLM** in `debug_env.sh`:
   ```bash
   vim debug_env.sh
   # Edit VLM_ENDPOINT and VLM_MODEL_NAME
   ```

3. **Add your provider** to `configs/llms/providers.yaml`:
   ```bash
   cat configs/llms/providers.yaml.example  # See examples
   vim configs/llms/providers.yaml           # Add your provider
   ```

4. **Create test task and run:**
   ```bash
   python3 setup_debug_task.py
   source debug_env.sh
   ./debug_run.sh
   ```

## 📚 Documentation Guide

| File | When to Read |
|------|--------------|
| **DEBUG_SETUP_INSTRUCTIONS.md** | First time setup - step-by-step instructions |
| **QUICKSTART_DEBUG.md** | Quick reference after initial setup |
| **DEBUG_RUN_WIKIPEDIA_TASK.md** | Troubleshooting or advanced scenarios |
| **DEBUG_VWA_WIKIPEDIA_LINK.md** | Understanding how VWA links work |

## 🎯 What Each Tool Does

### Templates (copy these)

| Template | Purpose | Your Action |
|----------|---------|-------------|
| `debug_env.sh.template` | VLM endpoint configuration | Copy to `debug_env.sh` and edit |
| `debug_run.sh.template` | Test runner script | Copy to `debug_run.sh` (optional edits) |

### Configuration Files

| File | Purpose | Your Action |
|------|---------|-------------|
| `configs/llms/providers.yaml` | VLM provider definitions | Add your provider entry |
| `configs/llms/providers.yaml.example` | Provider examples | Reference when editing |

### Scripts

| Script | Purpose | Your Action |
|--------|---------|-------------|
| `setup_debug_task.py` | Creates test tasks | Run once to create tasks |

## 🔧 Common VLM Configurations

### vLLM
```yaml
# In configs/llms/providers.yaml
vllm:
    provider: openai
    llm_api_base: http://localhost:8000/v1
    llm_api_version: ''
```

```bash
# In debug_env.sh
export VLM_ENDPOINT="http://localhost:8000/v1"
export VLM_MODEL_NAME="llava-hf/llava-v1.6-vicuna-13b-hf"
```

### SGLang
```yaml
# In configs/llms/providers.yaml
sglang:
    provider: sglang
    llm_api_base: http://localhost:30000/v1
    llm_api_version: ''
```

```bash
# In debug_env.sh
export VLM_ENDPOINT="http://localhost:30000/v1"
export VLM_MODEL_NAME="lmms-lab/llama3-llava-next-8b"
export PROVIDER="sglang"
```

### LM Studio
```yaml
# In configs/llms/providers.yaml
lmstudio:
    provider: openai
    llm_api_base: http://localhost:1234/v1
    llm_api_version: ''
```

```bash
# In debug_env.sh
export VLM_ENDPOINT="http://localhost:1234/v1"
export VLM_MODEL_NAME="your-loaded-model-name"
```

### Ollama
```yaml
# In configs/llms/providers.yaml
ollama:
    provider: openai
    llm_api_base: http://localhost:11434/v1
    llm_api_version: ''
```

```bash
# In debug_env.sh
export VLM_ENDPOINT="http://localhost:11434/v1"
export VLM_MODEL_NAME="llava:13b"
```

## ✅ Verification

**Check your VLM is accessible:**
```bash
source debug_env.sh
curl $VLM_ENDPOINT/models
```

**Should return:** List of available models

## 🎓 Learning Path

1. **Beginner**: Read `DEBUG_SETUP_INSTRUCTIONS.md` → Run simple task (999)
2. **Intermediate**: Read `QUICKSTART_DEBUG.md` → Run search task (998)
3. **Advanced**: Read `DEBUG_RUN_WIKIPEDIA_TASK.md` → Run real VWA tasks
4. **Expert**: Read `DEBUG_VWA_WIKIPEDIA_LINK.md` → Modify task configs

## 🐛 Troubleshooting Quick Links

| Issue | Solution |
|-------|----------|
| Connection refused | Check VLM is running: `curl $VLM_ENDPOINT/models` |
| Model not found | Check model name matches: `curl $VLM_ENDPOINT/models` |
| Browser errors | Install playwright: `playwright install chromium` |
| Import errors | Check PYTHONPATH: `export PYTHONPATH=$(pwd)` |
| Vision not working | Use text mode (see QUICKSTART_DEBUG.md) |

## 📝 What Gets Created

When you run the debugging setup:

**Created by you (from templates):**
- `debug_env.sh` - Your local VLM configuration
- `debug_run.sh` - Your local test script

**Created by scripts:**
- `configs/visualwebarena/test_wikipedia_debug/999.json` - Simple test task
- `configs/visualwebarena/test_wikipedia_debug/998.json` - Search test task
- `data/debug_wikipedia/` - Test results directory

**Excluded from git** (via `.gitignore`):
- `debug_env.sh`
- `debug_run.sh`
- `data/debug_wikipedia/`
- `configs/visualwebarena/test_wikipedia_debug/`

## 🎯 Test Tasks

### Task 999 (Simple)
- **URL**: https://en.wikipedia.org/wiki/Python_(programming_language)
- **Question**: What year was Python first released?
- **Answer**: 1991
- **Difficulty**: Easy - answer visible on page

### Task 998 (Search)
- **URL**: https://en.wikipedia.org/wiki/Main_Page
- **Question**: Search for 'Artificial Intelligence' and tell me what year the term was coined
- **Answer**: 1956
- **Difficulty**: Medium - requires search and navigation

## 💡 Pro Tips

1. **Start simple**: Use task 999 first to verify basic functionality
2. **Text mode**: If your VLM doesn't support vision, use text-only mode
3. **Increase steps**: Default is 3 steps; increase to 10+ for complex tasks
4. **Check logs**: Always check `data/debug_wikipedia/log_files/*.log` for details
5. **HTML output**: View `data/debug_wikipedia/render_999.html` to see agent actions

## 🔗 External Resources

- **VWA Repository**: https://github.com/web-arena-x/visualwebarena
- **ExACT Paper**: https://arxiv.org/abs/2410.02052
- **vLLM Docs**: https://docs.vllm.ai/
- **SGLang Docs**: https://sgl-project.github.io/

---

**Ready to start?** → Open `DEBUG_SETUP_INSTRUCTIONS.md`
