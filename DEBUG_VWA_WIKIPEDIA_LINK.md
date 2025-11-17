# Debugging VWA: Replacing Arena Wikipedia Links with Real Wikipedia URLs

This guide explains how to replace Visual Web Arena (VWA) Wikipedia task links with actual Wikipedia URLs for debugging your pipeline.

## Understanding the Link System

### How VWA Links Work

In the VWA codebase, task configurations use placeholder URLs that get replaced with actual environment-specific URLs:

1. **Raw config files** (`.raw.json`) contain placeholders like `__WIKIPEDIA__`, `__REDDIT__`, `__SHOPPING__`, etc.
2. **Environment variables** define the actual URLs for each service
3. **The `generate_test_configs.py` script** replaces placeholders with environment variable values

#### Example from Config File

```json
{
  "task_id": 45,
  "start_url": "__REDDIT__/f/dataisbeautiful/60156 |AND| __WIKIPEDIA__/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
  "intent": "What is the total area in square miles, of the states that are ranked as having 150+ endangered species?"
}
```

After running `generate_test_configs.py`, `__WIKIPEDIA__` gets replaced with your environment variable value (e.g., `http://your_wikipedia_domain:8888`).

## Method 1: Environment Variable Override (Recommended for Testing)

This is the cleanest approach for debugging. Instead of pointing to your VWA Wikipedia instance, point directly to Wikipedia.

### Steps:

1. **Set the WIKIPEDIA environment variable** to a real Wikipedia URL:
   ```bash
   export WIKIPEDIA="https://en.wikipedia.org"
   export DATASET="visualwebarena"
   ```

2. **Regenerate test configs** with the new URL:
   ```bash
   python runners/utils/generate_test_configs.py
   ```

3. **Run your pipeline** as normal - it will now use the real Wikipedia URL

### Important Notes:

- **URL Path Compatibility**: The VWA configs often include specific paths like `/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing`
  - These paths are specific to Kiwix (VWA's offline Wikipedia)
  - They won't work on real Wikipedia (en.wikipedia.org)
  - You'll need to modify the paths in the raw config files (see Method 2)

- **Multi-page Tasks**: Some tasks use multiple sites with `|AND|` separator:
  ```
  __REDDIT__/path |AND| __WIKIPEDIA__/path
  ```
  Both URLs will be processed separately.

## Method 2: Manual Config File Editing (For Specific Tasks)

If you only want to test a specific task with a real Wikipedia URL, you can manually edit the generated config files.

### Steps:

1. **Locate the task config file**:
   ```bash
   # Task configs are in subdirectories
   # For example, task 45 from reddit would be:
   configs/visualwebarena/test_reddit_v2/45.json
   ```

2. **Edit the `start_url` field**:

   **Before:**
   ```json
   {
     "start_url": "http://your_vwa_wiki:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
   }
   ```

   **After (example with a real Wikipedia article):**
   ```json
   {
     "start_url": "https://en.wikipedia.org/wiki/Endangered_species"
   }
   ```

3. **Run your pipeline** on just that task:
   ```bash
   # Example: test only task 45
   export DATASET=visualwebarena
   python runners/eval/eval_vwa_agent.py \
     --test_idx 45 \
     --test_config_base_dir configs/visualwebarena/test_reddit_v2 \
     --result_dir data/debug_results \
     # ... other arguments
   ```

### Example Wikipedia URLs for Common Topics:

```bash
# General topics
https://en.wikipedia.org/wiki/Main_Page
https://en.wikipedia.org/wiki/United_States
https://en.wikipedia.org/wiki/Geography

# For multi-page tasks (with Reddit + Wikipedia):
__REDDIT__/f/dataisbeautiful/60156 |AND| https://en.wikipedia.org/wiki/Endangered_species_in_the_United_States
```

## Method 3: Modify Raw Config Files (For Systematic Changes)

If you want to change multiple tasks, edit the raw config files before generation.

### Steps:

1. **Edit the raw config file**:
   ```bash
   vim configs/visualwebarena/test_reddit_v2.raw.json
   ```

2. **Find and replace Wikipedia paths**:

   Search for entries with VWA-specific Wikipedia paths and replace them with real Wikipedia URLs:

   **Before:**
   ```json
   {
     "task_id": 45,
     "start_url": "__REDDIT__/f/dataisbeautiful/60156 |AND| __WIKIPEDIA__/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
   }
   ```

   **After:**
   ```json
   {
     "task_id": 45,
     "start_url": "__REDDIT__/f/dataisbeautiful/60156 |AND| __WIKIPEDIA__/wiki/Endangered_species"
   }
   ```

3. **Set WIKIPEDIA environment variable**:
   ```bash
   export WIKIPEDIA="https://en.wikipedia.org"
   export DATASET="visualwebarena"
   ```

4. **Regenerate configs**:
   ```bash
   python runners/utils/generate_test_configs.py
   ```

Now `__WIKIPEDIA__` will be replaced with `https://en.wikipedia.org`, and your path will become:
`https://en.wikipedia.org/wiki/Endangered_species`

## Understanding the Config Generation Process

The `generate_test_configs.py` script performs these operations:

```python
# Located at: runners/utils/generate_test_configs.py

replace_map = {
    "__REDDIT__": REDDIT,           # from env var
    "__SHOPPING__": SHOPPING,       # from env var
    "__WIKIPEDIA__": WIKIPEDIA,     # from env var
    "__CLASSIFIEDS__": CLASSIFIEDS, # from env var
}

# For each .raw.json file:
# 1. Read the raw JSON as text
# 2. Replace all placeholders with env var values
# 3. Write the full config as .json
# 4. Split into individual task files (0.json, 1.json, etc.)
```

## File Structure Overview

```
ExACT/
├── configs/
│   └── visualwebarena/
│       ├── test_reddit_v2.raw.json         # Raw config with __WIKIPEDIA__ placeholders
│       ├── test_reddit_v2.json             # Generated full config (after replacement)
│       └── test_reddit_v2/                 # Generated individual task configs
│           ├── 0.json
│           ├── 1.json
│           ├── 45.json                     # Example: task 45
│           └── ...
```

## Code Locations for Reference

| Component | File | Line |
|-----------|------|------|
| Placeholder replacement | `runners/utils/generate_test_configs.py` | 49-64 |
| URL processing | `src/envs/browser.py` | 268-284 |
| Example config with Wikipedia | `configs/visualwebarena/test_reddit_v2.raw.json` | ~1587 |
| Environment setup | `README.md` | 35-46 |

## Common Issues & Solutions

### Issue 1: "Page not found" on Wikipedia

**Problem**: VWA paths don't exist on real Wikipedia
```
https://en.wikipedia.org/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
```

**Solution**: Replace with actual Wikipedia article URLs:
```
https://en.wikipedia.org/wiki/Main_Page
https://en.wikipedia.org/wiki/[Article_Name]
```

### Issue 2: Task Evaluation Fails

**Problem**: The evaluation logic might expect specific content from VWA Wikipedia

**Solution**: Check the `eval` section of the task config:
```json
{
  "eval": {
    "eval_types": ["string_match", "url_match"],
    "reference_answers": {...}
  }
}
```

You may need to update `reference_answers` to match content from real Wikipedia.

### Issue 3: Multi-tab Tasks Don't Work

**Problem**: Tasks using `|AND|` to open multiple tabs (Reddit + Wikipedia)

**Solution**: This should still work! The browser will:
1. Open first URL (e.g., Reddit)
2. Open second URL (e.g., Wikipedia) in a new tab
3. Set focus to the first tab

Example from `src/envs/browser.py:268-284`:
```python
if start_url:
    start_urls = start_url.split(" |AND| ")
    for url in start_urls:
        page = await self.context.new_page()
        await page.goto(url)
    self.page = self.context.pages[0]  # Focus on first tab
```

## Quick Debug Example

Here's a complete example to test one task with real Wikipedia:

```bash
# 1. Set environment
export DATASET="visualwebarena"
export WIKIPEDIA="https://en.wikipedia.org"
export REDDIT="http://your_reddit:9999"  # Keep your other URLs

# 2. Edit a specific task (optional - for testing)
# Open configs/visualwebarena/test_reddit_v2.raw.json
# Find task 45, change the Wikipedia path to /wiki/Endangered_species

# 3. Regenerate configs
python runners/utils/generate_test_configs.py

# 4. Check the generated config
cat configs/visualwebarena/test_reddit_v2/45.json | grep start_url

# 5. Run the single task
python runners/eval/eval_vwa_agent.py \
  --test_idx 45 \
  --test_config_base_dir configs/visualwebarena/test_reddit_v2 \
  --result_dir data/debug_results \
  --max_steps 5 \
  --observation_type accessibility_tree \
  --action_set_tag som \
  # ... add your agent arguments
```

## Reverting Changes

To revert back to VWA Wikipedia:

```bash
# 1. Reset environment variable
export WIKIPEDIA="http://your_wikipedia_domain:8888"

# 2. Regenerate configs
python runners/utils/generate_test_configs.py
```

---

## Additional Resources

- **VWA Repository**: https://github.com/web-arena-x/visualwebarena
- **Environment Setup**: See main `README.md` in this repository
- **Browser Env Documentation**: The `browser_env` package handles URL navigation

## Summary

**For quick debugging:**
- Use **Method 1** (environment variable) if you want to test with real Wikipedia across multiple tasks
- Use **Method 2** (manual edit) if you want to test a single specific task

**Key takeaway:** The `__WIKIPEDIA__` placeholder system is flexible - you can point it anywhere you want, including real Wikipedia, for debugging your pipeline!
