#!/usr/bin/env python3
"""
Setup a simple Wikipedia debug task.
This creates a minimal task configuration for testing your pipeline.
"""
import json
import os

def create_simple_wikipedia_task():
    """Create a minimal Wikipedia-only task for testing."""

    simple_task = {
        "sites": ["wikipedia"],
        "task_id": 999,
        "require_login": False,
        "storage_state": None,
        "start_url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
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

    # Create directory
    config_dir = "configs/visualwebarena/test_wikipedia_debug"
    os.makedirs(config_dir, exist_ok=True)

    # Save task config
    config_path = os.path.join(config_dir, "999.json")
    with open(config_path, "w") as f:
        json.dump(simple_task, f, indent=2)

    print(f"✓ Created simple Wikipedia task: {config_path}")
    print(f"  Task ID: {simple_task['task_id']}")
    print(f"  URL: {simple_task['start_url']}")
    print(f"  Intent: {simple_task['intent']}")
    print(f"  Expected answer: {simple_task['eval']['reference_answers']['exact_match']}")
    print("")
    print("Next steps:")
    print("  1. Edit debug_env.sh with your VLM endpoint and model name")
    print("  2. Run: source debug_env.sh")
    print("  3. Run: ./debug_run.sh")

def create_wikipedia_task_with_search():
    """Create a slightly more complex task that requires searching."""

    search_task = {
        "sites": ["wikipedia"],
        "task_id": 998,
        "require_login": False,
        "storage_state": None,
        "start_url": "https://en.wikipedia.org/wiki/Main_Page",
        "geolocation": None,
        "intent": "Search for 'Artificial Intelligence' and tell me what year the term was coined. Answer with just the year.",
        "image": [],
        "instantiation_dict": {},
        "require_reset": False,
        "eval": {
            "eval_types": ["string_match"],
            "reference_answers": {
                "exact_match": "1956"
            }
        }
    }

    config_dir = "configs/visualwebarena/test_wikipedia_debug"
    os.makedirs(config_dir, exist_ok=True)

    config_path = os.path.join(config_dir, "998.json")
    with open(config_path, "w") as f:
        json.dump(search_task, f, indent=2)

    print(f"✓ Created Wikipedia search task: {config_path}")
    print(f"  Task ID: {search_task['task_id']}")
    print(f"  URL: {search_task['start_url']}")
    print(f"  Intent: {search_task['intent']}")
    print(f"  Expected answer: {search_task['eval']['reference_answers']['exact_match']}")
    print("")
    print("To run this task:")
    print("  ./debug_run.sh  # Edit to change test_idx to 998")

if __name__ == "__main__":
    print("Setting up debug tasks...\n")

    # Create the simple task
    create_simple_wikipedia_task()
    print("")

    # Create the search task
    create_wikipedia_task_with_search()
    print("")

    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
