import os
import json
import re

def analyze_and_update_files(config_dir):
    """
    Analyzes JSON hook configuration files in the specified directory.
    It adds a 'branch_condition_meaning' field to entries with conditional branches.
    """
    print(f"--- Step 1: Analyzing and updating files in {config_dir} ---")

    cond_map = {
        "b.eq": "== (equal)",
        "b.ne": "!= (not equal)",
        "b.gt": "> (signed)",
        "b.ge": ">= (signed)",
        "b.lt": "< (signed)",
        "b.le": "<= (signed)",
        "b.hi": "> (unsigned)",
        "b.ls": "<= (unsigned)",
        "b.hs": ">= (unsigned/carry set)",
        "b.cs": ">= (unsigned/carry set)",
        "b.lo": "< (unsigned/carry clear)",
        "b.cc": "< (unsigned/carry clear)",
        "b.mi": "is negative (Most Significant Bit is 1)",
        "b.pl": "is positive or zero (Most Significant Bit is 0)",
        "b.vs": "overflow occurred",
        "b.vc": "no overflow occurred",
    }

    loss_map = {
        # A == B -> Log-Cosh Loss
        "b.eq": "Log-Cosh Loss",
        "b.vs": "Log-Cosh Loss",
        "b.vc": "Log-Cosh Loss",
        # A != B -> Hinge-like Margin Loss
        "b.ne": "Hinge-like Margin Loss",
        # A > B, A < B (strict) -> Logistic Loss
        "b.gt": "Logistic Loss",
        "b.lt": "Logistic Loss",
        "b.hi": "Logistic Loss",
        "b.lo": "Logistic Loss",
        "b.cc": "Logistic Loss",
        "b.mi": "Logistic Loss",
        # A >= B, A <= B (non-strict) -> Hinge Loss
        "b.ge": "Hinge Loss",
        "b.le": "Hinge Loss",
        "b.ls": "Hinge Loss",
        "b.hs": "Hinge Loss",
        "b.cs": "Hinge Loss",
        "b.pl": "Hinge Loss",
    }

    instr_re = re.compile(r"^\s*([a-zA-Z.]+)\s+([^,]+),\s*(.+)$")
    json_files_to_process = sorted([f for f in os.listdir(config_dir) if f.endswith('_hook_config.json')])

    for filename in json_files_to_process:
        filepath = os.path.join(config_dir, filename)
        print(f"\nProcessing {filename}...")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                hooks = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"  Error: Could not read or decode JSON from file. {e}")
            continue

        if not isinstance(hooks, list):
            print(f"  Warning: JSON content is not a list. Skipping.")
            continue

        file_modified = False
        for hook in hooks:
            branch_instr = hook.get("original_branch_instruction")
            if branch_instr in cond_map:
                hook_updated = False

                # 1. Add/Update loss function based on the instruction
                loss_function_name = loss_map.get(branch_instr)
                if loss_function_name and hook.get("loss_function_name") != loss_function_name:
                    hook["loss_function_name"] = loss_function_name
                    hook_updated = True

                # 2. Add/Update branch condition meaning
                instruction = hook.get("instruction", "")
                match = instr_re.match(instruction)
                if match:
                    op_code, op1, op2 = [s.strip() for s in match.groups()]
                    relationship = cond_map[branch_instr]
                    meaning = f"({op1}) {relationship} ({op2})"
                    if hook.get("branch_condition_meaning") != meaning:
                        hook["branch_condition_meaning"] = meaning
                        hook_updated = True
                else:
                    print(f"  > Kept hook at address {hook.get('address')}: instruction '{instruction}' could not be parsed for relation analysis.")
                
                if hook_updated:
                    file_modified = True
                    print(f"  > Updated hook at address {hook.get('address')}")
        
        if file_modified:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(hooks, f, indent=4, ensure_ascii=False)
                print(f"  SUCCESS: Updated {filename}.")
            except IOError as e:
                print(f"  ERROR: Could not write updated hooks to {filename}. {e}")
        else:
            print(f"  No changes needed for {filename}.")

def cleanup_directory(config_dir):
    """
    Cleans up the directory by deleting all files that do not end with '_hook_config.json'.
    """
    print(f"\n--- Step 2: Cleaning up directory: {config_dir} ---")
    files_to_delete = [f for f in os.listdir(config_dir) if not f.endswith('_hook_config.json')]
    
    if not files_to_delete:
        print("No files to delete.")
        return

    for filename in files_to_delete:
        filepath = os.path.join(config_dir, filename)
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
                print(f"  - Deleted: {filename}")
        except OSError as e:
            print(f"  - Error deleting file {filename}: {e}")
    
    print("\nCleanup complete.")

def main():
    """
    Main function to run the analysis and cleanup process.
    """
    script_dir = os.path.dirname(__file__)
    config_dir = os.path.join(script_dir, 'hook_config')

    if not os.path.isdir(config_dir):
        print(f"Error: Directory not found at '{config_dir}'")
        return

    analyze_and_update_files(config_dir)
    cleanup_directory(config_dir)

if __name__ == '__main__':
    main() 