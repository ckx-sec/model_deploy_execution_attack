import numpy as np
import cv2
import subprocess
import re
import os
import signal
import sys
import argparse
import json
import uuid
from concurrent.futures import ProcessPoolExecutor
import shutil
import tempfile
import glob

# --- Helper Functions for Host Machine Operations (Largely Unchanged) ---

def write_multiple_files_to_host(files_data, dest_dir):
    """
    Writes multiple image files from memory to a destination directory on the host.
    """
    for filename, data in files_data:
        path = os.path.join(dest_dir, filename)
        with open(path, 'wb') as f:
            f.write(data)

def remove_files_on_host_batch(file_pattern):
    """
    Removes multiple files matching a pattern on the host.
    """
    try:
        for f in glob.glob(file_pattern):
            os.remove(f)
    except OSError as e:
        print(f"Warning: Error while trying to batch remove '{file_pattern}': {e}")


# --- Worker Functions for Parallel Execution (Unchanged) ---

def _run_executable_and_parse_hooks(image_path_on_host, args):
    """
    Runs the executable on the host for a given image path and parses hook results.
    """
    # Assuming run_gdb_host.sh is a new script adapted for the host,
    # or the logic is brought directly into Python.
    script_path = os.path.join(os.path.dirname(__file__), "run_gdb_host.sh")
    
    # Paths are now all on the host filesystem.
    executable_on_host = args.executable
    model_on_host = args.model

    # The script needs absolute paths to function correctly regardless of CWD.
    # We explicitly call '/bin/bash' to avoid 'Exec format error' if the script itself doesn't have execute permissions.
    command = [
        '/bin/bash',
        script_path,
        os.path.abspath(executable_on_host),
        os.path.abspath(model_on_host),
        os.path.abspath(image_path_on_host),
        os.path.abspath(args.hooks) # Pass the absolute path to the hooks file as the last argument
    ]

    try:
        # Use a longer timeout as GDB startup can be slow.
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = e.stderr if hasattr(e, 'stderr') else "Timeout or error during execution"
        print(f"Error running host executable for '{image_path_on_host}': {stderr}")
        return False, {}

    hooked_values = {}
    is_successful = False
    full_output = result.stdout + "\n" + result.stderr
    output_lines = full_output.splitlines()

    for line in output_lines:
        if "true" in line.lower() and "HOOK_RESULT" not in line: is_successful = True
        if "HOOK_RESULT" in line:
            match = re.search(r'offset=(0x[0-9a-fA-F]+)\s+.*value=(.*)', line)
            if not match: continue
            
            offset, val_str = match.groups()
            val_str = val_str.strip()

            try:
                value = None
                if val_str.startswith('{'):
                    float_match = re.search(r'f\s*=\s*([-\d.e+]+)', val_str)
                    if float_match:
                        value = float(float_match.group(1))
                else:
                    value = float(val_str)
                
                if value is not None:
                    if offset not in hooked_values:
                        hooked_values[offset] = []
                    hooked_values[offset].append(value)
            except (ValueError, TypeError): pass
                
    return is_successful, hooked_values

def evaluate_mutation_on_host(task_args):
    """
    Worker function: evaluates a single mutated image and returns the loss.
    """
    image_path_on_host, target_hooks, args = task_args
    _, hooks = _run_executable_and_parse_hooks(image_path_on_host, args)
    loss = calculate_loss(hooks, target_hooks)
    return loss

# --- Core Infrastructure (Unchanged) ---

def run_attack_iteration(image_content, args, workdir, image_name_on_host):
    """
    Handles a single run of the executable, used for verification.
    """
    image_path_on_host = os.path.join(workdir, image_name_on_host)
    with open(image_path_on_host, 'wb') as f:
        f.write(image_content)
    is_successful, hooked_values = _run_executable_and_parse_hooks(image_path_on_host, args)
    os.remove(image_path_on_host)
    return is_successful, hooked_values

def calculate_loss(current_hooks, target_hooks, margin=0.0, weights=None):
    """
    Calculates the loss by comparing hook values from the current state to the target state.
    Loss is calculated as the mean squared error over all values in common hook addresses.

    :param current_hooks: A dictionary mapping hook addresses to lists of values from the current (attack) image.
                          Example: {"0x1234": [v1, v2], "0x5678": [v3, v4]}
    :param target_hooks: A dictionary of hook addresses and values from the golden image.
                         It defines the target state.
    :param margin: Unused parameter, kept for compatibility.
    :param weights: Unused parameter, kept for compatibility.
    :return: A float representing the total loss, averaged over all common data points.
    """
    if not isinstance(current_hooks, dict) or not isinstance(target_hooks, dict) or not current_hooks or not target_hooks:
        return float('inf')

    common_addresses = set(current_hooks.keys()) & set(target_hooks.keys())

    if not common_addresses:
        print("Warning: No common hook addresses found between current and target states.")
        return float('inf')

    total_squared_errors = []
    
    for addr in common_addresses:
        current_vals = current_hooks.get(addr, [])
        target_vals = target_hooks.get(addr, [])

        if len(current_vals) != len(target_vals):
            print(f"Warning: Hook count mismatch for address {addr}. Current: {len(current_vals)}, Target: {len(target_vals)}")
            return float('inf')

        if not current_vals:
            continue

        current_np = np.array(current_vals, dtype=np.float32)
        target_np = np.array(target_vals, dtype=np.float32)
        
        squared_errors = (current_np - target_np) ** 2
        total_squared_errors.extend(squared_errors.tolist())

    if not total_squared_errors:
        return 0.0

    return np.mean(total_squared_errors)


# --- SPSA Gradient Estimator (New Core Logic) ---

def estimate_gradient_spsa(image, args, target_hooks, workdir):
    """
    Estimates the gradient using SPSA.
    """
    run_id = uuid.uuid4().hex[:12]
    h, w = image.shape
    c = args.spsa_c

    # 1. Generate a single random perturbation vector (delta)
    # Delta entries are drawn from a Bernoulli distribution (+1, -1)
    delta = np.random.choice([-1, 1], size=(h, w))
    
    # 2. Create two perturbed images
    mutant_pos = image + c * delta
    mutant_neg = image - c * delta

    # 3. Prepare tasks for parallel evaluation
    _, encoded_pos = cv2.imencode(".png", mutant_pos.astype(np.uint8))
    _, encoded_neg = cv2.imencode(".png", mutant_neg.astype(np.uint8))
    
    fname_pos = f"temp_spsa_{run_id}_pos.png"
    fname_neg = f"temp_spsa_{run_id}_neg.png"
    
    path_pos = os.path.join(workdir, fname_pos)
    path_neg = os.path.join(workdir, fname_neg)

    with open(path_pos, 'wb') as f: f.write(encoded_pos)
    with open(path_neg, 'wb') as f: f.write(encoded_neg)

    tasks = [
        (path_pos, target_hooks, args),
        (path_neg, target_hooks, args)
    ]
    
    # 4. Evaluate the two mutations in parallel
    losses = np.zeros(2)
    print(f"--- Evaluating 2 SPSA mutations with {min(2, args.workers)} workers ---")
    try:
        # We only ever have 2 tasks, so no need for a huge worker pool
        with ProcessPoolExecutor(max_workers=min(2, args.workers)) as executor:
            results = executor.map(evaluate_mutation_on_host, tasks)
            for i, loss in enumerate(results):
                losses[i] = loss
        print("Evaluation complete.")
    finally:
        os.remove(path_pos)
        os.remove(path_neg)

    # 5. Calculate the SPSA gradient approximation
    loss_pos, loss_neg = losses[0], losses[1]
    
    # Avoid division by zero, although delta should not contain zeros
    grad = (loss_pos - loss_neg) / (2 * c * delta + 1e-10)
    
    return grad


# --- Main Attack Loop (Adapted for SPSA) ---

def main(args):
    loss_log_file = None
    attack_image = None
    best_loss_so_far = float('inf')

    try:
        os.setpgrp()
    except OSError:
        pass

    def sigint_handler(signum, frame):
        print("\nCtrl+C detected. Forcefully terminating all processes.")
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGINT, sigint_handler)
    
    temp_dir_base = "/dev/shm" if os.path.exists("/dev/shm") else None
    workdir = tempfile.mkdtemp(prefix="spsa_host_attack_", dir=temp_dir_base)
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"--- SPSA Attacker (Host Version) Started. Outputs: {args.output_dir}, Temp workdir: {workdir} ---")

        loss_log_path = os.path.join(args.output_dir, "spsa_loss_log_host.csv")
        loss_log_file = open(loss_log_path, 'w')
        loss_log_file.write("iteration,loss\n")
        print(f"--- Loss values will be logged to: {loss_log_path} ---")

        print("--- Verifying local paths ---")
        static_files = [args.executable, args.model, args.hooks, args.golden_image, args.image]
        for f in static_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required file not found: {f}")

        print("--- Getting target state from golden image ---")
        golden_image_bytes = cv2.imread(args.golden_image, cv2.IMREAD_UNCHANGED)
        _, encoded_golden = cv2.imencode(".png", golden_image_bytes)
        is_golden_ok, target_hooks = run_attack_iteration(encoded_golden.tobytes(), args, workdir, "temp_golden.png")

        if not is_golden_ok or not target_hooks:
            raise RuntimeError("Golden run failed or captured no hooks. Cannot proceed.")
        print(f"Target hooks captured: {target_hooks}")

        original_image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        attack_image = original_image.copy().astype(np.float32)

        # Adam optimizer parameters (retained from NES version)
        m = np.zeros_like(attack_image, dtype=np.float32)
        v = np.zeros_like(attack_image, dtype=np.float32)
        beta1 = 0.9
        beta2 = 0.999
        epsilon_adam = 1e-8

        # --- Learning Rate Decay State ---
        current_lr = args.learning_rate
        best_loss_for_decay = float('inf')
        patience_counter = 0
        if args.enable_lr_decay:
            print("--- Stagnation-based LR decay enabled ---")
            print(f"    Decay Rate: {args.lr_decay_rate}, Patience: {args.stagnation_patience}, Min Delta: {args.min_loss_delta}")

        for i in range(args.iterations):
            print(f"--- Iteration {i+1}/{args.iterations} (LR: {current_lr:.6f}) ---")
            
            # Use the SPSA gradient estimator
            grad = estimate_gradient_spsa(attack_image, args, target_hooks, workdir)
            
            # Adam Optimizer Update
            t = i + 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            update_step = current_lr * m_hat / (np.sqrt(v_hat) + epsilon_adam)
            attack_image -= update_step

            # Clipping and projection
            perturbation = np.clip(attack_image - original_image.astype(np.float32), -args.l_inf_norm, args.l_inf_norm)
            attack_image = np.clip(original_image.astype(np.float32) + perturbation, 0, 255)

            # Verification and Logging
            _, encoded_image = cv2.imencode(".png", attack_image.astype(np.uint8))
            is_successful, current_hooks = run_attack_iteration(encoded_image.tobytes(), args, workdir, "temp_verify.png")
            loss = calculate_loss(current_hooks, target_hooks)
            
            print(f"Attack result: {'Success' if is_successful else 'Fail'}. Internal state loss: {loss:.6f}")
            loss_log_file.write(f"{i+1},{loss:.6f}\n")
            loss_log_file.flush()

            # --- Learning Rate Decay Logic ---
            if args.enable_lr_decay:
                if loss < best_loss_for_decay - args.min_loss_delta:
                    best_loss_for_decay = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                print(f"Stagnation patience: {patience_counter}/{args.stagnation_patience}")

                if patience_counter >= args.stagnation_patience:
                    current_lr *= args.lr_decay_rate
                    print(f"!!! LR DECAY TRIGGERED. New learning rate: {current_lr:.6f} !!!")
                    patience_counter = 0 # Reset after decay

            # Save latest and best images
            latest_image_path = os.path.join(args.output_dir, "latest_attack_image_spsa_host.png")
            cv2.imwrite(latest_image_path, attack_image.astype(np.uint8))

            if loss < best_loss_so_far:
                best_loss_so_far = loss
                print(f"New best loss found: {loss:.6f}. Saving best image.")
                best_image_path = os.path.join(args.output_dir, "best_attack_image_spsa_host.png")
                cv2.imwrite(best_image_path, attack_image.astype(np.uint8))

            if is_successful:
                print("\nAttack successful!")
                successful_image_path = os.path.join(args.output_dir, "successful_attack_image_spsa_host.png")
                cv2.imwrite(successful_image_path, attack_image.astype(np.uint8))
                print(f"Adversarial image saved to: {successful_image_path}")
                break

    except (FileNotFoundError, RuntimeError, KeyboardInterrupt) as e:
        print(f"\nAn error or interrupt occurred: {e}")
        if attack_image is not None:
            print("Saving the last best image...")
            interrupted_image_path = os.path.join(args.output_dir, "interrupted_attack_image_spsa_host.png")
            cv2.imwrite(interrupted_image_path, attack_image.astype(np.uint8))
            print(f"Last best image saved to: {interrupted_image_path}")
    finally:
        if loss_log_file:
            loss_log_file.close()
        if workdir and os.path.exists(workdir):
            shutil.rmtree(workdir)
            print(f"Temporary directory {workdir} cleaned up.")
        print("Cleanup finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A grey-box adversarial attack using SPSA (Host Version).")
    # File Paths
    parser.add_argument("--executable", required=True, help="Local path to the target executable.")
    parser.add_argument("--image", required=True, help="Local path to the initial image to be attacked.")
    parser.add_argument("--hooks", required=True, help="Local path to the JSON file defining hook points.")
    parser.add_argument("--model", required=True, help="Local path to the model file (e.g., .onnx).")
    parser.add_argument("--golden-image", required=True, help="Local path to the image that produces the target state.")
    # Attack Hyperparameters
    parser.add_argument("--iterations", type=int, default=1000, help="Maximum number of attack iterations.")
    parser.add_argument("--learning-rate", type=float, default=1.0, help="Initial learning rate for the Adam optimizer.")
    parser.add_argument("--l-inf-norm", type=float, default=20.0, help="Maximum L-infinity norm for the perturbation.")
    # SPSA Hyperparameters
    parser.add_argument("--spsa-c", type=float, default=2.0, help="Perturbation magnitude for SPSA.")
    # Learning Rate Decay
    decay_group = parser.add_argument_group("Learning Rate Decay")
    decay_group.add_argument("--enable-lr-decay", action="store_true", help="Enable learning rate decay when loss stagnates.")
    decay_group.add_argument("--lr-decay-rate", type=float, default=0.8, help="Factor to multiply LR by on decay (e.g., 0.8).")
    decay_group.add_argument("--stagnation-patience", type=int, default=20, help="Number of iterations with no improvement before decaying LR.")
    decay_group.add_argument("--min-loss-delta", type=float, default=1e-4, help="Minimum change in loss to be considered an improvement.")
    # System
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel processes for evaluation.")
    parser.add_argument("--output-dir", type=str, default="attack_outputs_spsa_host", help="Directory to save output images and logs.")
    
    cli_args = parser.parse_args()
    main(cli_args) 