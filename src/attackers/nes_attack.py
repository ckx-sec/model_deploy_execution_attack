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

# --- Helper Functions for Host Machine Operations ---

def write_multiple_files_to_host(files_data, dest_dir):
    """
    Writes multiple image files from memory to a destination directory on the host.
    
    :param files_data: A list of tuples, where each tuple is (filename, file_data_as_bytes).
    :param dest_dir: The destination directory on the host.
    """
    for filename, data in files_data:
        path = os.path.join(dest_dir, filename)
        with open(path, 'wb') as f:
            f.write(data)

def remove_files_on_host_batch(file_pattern):
    """
    Removes multiple files matching a pattern on the host.
    
    :param file_pattern: A shell glob pattern (e.g., "/tmp/some_dir/temp_nes_*.png").
    """
    try:
        for f in glob.glob(file_pattern):
            os.remove(f)
    except OSError as e:
        # Log this but don't re-raise, as cleanup failure is not critical.
        print(f"Warning: Error while trying to batch remove '{file_pattern}': {e}")


# --- Worker Functions for Parallel Execution ---

def _run_executable_and_parse_hooks(image_path_on_host, args):
    """
    Runs the executable on the host for a given image path and parses hook results.
    This is the core execution logic, separated from file I/O.
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
    Evaluates a single mutated image present on the host filesystem
    by running the executable and calculating the loss.
    This is the worker function for the process pool. It performs no I/O other than reading the image.
    """
    image_path_on_host, target_hooks, args = task_args
    
    # Run the executable for the specific image.
    _, hooks = _run_executable_and_parse_hooks(image_path_on_host, args)
    
    # Calculate the loss based on the results.
    loss = calculate_loss(hooks, target_hooks)
    return loss

# --- Core Infrastructure (Refactored for Host) ---

def run_attack_iteration(image_content, args, workdir, image_name_on_host):
    """
    Handles a single run of the executable for a given image on the host.
    Accepts raw image data (bytes).
    This function is now primarily used for single, non-batched runs like
    the initial golden image or final verification steps.
    """
    image_path_on_host = os.path.join(workdir, image_name_on_host)

    # image_content is expected to be bytes
    with open(image_path_on_host, 'wb') as f:
        f.write(image_content)

    is_successful, hooked_values = _run_executable_and_parse_hooks(image_path_on_host, args)

    # Clean up the temporary file
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


# --- NES Gradient Estimator (Optimized for Host) ---

def estimate_gradient_nes(image, args, target_hooks, workdir):
    """
    Estimates the gradient using NES with Antithetic Sampling.
    This version is optimized to use batched file I/O on the host.
    """
    run_id = uuid.uuid4().hex[:12]  # A unique ID for this batch of evaluations
    h, w, c = image.shape
    pop_size = args.population_size
    sigma = args.sigma
    
    if pop_size % 2 != 0:
        raise ValueError(f"Population size must be even. Got {pop_size}.")

    half_pop_size = pop_size // 2
    
    # 1. Generate noise and create all mutated images in memory
    noise_vectors = [np.random.randn(h, w, c) for _ in range(half_pop_size)]
    mutations_data_for_writing = []
    tasks = []

    for i, noise in enumerate(noise_vectors):
        # Create positive and negative mutations
        mutant_pos = image + sigma * noise
        mutant_neg = image - sigma * noise
        
        # Encode images to PNG format in memory
        _, encoded_pos = cv2.imencode(".png", mutant_pos.astype(np.uint8))
        _, encoded_neg = cv2.imencode(".png", mutant_neg.astype(np.uint8))

        # Generate unique names for the files on the host using the run_id
        unique_id_pos = f"{run_id}_{i}_pos"
        unique_id_neg = f"{run_id}_{i}_neg"
        
        fname_pos = f"temp_nes_{unique_id_pos}.png"
        fname_neg = f"temp_nes_{unique_id_neg}.png"
        
        mutations_data_for_writing.append((fname_pos, encoded_pos.tobytes()))
        mutations_data_for_writing.append((fname_neg, encoded_neg.tobytes()))

        # Prepare tasks for the process pool
        path_pos = os.path.join(workdir, fname_pos)
        path_neg = os.path.join(workdir, fname_neg)
        tasks.append((path_pos, target_hooks, args))
        tasks.append((path_neg, target_hooks, args))

    # --- BATCH I/O OPERATIONS ON HOST ---
    try:
        # 2. Write all generated images to the temporary directory
        print(f"--- Writing {len(mutations_data_for_writing)} images to host temporary directory ---")
        write_multiple_files_to_host(mutations_data_for_writing, workdir)

        # 3. Run evaluations in parallel.
        print(f"--- Evaluating {pop_size} mutations (using {half_pop_size} antithetic pairs) with {args.workers} workers ---")
        losses = np.zeros(pop_size)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(evaluate_mutation_on_host, tasks)
            for i, loss in enumerate(results):
                losses[i] = loss
                print(f"Evaluation progress: {i + 1}/{pop_size}", end='\r')
        print("\nEvaluation complete.")

    finally:
        # 4. Clean up all temporary files from the host
        print("--- Batch removing temporary images from host ---")
        cleanup_pattern = os.path.join(workdir, f"temp_nes_{run_id}_*.png")
        remove_files_on_host_batch(cleanup_pattern)

    # --- GRADIENT CALCULATION (Unchanged) ---
    if np.inf in losses:
        non_inf_max = np.max(losses[losses != np.inf], initial=0)
        losses[losses == np.inf] = non_inf_max + 1

    gradient = np.zeros_like(image, dtype=np.float32)
    for i in range(half_pop_size):
        loss_positive = losses[2 * i]
        loss_negative = losses[2 * i + 1]
        noise = noise_vectors[i]
        gradient += (loss_positive - loss_negative) * noise

    gradient /= (pop_size * sigma) 
    
    return gradient

# --- Main Attack Loop (Adapted for Host) ---

def main(args):
    loss_log_file = None
    attack_image = None
    best_loss_so_far = float('inf')

    # --- Graceful Termination on Ctrl+C ---
    # Set this process as a group leader. All subprocesses (workers) will be in this group.
    try:
        os.setpgrp()
    except OSError:
        pass  # May fail if already a group leader (e.g., in an interactive shell), which is fine.

    def sigint_handler(signum, frame):
        """
        Custom signal handler for Ctrl+C. This ensures the entire process group is terminated.
        """
        print("\nCtrl+C detected. Forcefully terminating all processes.")
        # Killing the entire process group is the most reliable way to stop all workers.
        # We use SIGKILL for an immediate, forceful shutdown.
        os.killpg(os.getpgrp(), signal.SIGKILL)

    # Register the handler for SIGINT (Ctrl+C).
    signal.signal(signal.SIGINT, sigint_handler)

    # Create a temporary directory on the host to act as the workdir.
    # --- Optimization: Use in-memory filesystem if available ---
    temp_dir_base = "/dev/shm" if os.path.exists("/dev/shm") else None
    workdir = tempfile.mkdtemp(prefix="nes_host_attack_", dir=temp_dir_base)
    if temp_dir_base:
        print(f"--- Optimization: Using in-memory temp directory: {workdir} ---")
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"--- NES Attacker (Host Version) Started. Outputs: {args.output_dir}, Temp workdir: {workdir} ---")

        loss_log_path = os.path.join(args.output_dir, "nes_loss_log_host.csv")
        loss_log_file = open(loss_log_path, 'w')
        loss_log_file.write("iteration,loss\n")
        print(f"--- Loss values will be logged to: {loss_log_path} ---")

        # Hyperparameter schedule state
        is_tuning_phase = False
        best_loss = float('inf')
        patience_counter = 0
        if args.enable_schedule:
            print("--- Hyperparameter schedule enabled ---")

        # Stagnation-resetting decay state
        stagnation_patience_counter = 0
        iteration_of_last_decay = 0
        total_decay_count = 0
        best_loss_for_stagnation = float('inf')
        if args.enable_stagnation_decay:
            print("--- Stagnation-resetting decay enabled ---")

        print("--- Preparing environment: Verifying local paths ---")
        # No need to copy files, just verify they exist
        static_files = [args.executable, args.model, args.hooks]
        gdb_script_path = os.path.join(os.path.dirname(__file__), "gdb_script_host.py")
        static_files.append(gdb_script_path) # Assumes gdb_script_host.py is still used by the host runner
        
        for f in static_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required file not found: {f}")

        print("--- Getting target state from golden image ---")
        # Use run_attack_iteration for this single run
        golden_image = cv2.imread(args.golden_image, cv2.IMREAD_COLOR)
        if golden_image is None:
             raise FileNotFoundError(f"Could not read golden image: {args.golden_image}")
        is_success_encoding, encoded_golden = cv2.imencode(".png", golden_image)
        if not is_success_encoding:
            raise RuntimeError("Failed to encode golden image.")

        is_golden_ok, target_hooks = run_attack_iteration(encoded_golden.tobytes(), args, workdir, "temp_golden_image.png")

        if not target_hooks:
            raise RuntimeError("Golden run captured no hooks. Cannot proceed.")
        print(f"Target hooks captured: {target_hooks}")

        original_image = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if original_image is None:
            raise FileNotFoundError(f"Could not read original image: {args.image}")

        # Ensure images have the same dimensions
        if original_image.shape != golden_image.shape:
            print(f"Warning: Image shapes do not match. Original: {original_image.shape}, Golden: {golden_image.shape}. Resizing original to match golden.")
            original_image = cv2.resize(original_image, (golden_image.shape[1], golden_image.shape[0]))

        attack_image = original_image.copy().astype(np.float32)

        # Adam optimizer parameters
        m = np.zeros_like(attack_image, dtype=np.float32)
        v = np.zeros_like(attack_image, dtype=np.float32)
        beta1 = 0.9
        beta2 = 0.999
        epsilon_adam = 1e-8

        for i in range(args.iterations):
            print(f"--- Iteration {i+1}/{args.iterations} ---")
            
            # Decay-and-Reset Logic
            decay_reason = None
            if args.enable_stagnation_decay:
                if (i - iteration_of_last_decay) >= args.lr_decay_steps:
                    decay_reason = f"SCHEDULED ({args.lr_decay_steps} steps passed)"
                elif stagnation_patience_counter >= args.stagnation_patience:
                    decay_reason = f"STAGNATION ({args.stagnation_patience} stagnant iterations)"

                if decay_reason:
                    total_decay_count += 1
                    iteration_of_last_decay = i
                    stagnation_patience_counter = 0

            current_lr = args.learning_rate * (args.lr_decay_rate ** total_decay_count)
            if decay_reason:
                 print(f"DECAY TRIGGERED by {decay_reason}. New LR: {current_lr:.6f}")

            # Use the new, host-based gradient estimator
            grad = estimate_gradient_nes(attack_image, args, target_hooks, workdir)
            
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

            # Check for success and report loss using in-memory data
            is_success_encoding, encoded_image = cv2.imencode(".png", attack_image.astype(np.uint8))
            if not is_success_encoding:
                print("Warning: Failed to encode attack image for verification.")
                is_successful, current_hooks, loss = False, {}, float('inf')
            else:
                # Use run_attack_iteration for the final check of the main attack image
                is_successful, current_hooks = run_attack_iteration(encoded_image.tobytes(), args, workdir, "temp_attack_image.png")
                loss = calculate_loss(current_hooks, target_hooks)
            
            print(f"Attack result: {'Success' if is_successful else 'Fail'}. Internal state loss: {loss:.6f}")
            loss_log_file.write(f"{i+1},{loss:.6f}\n")
            loss_log_file.flush()

            # --- Save latest and best images ---
            latest_image_path = os.path.join(args.output_dir, "latest_attack_image_nes_host.png")
            cv2.imwrite(latest_image_path, attack_image.astype(np.uint8))

            if loss < best_loss_so_far:
                best_loss_so_far = loss
                print(f"New best loss found: {loss:.6f}. Saving best image.")
                best_image_path = os.path.join(args.output_dir, "best_attack_image_nes_host.png")
                cv2.imwrite(best_image_path, attack_image.astype(np.uint8))

            # Update Stagnation Counter
            if args.enable_stagnation_decay:
                if decay_reason: best_loss_for_stagnation = loss
                if loss < best_loss_for_stagnation - args.min_loss_delta:
                    best_loss_for_stagnation = loss
                    stagnation_patience_counter = 0
                else: stagnation_patience_counter += 1
                print(f"Stagnation patience: {stagnation_patience_counter}/{args.stagnation_patience}")

            # Hyperparameter Schedule Logic
            if args.enable_schedule and not is_tuning_phase:
                if loss < best_loss - args.min_loss_delta:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= args.tuning_patience:
                    is_tuning_phase = True
                    args.sigma = args.tune_sigma
                    args.population_size = args.tune_population_size
                    print("\nSWITCHING TO FINE-TUNING PHASE\n")

            if is_successful:
                print("\nAttack successful!")
                successful_image_path = os.path.join(args.output_dir, "successful_attack_image_nes_host.png")
                cv2.imwrite(successful_image_path, attack_image.astype(np.uint8))
                print(f"Adversarial image saved to: {successful_image_path}")
                break

    except (FileNotFoundError, RuntimeError) as e:
        print(f"\nAn error occurred: {e}")
        if attack_image is not None:
            # This part is now less likely to be reached on interrupt, but kept for other errors.
            print("Interrupt received. Saving the last generated image...")
            interrupted_image_path = os.path.join(args.output_dir, "interrupted_attack_image_nes_host.png")
            cv2.imwrite(interrupted_image_path, attack_image.astype(np.uint8))
            print(f"Last image saved to: {interrupted_image_path}")
    finally:
        if loss_log_file:
            loss_log_file.close()
        # Clean up the temporary directory
        if workdir and os.path.exists(workdir):
            shutil.rmtree(workdir)
            print(f"Temporary directory {workdir} cleaned up.")
        print("Cleanup finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A grey-box adversarial attack using NES (Host Version).")
    # File Paths
    parser.add_argument("--executable", required=True, help="Local path to the target executable.")
    parser.add_argument("--image", required=True, help="Local path to the initial image to be attacked.")
    parser.add_argument("--hooks", required=True, help="Local path to the JSON file defining hook points.")
    parser.add_argument("--model", required=True, help="Local path to the model file (e.g., .onnx).")
    parser.add_argument("--golden-image", required=True, help="Local path to the image that produces the target state.")
    # Attack Hyperparameters
    parser.add_argument("--iterations", type=int, default=10000, help="Maximum number of attack iterations.")
    parser.add_argument("--learning-rate", type=float, default=2.0, help="Initial learning rate for the attack.")
    parser.add_argument("--l-inf-norm", type=float, default=20.0, help="Maximum L-infinity norm for the perturbation.")
    parser.add_argument("--lr-decay-rate", type=float, default=0.97, help="Learning rate decay rate.")
    parser.add_argument("--lr-decay-steps", type=int, default=10, help="Decay learning rate every N steps.")
    # NES Hyperparameters
    parser.add_argument("--population-size", type=int, default=50, help="Population size for NES. Must be even.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Sigma for NES.")
    # Hyperparameter Scheduling
    schedule_group = parser.add_argument_group("Hyperparameter Scheduling")
    schedule_group.add_argument("--enable-schedule", action="store_true", help="Enable the two-phase (explore/tune) hyperparameter schedule.")
    schedule_group.add_argument("--tune-sigma", type=float, default=1.0, help="Sigma for the fine-tuning phase.")
    schedule_group.add_argument("--tune-population-size", type=int, default=20, help="Population size for the fine-tuning phase.")
    schedule_group.add_argument("--tuning-patience", type=int, default=5, help="Iterations with no improvement before switching to tuning phase.")
    schedule_group.add_argument("--min-loss-delta", type=float, default=1e-4, help="Minimum change in loss to be considered an improvement.")
    # Stagnation-Resetting Decay Scheduling
    stagnation_group = parser.add_argument_group("Stagnation-Resetting Decay")
    stagnation_group.add_argument("--enable-stagnation-decay", action="store_true", help="Enable decay-and-reset when loss stagnates.")
    stagnation_group.add_argument("--stagnation-patience", type=int, default=30, help="Iterations before forcing a decay.")
    # System
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel processes for evaluation.")
    parser.add_argument("--output-dir", type=str, default="attack_outputs_nes_host", help="Directory to save output images and logs.")
    
    cli_args = parser.parse_args()
    main(cli_args) 