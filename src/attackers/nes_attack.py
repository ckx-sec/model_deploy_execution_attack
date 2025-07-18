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
import time

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
    image_shape = image.shape
    pop_size = args.population_size
    sigma = args.sigma
    
    if pop_size % 2 != 0:
        raise ValueError(f"Population size must be even. Got {pop_size}.")

    half_pop_size = pop_size // 2
    
    # 1. Generate noise and create all mutated images in memory
    # Generate noise with the same shape as the input image (works for both grayscale and color)
    noise_vectors = [np.random.randn(*image_shape) for _ in range(half_pop_size)]
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
    detailed_log_file = None
    attack_image = None
    best_loss_so_far = float('inf')
    best_image_path = None
    total_queries = 0
    start_time = time.time()

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
        
        # --- Generate detailed log file name ---
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        params_to_exclude = {'executable', 'image', 'hooks', 'model', 'golden_image', 'start_adversarial', 'output_dir', 'workers'}
        args_dict = vars(args)
        param_str = "_".join([f"{key}-{val}" for key, val in sorted(args_dict.items()) if key not in params_to_exclude and val is not None and val is not False])
        param_str = re.sub(r'[^a-zA-Z0-9_\-.]', '_', param_str)
        log_filename = f"{timestamp}_{script_name}_{param_str[:100]}.csv"
        detailed_log_path = os.path.join(args.output_dir, log_filename)
        
        detailed_log_file = open(detailed_log_path, 'w')
        detailed_log_file.write("iteration,total_queries,loss,iter_time_s,total_time_s\n")
        print(f"--- Detailed metrics will be logged to: {detailed_log_path} ---")

        # --- Learning Rate and Scheduler Setup ---
        
        # Stagnation-resetting decay state
        stagnation_patience_counter = 0
        iteration_of_last_decay = 0
        total_decay_count = 0
        best_loss_for_stagnation = float('inf')
        if args.enable_stagnation_decay:
            print("--- Stagnation-resetting decay enabled ---")

        # Cosine Annealing with Warm Restarts parameters
        if args.enable_warm_restarts:
            print("--- 'Restart from Best' enabled ---")
            T_i = args.lr_restart_cycle_len
            current_cycle = 0
            iteration_in_cycle = 0
        
        # Hyperparameter schedule state (old logic) - Can be used alongside restarts
        is_tuning_phase = False
        best_loss = float('inf')
        patience_counter = 0
        if args.enable_schedule:
            print("--- Hyperparameter schedule enabled (explore/tune) ---")


        print("--- Preparing environment: Verifying local paths ---")
        # No need to copy files, just verify they exist
        static_files = [args.executable, args.model, args.hooks]
        gdb_script_path = os.path.join(os.path.dirname(__file__), "gdb_script_host.py")
        static_files.append(gdb_script_path) # Assumes gdb_script_host.py is still used by the host runner
        
        for f in static_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required file not found: {f}")

        print("--- Getting target state from golden image ---")
        
        # Load images while preserving their original channels
        golden_image = cv2.imread(args.golden_image, cv2.IMREAD_UNCHANGED)
        if golden_image is None:
            raise FileNotFoundError(f"Could not read golden image: {args.golden_image}")
        
        original_image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
        if original_image is None:
            raise FileNotFoundError(f"Could not read original image: {args.image}")

        # --- Determine processing mode (Grayscale or Color) ---
        # An image is considered grayscale if it has 2 dimensions (height, width)
        is_golden_gray = golden_image.ndim == 2
        is_original_gray = original_image.ndim == 2

        if is_golden_gray and is_original_gray:
            print("--- Detected Grayscale Mode: Both images are single-channel. Processing in 1-channel mode. ---")
            # Images are already single-channel, nothing to do.
        else:
            print("--- Detected Color Mode: At least one image is multi-channel. Processing in 3-channel mode. ---")
            # If an image is grayscale, convert it to color to ensure both are 3-channel.
            if is_golden_gray:
                golden_image = cv2.cvtColor(golden_image, cv2.COLOR_GRAY2BGR)
            if is_original_gray:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        # Use run_attack_iteration for this single run
        is_success_encoding, encoded_golden = cv2.imencode(".png", golden_image)
        if not is_success_encoding:
            raise RuntimeError("Failed to encode golden image.")

        is_golden_ok, target_hooks = run_attack_iteration(encoded_golden.tobytes(), args, workdir, "temp_golden_image.png")
        total_queries += 1

        if not target_hooks:
            raise RuntimeError("Golden run captured no hooks. Cannot proceed.")
        print(f"Target hooks captured: {target_hooks}")

        # Ensure images have the same dimensions; this might recreate original_image
        if original_image.shape[:2] != golden_image.shape[:2]:
            print(f"Warning: Image shapes do not match. Original: {original_image.shape}, Golden: {golden_image.shape}. Resizing original to match golden.")
            original_image = cv2.resize(original_image, (golden_image.shape[1], golden_image.shape[0]))

        # After resize, channels must also match. This handles the case where resizing a gray image to color dimensions doesn't add channels.
        if original_image.ndim != golden_image.ndim:
            print(f"Warning: Channel mismatch after resize. Re-converting original image to match golden image channels.")
            if golden_image.ndim == 3 and original_image.ndim == 2:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            elif golden_image.ndim == 2 and original_image.ndim == 3: # Should not happen with current logic, but for safety
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)


        attack_image = original_image.copy().astype(np.float32)

        # Adam optimizer parameters
        m = np.zeros_like(attack_image, dtype=np.float32)
        v = np.zeros_like(attack_image, dtype=np.float32)
        beta1 = 0.9
        beta2 = 0.999
        epsilon_adam = 1e-8
        adam_step_counter = 0

        for i in range(args.iterations):
            iter_start_time = time.time()
            print(f"--- Iteration {i+1}/{args.iterations} (Total Queries: {total_queries}) ---")
            
            # --- "Restart from Best" Logic ---
            is_restarting_now = False # Flag to indicate if a restart is happening in this iteration
            if args.enable_warm_restarts:
                iteration_in_cycle += 1
                if iteration_in_cycle > T_i:
                    is_restarting_now = True # Mark this iteration as the one performing a restart
                    iteration_in_cycle = 1
                    current_cycle += 1
                    
                    if current_cycle < args.lr_restart_cycles:
                        T_i = int(T_i * args.lr_restart_cycle_mult)
                        
                        if best_image_path and os.path.exists(best_image_path):
                            print(f"--- RESTART FROM BEST: Loading best image from {best_image_path}. ---")
                            loaded_image = cv2.imread(best_image_path, cv2.IMREAD_UNCHANGED)
                            if loaded_image is not None:
                                attack_image = loaded_image.astype(np.float32)
                                # 建议的修改：同时更新扰动的基准图片
                                print("--- Updating base image for perturbation constraint. ---")
                                original_image = loaded_image.copy()
                            else:
                                print(f"Warning: Failed to load best image. Performing warm restart on current image.")
                        else:
                            print(f"--- WARM RESTART (no best image yet): Resetting optimizer at current position. ---")

                        # Reset Adam's momentum
                        m = np.zeros_like(attack_image, dtype=np.float32)
                        v = np.zeros_like(attack_image, dtype=np.float32)
                        adam_step_counter = 0
                        
                        # Reset learning rate decay counters
                        print("--- Resetting learning rate and decay counters. ---")
                        total_decay_count = 0
                        iteration_of_last_decay = i
                        stagnation_patience_counter = 0 # Also reset stagnation
                        best_loss_for_stagnation = float('inf')

                        # FIX: Reset the hyperparameter schedule phase
                        if args.enable_schedule:
                            print("--- Resetting hyperparameter schedule phase. ---")
                            is_tuning_phase = False
                            patience_counter = 0
                            best_loss = float('inf')
                        
                        # FIX: Reset the random number generator state to simulate a true restart
                        print("--- Resetting random number generator state. ---")
                        np.random.seed(None)


                        print(f"--- Starting new cycle {current_cycle+1} with length {T_i}. Adam optimizer reset. ---")
                    else:
                        print("--- All restart cycles completed. ---")
                        # We can disable future restarts by setting a very high cycle length
                        T_i = float('inf')
            
            # --- Standard Learning Rate Decay Logic (Applied universally) ---
            decay_reason = None
            if args.enable_stagnation_decay:
                # Check for scheduled decay
                if (i - iteration_of_last_decay) >= args.lr_decay_steps:
                    decay_reason = f"SCHEDULED ({args.lr_decay_steps} steps passed)"
                # Check for stagnation decay
                elif stagnation_patience_counter >= args.stagnation_patience:
                    decay_reason = f"STAGNATION ({args.stagnation_patience} stagnant iterations)"

                if decay_reason:
                    total_decay_count += 1
                    iteration_of_last_decay = i
                    stagnation_patience_counter = 0 # Reset after decay
                    best_loss_for_stagnation = float('inf') # Reset stagnation baseline

            current_lr = args.learning_rate * (args.lr_decay_rate ** total_decay_count)
            if decay_reason:
                 print(f"DECAY TRIGGERED by {decay_reason}. New LR: {current_lr:.6f}")


            # Use the new, host-based gradient estimator
            grad = estimate_gradient_nes(attack_image, args, target_hooks, workdir)
            total_queries += args.population_size
            
            # Adam Optimizer Update
            adam_step_counter += 1
            t = adam_step_counter
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
                total_queries += 1
                loss = calculate_loss(current_hooks, target_hooks)
            
            iter_time = time.time() - iter_start_time
            total_time_so_far = time.time() - start_time
            print(f"Attack result: {'Success' if is_successful else 'Fail'}. Loss: {loss:.6f}. Iter Time: {iter_time:.2f}s. Total Time: {total_time_so_far:.2f}s")
            
            # Write to the new detailed log file
            detailed_log_file.write(f"{i+1},{total_queries},{loss:.6f},{iter_time:.2f},{total_time_so_far:.2f}\n")
            detailed_log_file.flush()

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
                if loss < best_loss_for_stagnation - args.min_loss_delta:
                    best_loss_for_stagnation = loss
                    stagnation_patience_counter = 0
                else: 
                    stagnation_patience_counter += 1
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
        if detailed_log_file:
            detailed_log_file.close()
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
    parser.add_argument("--iterations", type=int, default=100, help="Maximum number of attack iterations.")
    parser.add_argument("--learning-rate", type=float, default=20.0, help="Initial learning rate for the attack.")
    parser.add_argument("--l-inf-norm", type=float, default=20.0, help="Maximum L-infinity norm for the perturbation.")
    parser.add_argument("--lr-decay-rate", type=float, default=0.97, help="Learning rate decay rate.")
    parser.add_argument("--lr-decay-steps", type=int, default=10, help="Decay learning rate every N steps.")
    # NES Hyperparameters
    parser.add_argument("--population-size", type=int, default=200, help="Population size for NES. Must be even.")
    parser.add_argument("--sigma", type=float, default=0.15, help="Sigma for NES.")
    # Hyperparameter Scheduling
    schedule_group = parser.add_argument_group("Hyperparameter Scheduling")
    schedule_group.add_argument("--enable-schedule", action="store_true", help="Enable the two-phase (explore/tune) hyperparameter schedule.")
    schedule_group.add_argument("--tune-sigma", type=float, default=1.0, help="Sigma for the fine-tuning phase.")
    schedule_group.add_argument("--tune-population-size", type=int, default=20, help="Population size for the fine-tuning phase.")
    schedule_group.add_argument("--tuning-patience", type=int, default=5, help="Iterations with no improvement before switching to tuning phase.")
    schedule_group.add_argument("--min-loss-delta", type=float, default=0.1, help="Minimum change in loss to be considered an improvement for stagnation/tuning.")
    # Stagnation-Resetting Decay
    stagnation_group = parser.add_argument_group("Stagnation-Resetting Decay")
    stagnation_group.add_argument("--enable-stagnation-decay", action="store_true", help="Enable decay-and-reset when loss stagnates.")
    stagnation_group.add_argument("--stagnation-patience", type=int, default=10, help="Iterations before forcing a decay.")
    # Cosine Annealing with Warm Restarts
    warm_restart_group = parser.add_argument_group("Cosine Annealing with Warm Restarts")
    warm_restart_group.add_argument("--enable-warm-restarts", action="store_true", help="Enable 'Restart from Best' strategy.")
    warm_restart_group.add_argument("--lr-restart-cycles", type=int, default=5, help="Number of warm restart cycles.")
    warm_restart_group.add_argument("--lr-restart-cycle-len", type=int, default=50, help="Length of the first cycle in iterations.")
    warm_restart_group.add_argument("--lr-restart-cycle-mult", type=int, default=2, help="Factor to multiply cycle length by after each restart.")
    # System
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel processes for evaluation.")
    parser.add_argument("--output-dir", type=str, default="attack_outputs_nes_host", help="Directory to save output images and logs.")
    
    cli_args = parser.parse_args()
    main(cli_args) 