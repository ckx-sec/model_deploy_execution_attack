import numpy as np
import cv2
import subprocess
import re
import os
import argparse
import json
import uuid
from concurrent.futures import ProcessPoolExecutor
import io
import tarfile

# --- New Helper Functions for Batched Docker Operations ---

def copy_multiple_files_to_container_via_tar(files_data, dest_dir, container_name):
    """
    Copies multiple files into a container in a single 'docker cp' operation using a tar stream.
    This is significantly faster than one 'docker exec' per file.
    
    :param files_data: A list of tuples, where each tuple is (filename, file_data_as_bytes).
    :param dest_dir: The destination directory inside the container (e.g., '/app').
    :param container_name: The name of the Docker container.
    """
    tar_stream = io.BytesIO()
    # Create an in-memory tar archive
    with tarfile.open(fileobj=tar_stream, mode='w') as tar:
        for filename, data in files_data:
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data))
    
    tar_stream.seek(0)
    
    # Use 'docker cp -' which accepts a tar archive from stdin and extracts it
    # to the destination directory in the container.
    command = f"docker cp - {container_name}:{dest_dir}"
    try:
        subprocess.run(
            command,
            shell=True,
            check=True,
            input=tar_stream.read(),
            capture_output=True,
            timeout=60  # Increased timeout for potentially large batches
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else "Timeout during batch copy"
        raise RuntimeError(f"Docker cp command failed for batch tar stream: {stderr}")

def remove_files_in_container_batch(file_pattern, container_name):
    """
    Removes multiple files matching a pattern inside a container using a single command.
    
    :param file_pattern: A shell glob pattern (e.g., "/app/temp_nes_*.png").
    :param container_name: The name of the Docker container.
    """
    try:
        # Use 'sh -c' to allow shell wildcards like '*' to be expanded inside the container.
        command = f"docker exec {container_name} sh -c 'rm -f {file_pattern}'"
        subprocess.run(
            command, shell=True, check=False, capture_output=True, text=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        # Log this but don't re-raise, as cleanup failure is not critical.
        print(f"Warning: Timeout while trying to batch remove '{file_pattern}' in container.")

# --- Worker Functions for Parallel Execution ---

def copy_file_to_container(local_path, container_path, container_name):
    """A helper function to copy a single local file into a running Docker container."""
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")
    try:
        subprocess.run(
            f"docker cp {local_path} {container_name}:{container_path}",
            shell=True, check=True, capture_output=True, text=True, timeout=15
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = e.stderr if hasattr(e, 'stderr') else "Timeout during copy"
        raise RuntimeError(f"Docker cp command failed for '{local_path}': {stderr}")

def copy_data_to_container(image_data, container_path, container_name):
    """A helper function to copy image data from memory into a running Docker container."""
    try:
        command = f"docker exec -i {container_name} sh -c 'cat > {container_path}'"
        subprocess.run(
            command,
            shell=True, check=True, input=image_data, capture_output=True, timeout=15
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = e.stderr if hasattr(e, 'stderr') else "Timeout during copy"
        raise RuntimeError(f"Docker exec command failed for stream data: {stderr}")

def remove_file_in_container(container_path, container_name):
    """A helper function to remove a single file inside a running Docker container."""
    try:
        command = f"docker exec {container_name} rm {container_path}"
        subprocess.run(
            command, shell=True, check=False, capture_output=True, text=True, timeout=15
        )
    except subprocess.TimeoutExpired:
        print(f"Warning: Timeout while trying to remove '{container_path}' in container.")

def _run_executable_and_parse_hooks(image_path_in_container, args):
    """
    Runs the executable inside the container for a given image path and parses hook results.
    This is the core execution logic, separated from file I/O.
    """
    script_path = os.path.join(os.path.dirname(__file__), "run_gdb_docker.sh")
    executable_in_container = os.path.join(args.workdir, os.path.basename(args.executable))
    model_in_container = os.path.join(args.workdir, os.path.basename(args.model))

    command = [script_path, executable_in_container, model_in_container, image_path_in_container]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return False, []

    hooked_values = []
    is_successful = False
    full_output = result.stdout + "\n" + result.stderr
    output_lines = full_output.splitlines()

    for line in output_lines:
        if "true" in line.lower() and "HOOK_RESULT" not in line: is_successful = True
        if "HOOK_RESULT" in line:
            val_str_match = re.search(r'value=(.*)', line)
            if not val_str_match: continue
            val_str = val_str_match.group(1).strip()
            try:
                if val_str.startswith('{'):
                    float_match = re.search(r'f\s*=\s*([-\d.e+]+)', val_str)
                    if float_match: hooked_values.append(float(float_match.group(1)))
                else:
                    hooked_values.append(float(val_str))
            except (ValueError, TypeError): pass
                
    return is_successful, hooked_values

def evaluate_mutation_in_container(task_args):
    """
    Evaluates a single mutated image that is *already present* in the container
    by running the executable and calculating the loss.
    This is the worker function for the process pool. It performs no Docker I/O.
    """
    image_path_in_container, target_hooks, args = task_args
    
    # This is the step the user mentioned cannot be batched.
    # We run the executable for the specific image.
    _, hooks = _run_executable_and_parse_hooks(image_path_in_container, args)
    
    # Calculate the loss based on the results.
    loss = calculate_loss(hooks, target_hooks)
    return loss

# --- Core Infrastructure (Refactored) ---

def run_attack_iteration(image_content, args, image_name_in_container=None):
    """
    Handles a single run of the executable for a given image, including I/O.
    Accepts either a file path (str) or raw image data (bytes).
    This function is now primarily used for single, non-batched runs like
    the initial golden image evaluation.
    """
    workdir = args.workdir

    # Determine image name and path in container, then copy content
    if isinstance(image_content, str):  # It's a file path
        image_name = os.path.basename(image_content) if image_name_in_container is None else image_name_in_container
        image_in_container = os.path.join(workdir, image_name)
        copy_file_to_container(image_content, image_in_container, args.container_name)
    elif isinstance(image_content, bytes):  # It's raw image data from memory
        if image_name_in_container is None:
            raise ValueError("image_name_in_container must be provided for byte stream content.")
        image_in_container = os.path.join(workdir, image_name_in_container)
        copy_data_to_container(image_content, image_in_container, args.container_name)
    else:
        raise TypeError("image_content must be a file path (str) or image data (bytes)")

    is_successful, hooked_values = _run_executable_and_parse_hooks(image_in_container, args)

    # Clean up the temporary file from the container
    remove_file_in_container(image_in_container, args.container_name)
    
    return is_successful, hooked_values

def calculate_loss(current_hooks, target_hooks, margin=0.0, weights=None):
    """
    Calculates the loss by comparing hook pairs from the current state to the target state.
    This loss function is designed for scenarios where each pair represents a comparison
    against a threshold, aiming to replicate the comparison's outcome (e.g., value > threshold).

    :param current_hooks: List of hook values from the current (attack) image.
                          Expected format: [v1_current, t1_current, v2_current, t2_current, ...]
    :param target_hooks: List of hook values from the golden image, which define the
                         target relationships. Expected format: [v1_target, t1_target, ...]
    :param margin: A safety margin to make the attack more robust for inequality checks.
    :param weights: (Optional) A list of weights for the loss of each hook pair. Defaults to 1.
    :return: A float representing the total loss, averaged over the number of pairs.
    """
    if not isinstance(current_hooks, list) or not isinstance(target_hooks, list) or not current_hooks:
        return float('inf')
    if len(current_hooks) != len(target_hooks) or len(current_hooks) % 2 != 0:
        print(f"Warning: Hook count mismatch or not even. Current: {len(current_hooks)}, Target: {len(target_hooks)}")
        return float('inf')

    current = np.array(current_hooks, dtype=np.float32)
    target = np.array(target_hooks, dtype=np.float32)
    return np.mean((current - target) ** 2)


# --- NES Gradient Estimator (Optimized with Batching) ---

def estimate_gradient_nes(image, args, target_hooks):
    """
    Estimates the gradient using NES with Antithetic Sampling.
    This version is optimized to use batched Docker operations.
    """
    run_id = uuid.uuid4().hex[:12]  # A unique ID for this batch of evaluations
    h, w = image.shape
    pop_size = args.population_size
    sigma = args.sigma
    
    if pop_size % 2 != 0:
        raise ValueError(f"Population size must be even. Got {pop_size}.")

    half_pop_size = pop_size // 2
    
    # 1. Generate noise and create all mutated images in memory
    noise_vectors = [np.random.randn(h, w) for _ in range(half_pop_size)]
    mutations_data_for_tar = []
    tasks = []

    for i, noise in enumerate(noise_vectors):
        # Create positive and negative mutations
        mutant_pos = image + sigma * noise
        mutant_neg = image - sigma * noise
        
        # Encode images to PNG format in memory
        _, encoded_pos = cv2.imencode(".png", mutant_pos.astype(np.uint8))
        _, encoded_neg = cv2.imencode(".png", mutant_neg.astype(np.uint8))

        # Generate unique names for the files inside the container using the run_id
        unique_id_pos = f"{run_id}_{i}_pos"
        unique_id_neg = f"{run_id}_{i}_neg"
        
        fname_pos = f"temp_nes_{unique_id_pos}.png"
        fname_neg = f"temp_nes_{unique_id_neg}.png"
        
        mutations_data_for_tar.append((fname_pos, encoded_pos.tobytes()))
        mutations_data_for_tar.append((fname_neg, encoded_neg.tobytes()))

        # Prepare tasks for the process pool
        path_pos = os.path.join(args.workdir, fname_pos)
        path_neg = os.path.join(args.workdir, fname_neg)
        tasks.append((path_pos, target_hooks, args))
        tasks.append((path_neg, target_hooks, args))

    # --- BATCH I/O OPERATIONS ---
    try:
        # 2. Copy all generated images to the container in a single operation
        print(f"--- Copying {len(mutations_data_for_tar)} images to container via TAR stream ---")
        copy_multiple_files_to_container_via_tar(mutations_data_for_tar, args.workdir, args.container_name)

        # 3. Run evaluations in parallel. The worker function is now I/O-free.
        print(f"--- Evaluating {pop_size} mutations (using {half_pop_size} antithetic pairs) with {args.workers} workers ---")
        losses = np.zeros(pop_size)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # We use evaluate_mutation_in_container, which does no I/O
            results = executor.map(evaluate_mutation_in_container, tasks)
            for i, loss in enumerate(results):
                losses[i] = loss
                print(f"Evaluation progress: {i + 1}/{pop_size}", end='\r')
        print("\nEvaluation complete.")

    finally:
        # 4. Clean up all temporary files from the container in a single operation
        print("--- Batch removing temporary images from container ---")
        cleanup_pattern = f"{args.workdir}/temp_nes_{run_id}_*.png"
        remove_files_in_container_batch(cleanup_pattern, args.container_name)

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

# --- Main Attack Loop (Largely Unchanged) ---

def main(args):
    loss_log_file = None
    attack_image = None
    best_loss_so_far = float('inf')
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"--- NES Attacker (Batched) Started. Outputs will be in: {args.output_dir} ---")

        loss_log_path = os.path.join(args.output_dir, "nes_loss_log_batched.csv")
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

        print("--- Setting up environment: Copying static files to container ---")
        static_files = [args.executable, args.model, args.hooks]
        static_files.append(os.path.join(os.path.dirname(__file__), "gdb_script.py"))
        
        for f in static_files:
            dest_name = "hooks.json" if f.endswith(".json") else "gdb_script.py" if f.endswith(".py") else os.path.basename(f)
            copy_file_to_container(f, os.path.join(args.workdir, dest_name), args.container_name)

        print("--- Getting target state from golden image ---")
        # Use the original run_attack_iteration for this single run, which handles its own I/O and cleanup
        is_golden_ok, target_hooks = run_attack_iteration(args.golden_image, args, "temp_golden_image.png")
        if not is_golden_ok or not target_hooks:
            raise RuntimeError("Golden run failed or captured no hooks. Cannot proceed.")

        original_image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
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

            # Use the new, batched gradient estimator
            grad = estimate_gradient_nes(attack_image, args, target_hooks)
            
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
                is_successful, current_hooks, loss = False, [], float('inf')
            else:
                # Use run_attack_iteration for the final check of the main attack image
                is_successful, current_hooks = run_attack_iteration(encoded_image.tobytes(), args, "temp_attack_image.png")
                loss = calculate_loss(current_hooks, target_hooks)
            
            print(f"Attack result: {'Success' if is_successful else 'Fail'}. Internal state loss: {loss:.6f}")
            loss_log_file.write(f"{i+1},{loss:.6f}\n")
            loss_log_file.flush()

            # --- Save latest and best images ---
            latest_image_path = os.path.join(args.output_dir, "latest_attack_image_nes_batched.png")
            cv2.imwrite(latest_image_path, attack_image.astype(np.uint8))

            if loss < best_loss_so_far:
                best_loss_so_far = loss
                print(f"New best loss found: {loss:.6f}. Saving best image.")
                best_image_path = os.path.join(args.output_dir, "best_attack_image_nes_batched.png")
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
                successful_image_path = os.path.join(args.output_dir, "successful_attack_image_nes_batched.png")
                cv2.imwrite(successful_image_path, attack_image.astype(np.uint8))
                print(f"Adversarial image saved to: {successful_image_path}")
                break

    except (KeyboardInterrupt, FileNotFoundError, RuntimeError) as e:
        print(f"\nAn error occurred: {e}")
        if attack_image is not None:
            print("Interrupt received. Saving the last generated image...")
            interrupted_image_path = os.path.join(args.output_dir, "interrupted_attack_image_nes_batched.png")
            cv2.imwrite(interrupted_image_path, attack_image.astype(np.uint8))
            print(f"Last image saved to: {interrupted_image_path}")
    finally:
        if loss_log_file:
            loss_log_file.close()
        print("Cleanup finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A grey-box adversarial attack using NES (Batched Version).")
    # File Paths
    parser.add_argument("--executable", required=True, help="Local path to the target executable.")
    parser.add_argument("--image", required=True, help="Local path to the initial image to be attacked.")
    parser.add_argument("--hooks", required=True, help="Local path to the JSON file defining hook points.")
    parser.add_argument("--model", required=True, help="Local path to the model file (e.g., .onnx).")
    parser.add_argument("--golden-image", required=True, help="Local path to the image that produces the target state.")
    # Docker Config
    parser.add_argument("--container-name", default="greybox_attacker_container", help="Name of the running Docker container.")
    parser.add_argument("--workdir", default="/app", help="Working directory inside the Docker container.")
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
    parser.add_argument("--output-dir", type=str, default="attack_outputs_nes", help="Directory to save output images and logs.")
    
    cli_args = parser.parse_args()
    main(cli_args) 