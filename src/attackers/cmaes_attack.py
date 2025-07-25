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
import cma # Import CMA-ES library
import time

# --- Helper Functions for Host Machine Operations (Unchanged) ---

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

def evaluate_cma_solution(task_args):
    """
    Worker function for CMA-ES.
    It takes a solution vector, creates the image, runs evaluation, and returns the loss.
    """
    solution_vector, original_image_flat, image_shape, temp_image_path, target_hooks, args = task_args

    # 1. Recreate the image from the solution vector
    attack_image_flat = original_image_flat + solution_vector
    
    # 2. Reshape and clip to valid image constraints
    attack_image = np.clip(attack_image_flat.reshape(image_shape), 0, 255)
    
    # 3. Write image to disk for the executable
    _, encoded_image = cv2.imencode(".png", attack_image.astype(np.uint8))
    with open(temp_image_path, 'wb') as f:
        f.write(encoded_image)
        
    # 4. Run the executable and get hooks
    _, hooks = _run_executable_and_parse_hooks(temp_image_path, args)
    
    # 5. Calculate loss
    loss = calculate_loss(hooks, target_hooks)
    return loss


# --- Core Infrastructure (Unchanged) ---

def run_attack_iteration_for_verification(image_content, args, workdir, image_name_on_host):
    """
    Handles a single run for verification purposes.
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


# --- Main Attack Loop (Replaced with CMA-ES) ---

def main(args):
    detailed_log_file = None
    best_attack_image_low_res = None
    best_loss_so_far = float('inf')
    total_queries = 0
    start_time = time.time()

    # Define a smaller, maximum resolution for the attack to avoid memory issues with CMA-ES
    MAX_ATTACK_RESOLUTION_HW = (64, 64)

    # --- Graceful Termination on Ctrl+C (Unchanged) ---
    try:
        os.setpgrp()
    except OSError:
        pass

    def sigint_handler(signum, frame):
        print("\nCtrl+C detected. Forcefully terminating all processes.")
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGINT, sigint_handler)

    # --- Setup Temporary Directory (Unchanged) ---
    temp_dir_base = "/dev/shm" if os.path.exists("/dev/shm") else None
    workdir = tempfile.mkdtemp(prefix="cmaes_host_attack_", dir=temp_dir_base)
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
        param_str = re.sub(r'[^a-zA-Z0-9_\-.]', '_', param_str) # Sanitize
        log_filename = f"{timestamp}_{script_name}_{param_str[:100]}.csv"
        detailed_log_path = os.path.join(args.output_dir, log_filename)
        
        detailed_log_file = open(detailed_log_path, 'w')
        detailed_log_file.write("iteration,total_queries,generation_best_loss,overall_best_loss,iter_time_s,total_time_s\n")
        print(f"--- Detailed metrics will be logged to: {detailed_log_path} ---")


        print("--- Verifying local paths ---")
        static_files = [args.executable, args.model, args.hooks, args.golden_image, args.image]
        for f in static_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required file not found: {f}")

        print("--- Getting target state from golden image ---")
        golden_image = cv2.imread(args.golden_image, cv2.IMREAD_UNCHANGED)
        if golden_image is None: raise FileNotFoundError(f"Could not read golden image: {args.golden_image}")

        # Keep a copy of the original full-resolution image for reference if needed.
        original_image_full_res = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
        if original_image_full_res is None: raise FileNotFoundError(f"Could not read original image: {args.image}")

        # --- Determine processing mode (Grayscale or Color) ---
        is_golden_gray = golden_image.ndim == 2
        is_original_gray = original_image_full_res.ndim == 2

        if is_golden_gray and is_original_gray:
            print("--- Detected Grayscale Mode: Processing in 1-channel mode. ---")
        else:
            print("--- Detected Color Mode: Processing in 3-channel mode. ---")
            if is_golden_gray:
                golden_image = cv2.cvtColor(golden_image, cv2.COLOR_GRAY2BGR)
            if is_original_gray:
                original_image_full_res = cv2.cvtColor(original_image_full_res, cv2.COLOR_GRAY2BGR)


        # Determine attack resolution. If image is smaller than max, use its own size.
        h_orig, w_orig = golden_image.shape[:2]
        max_h, max_w = MAX_ATTACK_RESOLUTION_HW
        if h_orig > max_h or w_orig > max_w:
            print(f"Golden image ({h_orig}x{w_orig}) is larger than max resolution ({max_h}x{max_w}). Resizing down.")
            attack_resolution_hw = (max_h, max_w)
            golden_image = cv2.resize(golden_image, (attack_resolution_hw[1], attack_resolution_hw[0]), interpolation=cv2.INTER_AREA)
        else:
            print(f"Golden image ({h_orig}x{w_orig}) is smaller than or equal to max resolution. Using its original size.")
            attack_resolution_hw = (h_orig, w_orig)


        _, encoded_golden = cv2.imencode(".png", golden_image)
        is_golden_ok, target_hooks = run_attack_iteration_for_verification(encoded_golden.tobytes(), args, workdir, "temp_golden.png")
        total_queries += 1

        if not is_golden_ok or not target_hooks:
            raise RuntimeError("Golden run failed or captured no hooks. Cannot proceed.")
        print(f"Target hooks captured: {target_hooks}")
        
        # Resize original image to match the determined attack resolution for consistency.
        print(f"Resizing original image to match attack resolution: {attack_resolution_hw[0]}x{attack_resolution_hw[1]}")
        original_image = cv2.resize(original_image_full_res, (attack_resolution_hw[1], attack_resolution_hw[0]), interpolation=cv2.INTER_AREA)

        # Ensure channels match after resize, as cv2.resize on a gray image might not produce a gray image if size is tuple for color
        if original_image.ndim != golden_image.ndim:
            if golden_image.ndim == 3 and original_image.ndim == 2:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            elif golden_image.ndim == 2 and original_image.ndim == 3:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)


        image_dimensionality = np.prod(original_image.shape)
        
        # We are optimizing the PERTURBATION, not the image itself.
        # The initial solution is a zero vector, meaning no perturbation.
        initial_perturbation = np.zeros(image_dimensionality, dtype=np.float32)

        # --- Initialize CMA-ES ---
        # The 'bounds' parameter is used to constrain the search space of the perturbation.
        l_inf = args.l_inf_norm
        bounds = [-l_inf, l_inf] 
        
        es = cma.CMAEvolutionStrategy(initial_perturbation, args.sigma, {
            'popsize': args.population_size,
            'bounds': bounds,
            'maxiter': args.iterations,
            'verbose': -9
        })
        
        print(f"--- CMA-ES Initialized. Pop-size: {args.population_size}, Sigma: {args.sigma}, Iterations: {args.iterations} ---")

        # --- Main CMA-ES Loop ---
        iteration = 0
        original_image_flat = original_image.astype(np.float32).flatten()

        # Keep a full-resolution copy for the final step
        original_image_full_res_float = original_image_full_res.astype(np.float32)

        while not es.stop():
            iteration += 1
            iter_start_time = time.time()
            print(f"--- Iteration {iteration}/{args.iterations} (Total Queries: {total_queries}) ---")
            
            # 1. Ask for a new population of solutions (perturbation vectors)
            perturbation_vectors = es.ask()

            # 2. Prepare tasks for parallel evaluation
            tasks = []
            run_id = uuid.uuid4().hex[:12]
            for i, p_vec in enumerate(perturbation_vectors):
                temp_image_name = f"temp_cmaes_{run_id}_{i}.png"
                temp_image_path = os.path.join(workdir, temp_image_name)
                tasks.append((p_vec, original_image_flat, original_image.shape, temp_image_path, target_hooks, args))

            # 3. Evaluate the fitness (loss) of each solution in parallel
            losses = np.zeros(len(perturbation_vectors))
            print(f"--- Evaluating {len(perturbation_vectors)} solutions with {args.workers} workers ---")
            try:
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    results = executor.map(evaluate_cma_solution, tasks)
                    total_queries += len(perturbation_vectors) # Increment queries here
                    for i, loss in enumerate(results):
                        losses[i] = loss
                        print(f"Evaluation progress: {i + 1}/{len(perturbation_vectors)}", end='\r')
                print("\nEvaluation complete.")
            finally:
                # Clean up temporary image files for this generation
                cleanup_pattern = os.path.join(workdir, f"temp_cmaes_{run_id}_*.png")
                remove_files_on_host_batch(cleanup_pattern)

            # 4. Tell the optimizer the results
            es.tell(perturbation_vectors, losses)

            # --- Logging and Saving ---
            current_best_solution = es.result.xbest
            current_best_loss = es.result.fbest

            # --- MODIFICATION: Create and save the LOW-RESOLUTION attack image ---
            # This is the image that was actually evaluated and is known to be effective.
            # The perturbation is clipped to the L-inf norm constraint for correctness.
            best_perturbation_low_res = np.clip(current_best_solution, -l_inf, l_inf).reshape(original_image.shape)
            final_attack_image_low_res = np.clip(original_image.astype(np.float32) + best_perturbation_low_res, 0, 255).astype(np.uint8)

            iter_time = time.time() - iter_start_time
            total_time_so_far = time.time() - start_time
            print(f"Best loss gen: {np.min(losses):.6f}. Overall best: {current_best_loss:.6f}. Iter Time: {iter_time:.2f}s. Total Time: {total_time_so_far:.2f}s")
            detailed_log_file.write(f"{iteration},{total_queries},{np.min(losses):.6f},{current_best_loss:.6f},{iter_time:.2f},{total_time_so_far:.2f}\n")
            detailed_log_file.flush()

            # Save latest and best images (now in low resolution, which is the validated result)
            latest_image_path = os.path.join(args.output_dir, "latest_attack_image_cmaes_host_lowres.png")
            cv2.imwrite(latest_image_path, final_attack_image_low_res)

            if current_best_loss < best_loss_so_far:
                best_loss_so_far = current_best_loss
                best_attack_image_low_res = final_attack_image_low_res # Keep track of the best raw image
                print(f"New best loss found: {best_loss_so_far:.6f}. Saving best low-res image.")
                best_image_path = os.path.join(args.output_dir, "best_attack_image_cmaes_host_lowres.png")
                cv2.imwrite(best_image_path, best_attack_image_low_res)
            
            # Check for success condition on the downscaled image
            _, encoded_image = cv2.imencode(".png", best_attack_image_low_res)
            is_successful, _ = run_attack_iteration_for_verification(encoded_image.tobytes(), args, workdir, "temp_verify.png")
            total_queries += 1 # Increment for verification check

            if is_successful:
                print("\n--- Low-resolution attack successful! ---")
                print("--- Now attempting to upscale perturbation to full resolution... ---")

                # Upscale the perturbation
                h_full, w_full = original_image_full_res.shape[:2]
                best_perturbation_high_res = cv2.resize(best_perturbation_low_res, (w_full, h_full), interpolation=cv2.INTER_CUBIC)

                # Apply to full-res original image
                final_attack_image_high_res = np.clip(original_image_full_res_float + best_perturbation_high_res, 0, 255).astype(np.uint8)

                # Final verification
                _, encoded_high_res = cv2.imencode(".png", final_attack_image_high_res)
                is_high_res_successful, _ = run_attack_iteration_for_verification(encoded_high_res.tobytes(), args, workdir, "temp_verify_highres.png")
                total_queries += 1 # Increment for final high-res check
                
                if is_high_res_successful:
                    print("\n--- SUCCESS! Upscaled attack image is verified to be adversarial. ---")
                    successful_image_path = os.path.join(args.output_dir, "successful_attack_image_cmaes_host_FULL_RES.png")
                    cv2.imwrite(successful_image_path, final_attack_image_high_res)
                    print(f"Full-resolution adversarial image saved to: {successful_image_path}")
                else:
                    print("\n--- WARNING: Upscaled attack image is NOT adversarial. ---")
                    print("The perturbation did not transfer to the higher resolution.")
                    failed_image_path = os.path.join(args.output_dir, "failed_upscaled_attack_image_cmaes_host.png")
                    cv2.imwrite(failed_image_path, final_attack_image_high_res)
                    print(f"The failed upscaled image has been saved for analysis: {failed_image_path}")

                # Save the successful low-res version anyway for comparison
                successful_lowres_path = os.path.join(args.output_dir, "successful_attack_image_cmaes_host_LOW_RES.png")
                cv2.imwrite(successful_lowres_path, best_attack_image_low_res)
                print(f"The successful low-resolution base image saved to: {successful_lowres_path}")

                break

        print("\n--- CMA-ES optimization finished ---")
        es.result_pretty()

    except (FileNotFoundError, RuntimeError, KeyboardInterrupt) as e:
        print(f"\nAn error or interrupt occurred: {e}")
        # Also save the low-res image on interrupt
        if best_attack_image_low_res is not None:
            print("Saving the last best low-resolution image...")
            interrupted_image_path = os.path.join(args.output_dir, "interrupted_attack_image_cmaes_host_lowres.png")
            cv2.imwrite(interrupted_image_path, best_attack_image_low_res)
            print(f"Last best image saved to: {interrupted_image_path}")
            
    finally:
        if detailed_log_file: detailed_log_file.close()
        if workdir and os.path.exists(workdir):
            shutil.rmtree(workdir)
            print(f"Temporary directory {workdir} cleaned up.")
        print("Cleanup finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A grey-box adversarial attack using CMA-ES (Host Version).")
    # File Paths
    parser.add_argument("--executable", required=True, help="Local path to the target executable.")
    parser.add_argument("--image", required=True, help="Local path to the initial image to be attacked.")
    parser.add_argument("--hooks", required=True, help="Local path to the JSON file defining hook points.")
    parser.add_argument("--model", required=True, help="Local path to the model file (e.g., .onnx).")
    parser.add_argument("--golden-image", required=True, help="Local path to the image that produces the target state.")
    # Attack Hyperparameters
    parser.add_argument("--iterations", type=int, default=1000, help="Maximum number of attack iterations (generations for CMA-ES).")
    parser.add_argument("--l-inf-norm", type=float, default=20.0, help="Maximum L-infinity norm for the perturbation (bounds for CMA-ES).")
    # CMA-ES Hyperparameters
    parser.add_argument("--population-size", type=int, default=20, help="Population size (lambda) for CMA-ES.")
    parser.add_argument("--sigma", type=float, default=5.0, help="Initial standard deviation (step size) for CMA-ES.")
    # System
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel processes for evaluation.")
    parser.add_argument("--output-dir", type=str, default="attack_outputs_cmaes_host", help="Directory to save output images and logs.")
    
    cli_args = parser.parse_args()
    main(cli_args) 