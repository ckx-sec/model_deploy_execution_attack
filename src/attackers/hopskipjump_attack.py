import numpy as np
import cv2
import subprocess
import re
import os
import signal
import sys
import argparse
import json
import shutil
import tempfile
import uuid
import time
from concurrent.futures import ProcessPoolExecutor

# --- Core Host Execution Logic & Helpers (Reused) ---

def _run_executable_and_parse_success(image_path_on_host, args):
    """
    Runs the executable on the host for a given image path and parses the success state.
    This is the core decision oracle for the attack.
    """
    script_path = os.path.join(os.path.dirname(__file__), "run_gdb_host.sh") 
    executable_on_host = args.executable
    model_on_host = args.model

    command = [
        '/bin/bash',
        script_path,
        os.path.abspath(executable_on_host),
        os.path.abspath(model_on_host),
        os.path.abspath(image_path_on_host),
        os.path.abspath(args.hooks)
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = e.stderr if hasattr(e, 'stderr') else "Timeout or error during execution"
        print(f"Error running host executable for '{image_path_on_host}': {stderr}")
        return False

    full_output = result.stdout + "\n" + result.stderr
    is_successful = any("true" in line.lower() for line in full_output.splitlines())
    return is_successful

def check_image_is_adversarial(image_content, args, workdir):
    """
    A helper that takes raw image bytes, writes them to a temporary file,
    runs the executable, and returns the boolean success status.
    """
    image_name_on_host = f"temp_check_{uuid.uuid4().hex[:12]}.png"
    image_path_on_host = os.path.join(workdir, image_name_on_host)
    with open(image_path_on_host, 'wb') as f:
        f.write(image_content)

    is_successful = _run_executable_and_parse_success(image_path_on_host, args)
    os.remove(image_path_on_host)
    return is_successful


# --- HopSkipJumpAttack Core Functions ---

def binary_search_to_boundary(original_image, adversarial_image, args, workdir, query_counter_ref):
    """
    Performs a binary search between two images to find a point on the decision boundary.
    `query_counter_ref` is a list [count] to be mutable.
    """
    low_bound = np.zeros_like(original_image)
    high_bound = np.ones_like(original_image)
    
    # Search for a valid boundary point
    for _ in range(10): # Try to find a valid starting high bound
        mid_image = (high_bound * adversarial_image + (1 - high_bound) * original_image)
        _, encoded = cv2.imencode(".png", np.clip(mid_image, 0, 255).astype(np.uint8))
        query_counter_ref[0] += 1
        if _check_image_is_adversarial_sequential(encoded.tobytes(), args, workdir):
            low_bound = high_bound
            break
        high_bound *= 0.9
    else:
        return adversarial_image # Could not find boundary, return original adv

    # Binary search for the boundary
    for _ in range(args.binary_search_steps):
        mid_bound = (low_bound + high_bound) / 2
        mid_image = (mid_bound * adversarial_image + (1 - mid_bound) * original_image)
        _, encoded = cv2.imencode(".png", np.clip(mid_image, 0, 255).astype(np.uint8))
        query_counter_ref[0] += 1
        # The original check_image_is_adversarial is sequential, we need a parallel version for the gradient estimation
        if _check_image_is_adversarial_sequential(encoded.tobytes(), args, workdir):
            high_bound = mid_bound
        else:
            low_bound = mid_bound

    final_boundary_image = (high_bound * adversarial_image + (1 - high_bound) * original_image)
    return final_boundary_image

def _check_image_is_adversarial_sequential(image_content, args, workdir):
    """
    Sequential version of the check, for single queries like in binary search.
    """
    image_name_on_host = f"temp_check_{uuid.uuid4().hex[:12]}.png"
    image_path_on_host = os.path.join(workdir, image_name_on_host)
    with open(image_path_on_host, 'wb') as f:
        f.write(image_content)

    is_successful = _run_executable_and_parse_success(image_path_on_host, args)
    os.remove(image_path_on_host)
    return is_successful

def _evaluate_probe_on_host(task_args):
    """
    Worker function for parallel gradient estimation.
    Takes a path, runs the check, and returns a boolean.
    """
    image_path_on_host, args = task_args
    return _run_executable_and_parse_success(image_path_on_host, args)

def estimate_gradient_at_boundary(boundary_point, original_image, args, workdir, query_counter_ref):
    """
    Estimates the gradient (normal to the decision boundary) at a given point.
    This version is parallelized.
    """
    num_queries = args.num_grad_queries
    delta = args.grad_estimation_delta
    
    grad_sum = np.zeros_like(original_image, dtype=np.float32)
    
    print(f"--- Estimating gradient with {num_queries} parallel queries (using {args.workers} workers)... ---")
    
    run_id = uuid.uuid4().hex[:12]
    u_vectors = []
    tasks = []
    probe_image_paths = []
    
    # 1. Prepare all probes
    for i in range(num_queries):
        u = np.random.randn(*original_image.shape).astype(np.float32)
        u /= np.linalg.norm(u)
        u_vectors.append(u)
        
        perturbed_image = boundary_point + delta * u
        _, encoded = cv2.imencode(".png", np.clip(perturbed_image, 0, 255).astype(np.uint8))
        
        image_path = os.path.join(workdir, f"probe_{run_id}_{i}.png")
        probe_image_paths.append(image_path)
        with open(image_path, 'wb') as f:
            f.write(encoded)
        
        tasks.append((image_path, args))

    # 2. Run probes in parallel
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(_evaluate_probe_on_host, tasks))
        
        query_counter_ref[0] += num_queries # Increment total queries at once
        
        # 3. Aggregate results
        for i, is_adv in enumerate(results):
            if is_adv:
                grad_sum -= u_vectors[i]
            else:
                grad_sum += u_vectors[i]
    finally:
        # 4. Cleanup
        for path in probe_image_paths:
            try:
                os.remove(path)
            except OSError:
                pass

    print(" Done.")
    return grad_sum / num_queries


# --- Main Attack Loop ---

def main(args):
    detailed_log_file = None
    adversarial_image = None
    best_l2_distance = float('inf')
    start_time = time.time()

    try:
        os.setpgrp()
    except OSError:
        pass

    def sigint_handler(signum, frame):
        print("\nCtrl+C detected. Forcefully terminating all processes.")
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGINT, sigint_handler)
    
    temp_dir_base = "/dev/shm" if os.path.exists("/dev/shm") else None
    workdir = tempfile.mkdtemp(prefix="hopskip_host_attack_", dir=temp_dir_base)
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # --- Generate detailed log file name ---
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        params_to_exclude = {'executable', 'image', 'hooks', 'model', 'start_adversarial', 'output_dir', 'workers'}
        args_dict = vars(args)
        param_str = "_".join([f"{key}-{val}" for key, val in sorted(args_dict.items()) if key not in params_to_exclude and val is not None and val is not False])
        param_str = re.sub(r'[^a-zA-Z0-9_\-.]', '_', param_str) # Sanitize
        log_filename = f"{timestamp}_{script_name}_{param_str[:100]}.csv"
        detailed_log_path = os.path.join(args.output_dir, log_filename)
        
        detailed_log_file = open(detailed_log_path, 'w')
        detailed_log_file.write("iteration,total_queries,l2_distance,l_inf_norm,iter_time_s,total_time_s\n")
        print(f"--- Detailed metrics will be logged to: {detailed_log_path} ---")


        print("--- Verifying local paths ---")
        static_files = [args.executable, args.model, args.hooks, args.start_adversarial, args.image]
        for f in static_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required file not found: {f}")

        print("--- Loading images ---")
        original_image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED).astype(np.float32)
        adversarial_image = cv2.imread(args.start_adversarial, cv2.IMREAD_UNCHANGED).astype(np.float32)

        if original_image is None: raise FileNotFoundError(f"Could not load original image from {args.image}")
        if adversarial_image is None: raise FileNotFoundError(f"Could not load starting adversarial image from {args.start_adversarial}")

        # --- Determine processing mode (Grayscale or Color) ---
        is_orig_gray = original_image.ndim == 2
        is_adv_gray = adversarial_image.ndim == 2

        if is_orig_gray and is_adv_gray:
            print("--- Detected Grayscale Mode: Processing in 1-channel mode. ---")
        else:
            print("--- Detected Color Mode: Processing in 3-channel mode. ---")
            if is_orig_gray:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            if is_adv_gray:
                adversarial_image = cv2.cvtColor(adversarial_image, cv2.COLOR_GRAY2BGR)

        # 自动处理尺寸不一致
        if original_image.shape[:2] != adversarial_image.shape[:2]:
            print(f"[Auto Resize] Original image shape {original_image.shape} != start-adversarial image shape {adversarial_image.shape}, resizing to match.")
            # Resize original image to match adversarial to keep the adversarial property
            adversarial_image = cv2.resize(adversarial_image, (original_image.shape[1], original_image.shape[0]))

        # Ensure channels match after resize
        if original_image.ndim != adversarial_image.ndim:
             if original_image.ndim == 3 and adversarial_image.ndim == 2:
                 adversarial_image = cv2.cvtColor(adversarial_image, cv2.COLOR_GRAY2BGR)
             elif original_image.ndim == 2 and adversarial_image.ndim == 3:
                 adversarial_image = cv2.cvtColor(adversarial_image, cv2.COLOR_BGR2GRAY)


        print("--- Verifying initial image states ---")
        _, encoded_orig = cv2.imencode(".png", original_image.astype(np.uint8))
        query_counter_ref = [1] # Start with 1
        orig_is_adv = _check_image_is_adversarial_sequential(encoded_orig.tobytes(), args, workdir)
        if orig_is_adv: raise RuntimeError("Original image is already adversarial ('true').")
        print("Original image is correctly non-adversarial ('false').")

        _, encoded_adv = cv2.imencode(".png", adversarial_image.astype(np.uint8))
        query_counter_ref[0] += 1
        start_is_adv = _check_image_is_adversarial_sequential(encoded_adv.tobytes(), args, workdir)
        if not start_is_adv: raise RuntimeError("Starting adversarial image is not adversarial ('false').")
        print("Starting adversarial image is correctly adversarial ('true').")

        for i in range(args.iterations):
            iter_start_time = time.time()
            print(f"--- Iteration {i+1}/{args.iterations} (Total Queries: {query_counter_ref[0]}) ---")

            # 1. Hop: Binary search to find the boundary
            boundary_point = binary_search_to_boundary(original_image, adversarial_image, args, workdir, query_counter_ref)

            # 2. Skip Part 1: Estimate gradient at the boundary
            grad = estimate_gradient_at_boundary(boundary_point, original_image, args, workdir, query_counter_ref)

            # 3. Skip Part 2: Take step with decay
            # Corrected HopSkipJump step logic (Second Attempt):
            # The goal is to move from the boundary point towards the original image,
            # but constrained to the tangent plane of the decision boundary.
            # The gradient is normal to this plane.

            move_direction = original_image - boundary_point
            
            # Normalize the gradient to get the unit normal vector of the boundary
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-9: # Avoid division by zero
                unit_grad = grad
            else:
                unit_grad = grad / grad_norm

            # Project the move_direction onto the gradient vector
            proj_length = np.vdot(move_direction, unit_grad)
            proj_vector = proj_length * unit_grad
            
            # The corrected direction is the component of move_direction that is orthogonal to the gradient
            # This moves along the boundary towards the original image
            corrected_direction = move_direction - proj_vector
            
            # Normalize the final direction
            corrected_direction_norm = np.linalg.norm(corrected_direction)
            if corrected_direction_norm > 1e-9:
                 corrected_direction /= corrected_direction_norm

            # Use a more stable step size
            step_size = args.step_size * (args.step_size_decay ** i)
            
            candidate_image = boundary_point + step_size * corrected_direction

            # 4. Project to valid range
            candidate_image = np.clip(candidate_image, 0, 255)

            # 5. Jump: Check if candidate is adversarial, if so, project it back to the boundary
            _, encoded_candidate = cv2.imencode(".png", candidate_image.astype(np.uint8))
            query_counter_ref[0] += 1
            if _check_image_is_adversarial_sequential(encoded_candidate.tobytes(), args, workdir):
                print("Step successful, projecting new candidate back to the boundary (Jump).")
                adversarial_image = binary_search_to_boundary(original_image, candidate_image, args, workdir, query_counter_ref)
            else:
                print("Step failed, candidate is not adversarial. No update in this iteration.")
                # If the step failed, we don't update `adversarial_image`.
                # The next iteration's "Hop" will start from the same point as this one.


            # Logging and saving
            l2_dist = np.linalg.norm(adversarial_image - original_image)
            linf_dist = np.max(np.abs(adversarial_image - original_image))
            
            iter_time = time.time() - iter_start_time
            total_time_so_far = time.time() - start_time
            print(f"L2: {l2_dist:.4f}, L-inf: {linf_dist:.4f}. Iter Time: {iter_time:.2f}s. Total Time: {total_time_so_far:.2f}s")

            detailed_log_file.write(f"{i+1},{query_counter_ref[0]},{l2_dist:.6f},{linf_dist:.6f},{iter_time:.2f},{total_time_so_far:.2f}\n")
            detailed_log_file.flush()

            latest_image_path = os.path.join(args.output_dir, "latest_attack_image_hopskip_host.png")
            cv2.imwrite(latest_image_path, adversarial_image.astype(np.uint8))

            if l2_dist < best_l2_distance:
                best_l2_distance = l2_dist
                print(f"New best L2 distance found: {l2_dist:.4f}. Saving best image.")
                best_image_path = os.path.join(args.output_dir, "best_attack_image_hopskip_host.png")
                cv2.imwrite(best_image_path, adversarial_image.astype(np.uint8))

    except (FileNotFoundError, RuntimeError, KeyboardInterrupt) as e:
        print(f"\nAn error or interrupt occurred: {e}")
        if adversarial_image is not None:
            interrupted_image_path = os.path.join(args.output_dir, "interrupted_attack_image_hopskip_host.png")
            cv2.imwrite(interrupted_image_path, adversarial_image.astype(np.uint8))
    finally:
        if detailed_log_file: detailed_log_file.close()
        if workdir and os.path.exists(workdir): shutil.rmtree(workdir)
        print("Cleanup finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A grey-box adversarial attack using HopSkipJump (Host Version).")
    # File Paths
    parser.add_argument("--executable", required=True, help="Local path to the target executable.")
    parser.add_argument("--image", required=True, help="Local path to the initial NON-ADVERSARIAL image.")
    parser.add_argument("--hooks", required=True, help="Local path to the JSON file defining hook points (needed by runner).")
    parser.add_argument("--model", required=True, help="Local path to the model file (e.g., .onnx).")
    parser.add_argument("--start-adversarial", required=True, help="Local path to an image that is already adversarial ('true').")
    # Attack Hyperparameters
    parser.add_argument("--iterations", type=int, default=50, help="Maximum number of attack iterations (steps).")
    parser.add_argument("--binary-search-steps", type=int, default=10, help="Number of binary search steps to find the boundary.")
    parser.add_argument("--num-grad-queries", type=int, default=200, help="Number of samples to estimate gradient.")
    parser.add_argument("--grad-estimation-delta", type=float, default=0.1, help="Radius for gradient estimation probes.")
    parser.add_argument("--step-size", type=float, default=1.0, help="Initial step size for moving away from the boundary. Note: This is now a fixed pixel value, not a factor.")
    parser.add_argument("--step-size-decay", type=float, default=0.99, help="Decay rate for the step size factor after each iteration.")
    # System
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel processes for gradient estimation.")
    parser.add_argument("--output-dir", type=str, default="attack_outputs_hopskip_host", help="Directory to save output images and logs.")
    
    cli_args = parser.parse_args()
    main(cli_args) 