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
import time # Import time for convergence check
from collections import deque
from concurrent.futures import ProcessPoolExecutor

# --- Core Host Execution Logic (Largely Unchanged from other attackers) ---

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

def _check_image_is_adversarial_wrapper(args_tuple):
    """
    Wrapper for check_image_is_adversarial to be used with multiprocessing.Pool.map,
    as instance methods cannot be pickled easily.
    """
    return check_image_is_adversarial(*args_tuple)

def check_image_is_adversarial(image_content, args, workdir, image_name_on_host):
    """
    A helper function that takes raw image bytes, writes them to a temporary file,
    runs the executable, and returns the boolean success status.
    """
    image_path_on_host = os.path.join(workdir, image_name_on_host)
    with open(image_path_on_host, 'wb') as f:
        f.write(image_content)

    is_successful = _run_executable_and_parse_success(image_path_on_host, args)
    os.remove(image_path_on_host)
    return is_successful


# --- Main Boundary Attack Loop ---

def main(args):
    dist_log_file = None
    adversarial_image = None
    best_l2_distance = float('inf')
    last_best_l2_distance = float('inf')
    patience_counter = 0

    try:
        os.setpgrp()
    except OSError:
        pass

    def sigint_handler(signum, frame):
        print("\nCtrl+C detected. Forcefully terminating all processes.")
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGINT, sigint_handler)
    
    temp_dir_base = "/dev/shm" if os.path.exists("/dev/shm") else None
    workdir = tempfile.mkdtemp(prefix="boundary_host_attack_", dir=temp_dir_base)
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"--- Boundary Attacker (Host Version) Started. Outputs: {args.output_dir}, Temp workdir: {workdir} ---")

        dist_log_path = os.path.join(args.output_dir, "boundary_dist_log_host.csv")
        dist_log_file = open(dist_log_path, 'w')
        dist_log_file.write("iteration,query_count,l2_distance,l_inf_norm\n")
        print(f"--- Distance values will be logged to: {dist_log_path} ---")

        print("--- Verifying local paths ---")
        static_files = [args.executable, args.model, args.hooks, args.start_adversarial, args.image]
        for f in static_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required file not found: {f}")

        # --- Initial Setup for Boundary Attack ---
        # MODIFICATION: Load images in color (3 channels) instead of grayscale
        print("--- Loading images in COLOR mode ---")
        original_image = cv2.imread(args.image, cv2.IMREAD_COLOR).astype(np.float32)
        adversarial_image = cv2.imread(args.start_adversarial, cv2.IMREAD_COLOR).astype(np.float32)

        if original_image is None:
            raise FileNotFoundError(f"Failed to load original image at: {args.image}")
        if adversarial_image is None:
            raise FileNotFoundError(f"Failed to load starting adversarial image at: {args.start_adversarial}")
        
        # MODIFICATION: Automatically resize the starting adversarial image if dimensions do not match
        if original_image.shape != adversarial_image.shape:
            print(f"Warning: Original image shape {original_image.shape} and starting adversarial shape {adversarial_image.shape} differ.")
            print("Resizing starting adversarial image to match original image's dimensions.")
            adversarial_image = cv2.resize(
                adversarial_image, 
                (original_image.shape[1], original_image.shape[0]), 
                interpolation=cv2.INTER_AREA
            )

        print("--- Verifying initial image states ---")
        _, encoded_orig = cv2.imencode(".png", original_image.astype(np.uint8))
        orig_is_adv = check_image_is_adversarial(encoded_orig.tobytes(), args, workdir, "temp_orig.png")
        if orig_is_adv:
            raise RuntimeError("The original image is already adversarial (`true`). Boundary attack requires a non-adversarial starting point.")
        print("Original image is correctly non-adversarial (`false`).")

        _, encoded_adv = cv2.imencode(".png", adversarial_image.astype(np.uint8))
        start_is_adv = check_image_is_adversarial(encoded_adv.tobytes(), args, workdir, "temp_start_adv.png")
        if not start_is_adv:
            raise RuntimeError("The starting adversarial image is not adversarial (`false`). Boundary attack requires an adversarial starting point.")
        print("Starting adversarial image is correctly adversarial (`true`).")
        
        query_count = 2 # We've already made two queries

        # --- Initialize variables for adaptive step sizes ---
        source_step = args.source_step
        spherical_step = args.spherical_step
        if args.dynamic_steps:
            print("--- Adaptive step size enabled ---")
            source_step_success_history = deque(maxlen=args.adaptation_window)
        # ---

        # MODIFICATION: Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for i in range(args.iterations):
                if args.dynamic_steps:
                    print(f"--- Iteration {i+1}/{args.iterations} (Queries: {query_count}) [src_step={source_step:.1e}, sph_step={spherical_step:.1e}] ---")
                else:
                    print(f"--- Iteration {i+1}/{args.iterations} (Total Queries: {query_count}) ---")


                # 1. Try to move closer to the original image (Source Step)
                trial_image = (1 - source_step) * adversarial_image + source_step * original_image
                
                _, encoded_trial = cv2.imencode(".png", np.clip(trial_image, 0, 255).astype(np.uint8))
                is_successful = check_image_is_adversarial(encoded_trial.tobytes(), args, workdir, "temp_trial_source.png")
                query_count += 1
                
                # --- Adaptive Step Logic ---
                if args.dynamic_steps:
                    source_step_success_history.append(is_successful)
                    # Only adapt if we have enough history
                    if len(source_step_success_history) == args.adaptation_window:
                        current_success_rate = np.mean(source_step_success_history)
                        
                        if current_success_rate > args.target_success_rate:
                            # Too conservative, get more aggressive
                            source_step *= args.step_adaptation_factor
                            spherical_step /= args.step_adaptation_factor
                            print(f"Adaptive Steps: Success rate {current_success_rate:.2f} > target. Increasing source step, decreasing spherical.")
                        else:
                            # Too aggressive, get more conservative
                            source_step /= args.step_adaptation_factor
                            spherical_step *= args.step_adaptation_factor
                            print(f"Adaptive Steps: Success rate {current_success_rate:.2f} <= target. Decreasing source step, increasing spherical.")
                        
                        # Clamp steps to prevent them from becoming too large or small
                        source_step = np.clip(source_step, 1e-6, 0.5)
                        spherical_step = np.clip(spherical_step, 1e-6, 0.5)
                # ---

                if is_successful:
                    adversarial_image = trial_image
                    print(f"Source step successful. Moved closer to original.")
                else:
                    # 2. If moving closer fails, explore along the boundary in parallel (Spherical Step)
                    print(f"Source step failed. Exploring boundary with {args.workers} workers...")
                    
                    direction_to_original = original_image - adversarial_image
                    dist_to_original = np.linalg.norm(direction_to_original)

                    # Prepare arguments for parallel execution
                    starmap_args = []
                    trial_images_for_workers = []

                    for worker_idx in range(args.workers):
                        # Generate a random perturbation
                        perturbation = np.random.randn(*original_image.shape).astype(np.float32)
                        
                        # Project perturbation to be orthogonal
                        perturbation -= np.dot(perturbation.flatten(), direction_to_original.flatten()) / (dist_to_original**2 + 1e-9) * direction_to_original
                        
                        # Rescale using the current (possibly adapted) spherical_step
                        perturbation *= spherical_step * dist_to_original / (np.linalg.norm(perturbation) + 1e-9)

                        trial_image = adversarial_image + perturbation
                        trial_images_for_workers.append(trial_image)

                        _, encoded_trial = cv2.imencode(".png", np.clip(trial_image, 0, 255).astype(np.uint8))
                        
                        # Unique name for each worker's temp file
                        image_name = f"temp_trial_spherical_{worker_idx}.png"
                        starmap_args.append((encoded_trial.tobytes(), args, workdir, image_name))

                    # Run checks in parallel
                    # Use a lambda to unpack the tuple of arguments for each map call
                    results = executor.map(lambda p: check_image_is_adversarial(*p), starmap_args)
                    query_count += args.workers
                    
                    # Check if any worker found a valid adversarial example
                    step_succeeded = False
                    for worker_idx, success in enumerate(results):
                        if success:
                            adversarial_image = trial_images_for_workers[worker_idx]
                            print(f"Spherical step successful (from worker {worker_idx}). Explored boundary.")
                            step_succeeded = True
                            break # Found one, no need to check others

                    if not step_succeeded:
                        print("Spherical step failed for all workers in this iteration.")

                # Logging and saving
                l2_dist = np.linalg.norm(adversarial_image - original_image)
                linf_dist = np.max(np.abs(adversarial_image - original_image))
            
                print(f"Current L2 distance: {l2_dist:.4f}, L-inf norm: {linf_dist:.4f}")
                dist_log_file.write(f"{i+1},{query_count},{l2_dist:.6f},{linf_dist:.6f}\n")
                dist_log_file.flush()

                latest_image_path = os.path.join(args.output_dir, "latest_attack_image_boundary_host.png")
                cv2.imwrite(latest_image_path, adversarial_image.astype(np.uint8))

                if l2_dist < best_l2_distance:
                    best_l2_distance = l2_dist
                    print(f"New best L2 distance found: {l2_dist:.4f}. Saving best image.")
                    best_image_path = os.path.join(args.output_dir, "best_attack_image_boundary_host.png")
                    cv2.imwrite(best_image_path, adversarial_image.astype(np.uint8))
                
                # MODIFICATION: Convergence Check
                if abs(last_best_l2_distance - best_l2_distance) < args.convergence_threshold:
                    patience_counter += 1
                    print(f"Convergence patience counter: {patience_counter}/{args.patience}")
                else:
                    patience_counter = 0 # Reset counter if there is improvement

                last_best_l2_distance = best_l2_distance
                
                if patience_counter >= args.patience:
                    print(f"\n--- CONVERGENCE REACHED ---")
                    print(f"L2 distance has not improved by more than {args.convergence_threshold} for {args.patience} iterations.")
                    print("Stopping attack early.")
                    break


    except (FileNotFoundError, RuntimeError, KeyboardInterrupt) as e:
        print(f"\nAn error or interrupt occurred: {e}")
        if adversarial_image is not None:
            print("Saving the last best image...")
            interrupted_image_path = os.path.join(args.output_dir, "interrupted_attack_image_boundary_host.png")
            cv2.imwrite(interrupted_image_path, adversarial_image.astype(np.uint8))
            print(f"Last best image saved to: {interrupted_image_path}")
    finally:
        if dist_log_file:
            dist_log_file.close()
        if workdir and os.path.exists(workdir):
            shutil.rmtree(workdir)
            print(f"Temporary directory {workdir} cleaned up.")
        print("Cleanup finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A grey-box adversarial attack using Boundary Attack (Host Version).")
    # File Paths
    parser.add_argument("--executable", required=True, help="Local path to the target executable.")
    parser.add_argument("--image", required=True, help="Local path to the initial NON-ADVERSARIAL image.")
    parser.add_argument("--hooks", required=True, help="Local path to the JSON file defining hook points (needed by runner).")
    parser.add_argument("--model", required=True, help="Local path to the model file (e.g., .onnx).")
    parser.add_argument("--start-adversarial", required=True, help="Local path to an image that is already adversarial ('true').")
    # Attack Hyperparameters
    parser.add_argument("--iterations", type=int, default=10000, help="Maximum number of attack iterations.")
    parser.add_argument("--source-step", type=float, default=0.01, help="Initial step size for moving towards the source image.")
    parser.add_argument("--spherical-step", type=float, default=0.01, help="Initial step size for spherical boundary exploration.")
    
    # Adaptive Step Size Parameters
    parser.add_argument("--dynamic-steps", action='store_true', help="Enable adaptive adjustment of source and spherical steps.")
    parser.add_argument("--step-adaptation-factor", type=float, default=1.5, help="Factor for adapting step sizes (e.g., 1.5 -> 50%% change). Used if --dynamic-steps is enabled.")
    parser.add_argument("--target-success-rate", type=float, default=0.25, help="The target success rate for the source step. Used if --dynamic-steps is enabled.")
    parser.add_argument("--adaptation-window", type=int, default=30, help="Window size over which to calculate the success rate. Used if --dynamic-steps is enabled.")

    # Convergence Parameters
    parser.add_argument("--patience", type=int, default=50, help="Number of iterations to wait for improvement before stopping early.")
    parser.add_argument("--convergence-threshold", type=float, default=1e-4, help="The minimum L2 distance improvement to reset the patience counter.")
    # System
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for the spherical step. Defaults to 4.")
    parser.add_argument("--output-dir", type=str, default="attack_outputs_boundary_host", help="Directory to save output images and logs.")
    
    cli_args = parser.parse_args()
    main(cli_args) 