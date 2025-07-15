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
        # In a query-heavy attack, we suppress error spew to keep logs clean.
        # print(f"Error running host executable for '{image_path_on_host}': {stderr}")
        return False

    full_output = result.stdout + "\n" + result.stderr
    is_successful = any("true" in line.lower() for line in full_output.splitlines())
    return is_successful

def check_image_is_adversarial(image_data, args, workdir, query_counter_ref):
    """
    A helper that takes raw image bytes, runs the check, and increments the query counter.
    """
    image_name_on_host = f"temp_check_{uuid.uuid4().hex[:12]}.png"
    image_path_on_host = os.path.join(workdir, image_name_on_host)
    
    # image_data is expected to be a numpy array
    _, encoded_image = cv2.imencode(".png", np.clip(image_data, 0, 255).astype(np.uint8))
    
    with open(image_path_on_host, 'wb') as f:
        f.write(encoded_image)
    
    query_counter_ref[0] += 1
    is_successful = _run_executable_and_parse_success(image_path_on_host, args)
    
    os.remove(image_path_on_host)
    return is_successful


# --- Sign-OPT Core Functions ---

def binary_search_to_boundary(original_image, adversarial_image, args, workdir, query_counter_ref):
    """
    Performs a binary search between two images to find a point on the decision boundary.
    Assumes original_image is non-adversarial and adversarial_image is adversarial.
    """
    low_bound = 0.0
    high_bound = 1.0
    
    # Binary search for the boundary
    for _ in range(args.binary_search_steps):
        mid_bound = (low_bound + high_bound) / 2
        mid_image = (1 - mid_bound) * original_image + mid_bound * adversarial_image
        
        if check_image_is_adversarial(mid_image, args, workdir, query_counter_ref):
            high_bound = mid_bound
        else:
            low_bound = mid_bound

    final_boundary_image = (1 - high_bound) * original_image + high_bound * adversarial_image
    return final_boundary_image


# --- Main Attack Loop ---

def main(args):
    dist_log_file = None
    adversarial_image = None
    best_l2_distance = float('inf')

    try:
        os.setpgrp()
    except OSError:
        pass

    def sigint_handler(signum, frame):
        print("\nCtrl+C detected. Forcefully terminating all processes.")
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGINT, sigint_handler)
    
    temp_dir_base = "/dev/shm" if os.path.exists("/dev/shm") else None
    workdir = tempfile.mkdtemp(prefix="signopt_host_attack_", dir=temp_dir_base)
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"--- Sign-OPT Attacker (Host Version) Started. Outputs: {args.output_dir}, Temp workdir: {workdir} ---")

        dist_log_path = os.path.join(args.output_dir, "signopt_dist_log_host.csv")
        dist_log_file = open(dist_log_path, 'w')
        dist_log_file.write("iteration,query_count,l2_distance,l_inf_norm\n")
        print(f"--- Distance values will be logged to: {dist_log_path} ---")

        print("--- Verifying local paths ---")
        static_files = [args.executable, args.model, args.hooks, args.start_adversarial, args.image]
        for f in static_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required file not found: {f}")

        original_image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        adversarial_image = cv2.imread(args.start_adversarial, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        query_count = [0]

        print("--- Verifying initial image states ---")
        if check_image_is_adversarial(original_image, args, workdir, query_count):
            raise RuntimeError("Original image is already adversarial ('true').")
        print(f"Original image is correctly non-adversarial (`false`). Queries: {query_count[0]}")

        if not check_image_is_adversarial(adversarial_image, args, workdir, query_count):
            raise RuntimeError("Starting adversarial image is not adversarial ('false').")
        print(f"Starting adversarial image is correctly adversarial ('true'). Queries: {query_count[0]}")

        for i in range(args.iterations):
            print(f"--- Iteration {i+1}/{args.iterations} (Total Queries: {query_count[0]}) ---")

            # 1. Choose a random search direction
            u = np.random.randn(*original_image.shape).astype(np.float32)
            u /= np.linalg.norm(u)

            # 2. Estimate the sign of the gradient
            probe_image = adversarial_image + args.alpha * u
            
            # If probing in the `+u` direction is still adversarial, it means the gradient
            # direction is likely `-u` (pushing us further into adversarial territory).
            # We want to move against the gradient, so our update direction should be `u`.
            if check_image_is_adversarial(probe_image, args, workdir, query_count):
                update_direction = u
            else:
                # Otherwise, the gradient direction is likely `u`, and we should move in direction `-u`.
                update_direction = -u

            # 3. Take a step and project
            step_size = args.step_size
            
            # Create a candidate by moving along the update direction
            candidate_adv = adversarial_image + step_size * update_direction

            # Project this candidate back to the boundary. If the candidate is not adversarial,
            # this binary search will find the point on the line between it and the original image
            # that is on the boundary.
            if check_image_is_adversarial(candidate_adv, args, workdir, query_count):
                adversarial_image = binary_search_to_boundary(original_image, candidate_adv, args, workdir, query_count)
            else:
                # This can happen if the step size was too large. We can try a smaller step.
                # For simplicity, we just keep the old adversarial image for this iteration.
                print("Step was too large, skipping update for this iteration.")


            # Logging and saving
            l2_dist = np.linalg.norm(adversarial_image - original_image)
            linf_dist = np.max(np.abs(adversarial_image - original_image))
            
            print(f"Current L2 distance: {l2_dist:.4f}, L-inf norm: {linf_dist:.4f}")
            dist_log_file.write(f"{i+1},{query_count[0]},{l2_dist:.6f},{linf_dist:.6f}\n")
            dist_log_file.flush()

            latest_image_path = os.path.join(args.output_dir, "latest_attack_image_signopt_host.png")
            cv2.imwrite(latest_image_path, adversarial_image.astype(np.uint8))

            if l2_dist < best_l2_distance:
                best_l2_distance = l2_dist
                print(f"New best L2 distance found: {l2_dist:.4f}. Saving best image.")
                best_image_path = os.path.join(args.output_dir, "best_attack_image_signopt_host.png")
                cv2.imwrite(best_image_path, adversarial_image.astype(np.uint8))

    except (FileNotFoundError, RuntimeError, KeyboardInterrupt) as e:
        print(f"\nAn error or interrupt occurred: {e}")
        if adversarial_image is not None:
            interrupted_image_path = os.path.join(args.output_dir, "interrupted_attack_image_signopt_host.png")
            cv2.imwrite(interrupted_image_path, adversarial_image.astype(np.uint8))
    finally:
        if dist_log_file: dist_log_file.close()
        if workdir and os.path.exists(workdir): shutil.rmtree(workdir)
        print("Cleanup finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A grey-box adversarial attack using Sign-OPT (Host Version).")
    # File Paths
    parser.add_argument("--executable", required=True, help="Local path to the target executable.")
    parser.add_argument("--image", required=True, help="Local path to the initial NON-ADVERSARIAL image.")
    parser.add_argument("--hooks", required=True, help="Local path to the JSON file defining hook points (needed by runner).")
    parser.add_argument("--model", required=True, help="Local path to the model file (e.g., .onnx).")
    parser.add_argument("--start-adversarial", required=True, help="Local path to an image that is already adversarial ('true').")
    # Attack Hyperparameters
    parser.add_argument("--iterations", type=int, default=1000, help="Maximum number of attack iterations.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Probe distance for sign estimation.")
    parser.add_argument("--step-size", type=float, default=1.0, help="Step size for each iteration's update.")
    parser.add_argument("--binary-search-steps", type=int, default=10, help="Number of binary search steps for projection.")
    # System
    parser.add_argument("--output-dir", type=str, default="attack_outputs_signopt_host", help="Directory to save output images and logs.")
    
    cli_args = parser.parse_args()
    main(cli_args) 