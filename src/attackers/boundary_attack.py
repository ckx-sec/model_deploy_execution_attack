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
        original_image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        adversarial_image = cv2.imread(args.start_adversarial, cv2.IMREAD_GRAYSCALE).astype(np.float32)

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

        for i in range(args.iterations):
            print(f"--- Iteration {i+1}/{args.iterations} (Total Queries: {query_count}) ---")

            # 1. Try to move closer to the original image (Source Step)
            trial_image = (1 - args.source_step) * adversarial_image + args.source_step * original_image
            
            _, encoded_trial = cv2.imencode(".png", np.clip(trial_image, 0, 255).astype(np.uint8))
            is_successful = check_image_is_adversarial(encoded_trial.tobytes(), args, workdir, "temp_trial.png")
            query_count += 1
            
            if is_successful:
                adversarial_image = trial_image
                print(f"Source step successful. Moved closer to original.")
            else:
                # 2. If moving closer fails, explore along the boundary (Spherical Step)
                direction_to_original = original_image - adversarial_image
                dist_to_original = np.linalg.norm(direction_to_original)

                # Generate a random perturbation
                perturbation = np.random.randn(*original_image.shape).astype(np.float32)
                
                # Project perturbation to be orthogonal to the direction vector
                perturbation -= np.dot(perturbation.flatten(), direction_to_original.flatten()) / (dist_to_original**2) * direction_to_original
                
                # Rescale the perturbation
                perturbation *= args.spherical_step * dist_to_original / np.linalg.norm(perturbation)

                trial_image = adversarial_image + perturbation

                _, encoded_trial = cv2.imencode(".png", np.clip(trial_image, 0, 255).astype(np.uint8))
                is_successful = check_image_is_adversarial(encoded_trial.tobytes(), args, workdir, "temp_trial.png")
                query_count += 1

                if is_successful:
                    adversarial_image = trial_image
                    print(f"Spherical step successful. Explored boundary.")

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
    parser.add_argument("--source-step", type=float, default=0.01, help="Step size for moving towards the source image.")
    parser.add_argument("--spherical-step", type=float, default=0.01, help="Step size for spherical boundary exploration.")
    # System
    parser.add_argument("--output-dir", type=str, default="attack_outputs_boundary_host", help="Directory to save output images and logs.")
    
    cli_args = parser.parse_args()
    main(cli_args) 