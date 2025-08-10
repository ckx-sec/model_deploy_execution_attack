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
from enum import Enum, auto

# --- State Machine for Attack Conditions ---
class ConditionState(Enum):
    DORMANT = auto()  # Not yet active
    ACTIVE = auto()   # Currently being targeted
    LOCKED = auto()   # Successfully conquered and now being defended

# --- Condition Manager: The Brain of the Attack ---
class ConditionManager:
    """Manages the state, loss calculation, and progression of attack conditions."""
    def __init__(self, target_conditions, success_threshold, locking_weight_multiplier, punitive_factor, stability_patience):
        self.conditions = target_conditions
        self.success_threshold = success_threshold
        self.locking_weight_multiplier = locking_weight_multiplier
        self.punitive_factor = punitive_factor
        self.stability_patience = stability_patience

        self.current_stage = 1
        self.condition_data = {}
        for i, cond in enumerate(self.conditions):
            self.condition_data[i] = {
                'state': ConditionState.DORMANT,
                'stable_counter': 0,
                'name': cond.get('name', f'Condition_{i}')
            }

    def _get_operand_value(self, operand, hooks):
        """Resolves an operand's value from hooks or as a constant."""
        if operand['source'] == 'constant':
            return operand['value']
        elif operand['source'] == 'address':
            addr = operand['value']
            if addr not in hooks or not hooks[addr]:
                return None  # Hook data not available
            # In case of multiple hooks at the same address, use the first one.
            return hooks[addr][0]
        return None

    def calculate_individual_losses(self, current_hooks):
        """Calculates the raw, unweighted loss for every condition."""
        losses = {}
        for i, cond in enumerate(self.conditions):
            name = self.condition_data[i]['name']
            op1_val = self._get_operand_value(cond['operand1'], current_hooks)
            op2_val = self._get_operand_value(cond['operand2'], current_hooks)

            if op1_val is None or op2_val is None:
                losses[i] = float('inf')
                continue

            margin = cond.get('margin', 0.0)
            loss = 0.0
            
            cond_type = cond['type']
            if cond_type == 'greater_than':
                # MSE-style loss: penalize quadratically when op1 is not sufficiently greater than op2
                loss = max(0, op2_val - op1_val + margin)**2
            elif cond_type == 'less_than':
                # MSE-style loss: penalize quadratically when op1 is not sufficiently less than op2
                loss = max(0, op1_val - op2_val + margin)**2
            elif cond_type == 'equal_to':
                # MSE loss (L2 loss) for equality
                loss = (op1_val - op2_val)**2
            elif cond_type == 'not_equal_to':
                # MSE-style loss: penalize quadratically when the values are too close
                epsilon = cond.get('epsilon', 1e-6)
                loss = max(0, epsilon - abs(op1_val - op2_val))**2
            
            losses[i] = loss
        return losses

    def calculate_total_loss(self, current_hooks):
        """Calculates the final, weighted, and potentially punitive total loss."""
        individual_losses = self.calculate_individual_losses(current_hooks)
        total_loss = 0.0

        for i, cond in enumerate(self.conditions):
            loss = individual_losses.get(i, float('inf'))
            if loss == float('inf'):
                # We only care about inf loss for ACTIVE conditions
                if self.condition_data[i]['state'] == ConditionState.ACTIVE:
                    return float('inf')
                else:
                    continue

            state = self.condition_data[i]['state']
            base_weight = cond.get('base_weight', 1.0)
            
            stage_multiplier = 0.0
            if state == ConditionState.ACTIVE:
                stage_multiplier = 1.0
            elif state == ConditionState.LOCKED:
                stage_multiplier = self.locking_weight_multiplier

            punitive_factor = 1.0
            if state == ConditionState.LOCKED and loss > self.success_threshold:
                punitive_factor = self.punitive_factor
                print(f"WARNING: Locked condition '{self.condition_data[i]['name']}' is regressing (loss={loss:.4f})! Applying punitive factor.")

            total_loss += loss * base_weight * stage_multiplier * punitive_factor
        
        return total_loss

    def reconnaissance_and_calibrate(self, initial_hooks):
        """Implements 'Smart Start' by evaluating the initial image."""
        print("--- Running Reconnaissance and Calibrating Start Stage ---")
        initial_losses = self.calculate_individual_losses(initial_hooks)
        
        highest_stage = max(c.get('stage', 1) for c in self.conditions)
        
        for stage_num in range(1, highest_stage + 2):
            stage_conditions_indices = [i for i, c in enumerate(self.conditions) if c.get('stage', 1) == stage_num]
            if not stage_conditions_indices:
                continue

            can_be_active = True
            for i in stage_conditions_indices:
                if initial_losses.get(i, float('inf')) == float('inf'):
                    self.condition_data[i]['state'] = ConditionState.DORMANT
                    can_be_active = False
                elif initial_losses[i] <= self.success_threshold:
                    self.condition_data[i]['state'] = ConditionState.LOCKED
                    print(f"Condition '{self.condition_data[i]['name']}' (Stage {stage_num}) is ALREADY SATISFIED. Locking.")
                else:
                    # Will be marked active if no preceding condition is inf
                    pass # Keep as DORMANT for now

            if can_be_active:
                self.current_stage = stage_num
                # Now activate the non-locked ones
                for i in stage_conditions_indices:
                    if initial_losses.get(i, float('inf')) > self.success_threshold:
                        self.condition_data[i]['state'] = ConditionState.ACTIVE
                break # We found our starting stage
        
        print(f"--- Smart Start Complete. Starting attack at Stage {self.current_stage}. ---")


    def check_and_advance_stage(self, current_hooks):
        """Implements the 'Ratchet' logic to advance to the next stage."""
        active_conditions_indices = [i for i, c in enumerate(self.conditions) if self.condition_data[i]['state'] == ConditionState.ACTIVE]
        
        if not active_conditions_indices:
            return # No active conditions to check

        is_stage_stable = True
        current_losses = self.calculate_individual_losses(current_hooks)
        for i in active_conditions_indices:
            if current_losses.get(i, float('inf')) > self.success_threshold:
                self.condition_data[i]['stable_counter'] = 0
                is_stage_stable = False
            else:
                self.condition_data[i]['stable_counter'] += 1
        
        print(f"Stage {self.current_stage} stability counters: {[self.condition_data[i]['stable_counter'] for i in active_conditions_indices]}/{self.stability_patience}")
        
        # Check if ALL active conditions are stable for long enough
        all_stable = all(self.condition_data[i]['stable_counter'] >= self.stability_patience for i in active_conditions_indices)

        if all_stable:
            print(f"\n--- RATCHET CLICK: Stage {self.current_stage} has been conquered and is now LOCKED. ---\n")
            # Lock current stage
            for i in active_conditions_indices:
                self.condition_data[i]['state'] = ConditionState.LOCKED
            
            # Find and activate next stage
            next_stage = self.current_stage + 1
            next_stage_indices = [i for i, c in enumerate(self.conditions) if c.get('stage', 1) == next_stage]
            
            if not next_stage_indices:
                print("--- All stages conquered! Attack successful. ---")
                return True # Signal completion
            else:
                self.current_stage = next_stage
                print(f"--- Activating Stage {self.current_stage} ---")
                for i in next_stage_indices:
                    self.condition_data[i]['state'] = ConditionState.ACTIVE
        return False

    def is_attack_successful(self):
        return all(cd['state'] == ConditionState.LOCKED for cd in self.condition_data.values())


# --- Helper Functions (largely unchanged) ---
def write_multiple_files_to_host(files_data, dest_dir):
    for filename, data in files_data:
        path = os.path.join(dest_dir, filename)
        with open(path, 'wb') as f:
            f.write(data)

def remove_files_on_host_batch(file_pattern):
    try:
        for f in glob.glob(file_pattern):
            os.remove(f)
    except OSError as e:
        print(f"Warning: Error while trying to batch remove '{file_pattern}': {e}")


# --- Worker and Execution Functions ---

def _run_executable_and_parse_hooks(image_path_on_host, args):
    script_path = os.path.join(os.path.dirname(__file__), "run_gdb_host.sh") 
    command = [
        '/bin/bash',
        script_path,
        os.path.abspath(args.executable),
        os.path.abspath(args.model),
        os.path.abspath(image_path_on_host),
        os.path.abspath(args.hooks)
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = e.stderr if hasattr(e, 'stderr') else "Timeout or error during execution"
        print(f"Error running host executable for '{image_path_on_host}': {stderr}")
        return {} # Return empty hooks on error

    hooked_values = {}
    full_output = result.stdout + "\n" + result.stderr
    output_lines = full_output.splitlines()

    for line in output_lines:
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
                
    return hooked_values

def evaluate_mutation_on_host(task_args):
    """Worker function for the process pool."""
    image_path_on_host, condition_manager, args = task_args
    hooks = _run_executable_and_parse_hooks(image_path_on_host, args)
    loss = condition_manager.calculate_total_loss(hooks)
    return loss

def run_single_evaluation(image_content, args, workdir, image_name_on_host):
    """Handles a single, non-batched run of the executable."""
    image_path_on_host = os.path.join(workdir, image_name_on_host)
    with open(image_path_on_host, 'wb') as f:
        f.write(image_content)
    hooked_values = _run_executable_and_parse_hooks(image_path_on_host, args)
    os.remove(image_path_on_host)
    return hooked_values


# --- NES Gradient Estimator ---
def estimate_gradient_nes(image, args, condition_manager, workdir):
    run_id = uuid.uuid4().hex[:12]
    image_shape = image.shape
    pop_size = args.population_size
    sigma = args.sigma
    
    if pop_size % 2 != 0:
        raise ValueError(f"Population size must be even. Got {pop_size}.")

    half_pop_size = pop_size // 2
    noise_vectors = [np.random.randn(*image_shape) for _ in range(half_pop_size)]
    mutations_data_for_writing = []
    tasks = []

    for i, noise in enumerate(noise_vectors):
        mutant_pos = image + sigma * noise
        mutant_neg = image - sigma * noise
        
        _, encoded_pos = cv2.imencode(".png", np.clip(mutant_pos, 0, 255).astype(np.uint8))
        _, encoded_neg = cv2.imencode(".png", np.clip(mutant_neg, 0, 255).astype(np.uint8))

        fname_pos = f"temp_nes_{run_id}_{i}_pos.png"
        fname_neg = f"temp_nes_{run_id}_{i}_neg.png"
        
        mutations_data_for_writing.append((fname_pos, encoded_pos.tobytes()))
        mutations_data_for_writing.append((fname_neg, encoded_neg.tobytes()))

        path_pos = os.path.join(workdir, fname_pos)
        path_neg = os.path.join(workdir, fname_neg)
        tasks.append((path_pos, condition_manager, args))
        tasks.append((path_neg, condition_manager, args))

    try:
        write_multiple_files_to_host(mutations_data_for_writing, workdir)

        losses = np.zeros(pop_size)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(evaluate_mutation_on_host, tasks)
            for i, loss in enumerate(results):
                losses[i] = loss
                print(f"Evaluation progress: {i + 1}/{pop_size}", end='\r')
        print("\nEvaluation complete.")

    finally:
        cleanup_pattern = os.path.join(workdir, f"temp_nes_{run_id}_*.png")
        remove_files_on_host_batch(cleanup_pattern)

    if np.inf in losses:
        non_inf_max = np.max(losses[losses != np.inf], initial=0)
        losses[losses == np.inf] = non_inf_max + 1

    gradient = np.zeros_like(image, dtype=np.float32)
    for i in range(half_pop_size):
        loss_positive = losses[2 * i]
        loss_negative = losses[2 * i + 1]
        gradient += (loss_positive - loss_negative) * noise_vectors[i]

    gradient /= (pop_size * sigma) 
    return gradient


# --- Main Attack Loop ---
def main(args):
    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir_base = "/dev/shm" if os.path.exists("/dev/shm") else None
    workdir = tempfile.mkdtemp(prefix="nes_ratchet_attack_", dir=temp_dir_base)
    
    # Setup logging
    detailed_log_file = open(os.path.join(args.output_dir, "attack_log.csv"), 'w')
    detailed_log_file.write("iteration,total_queries,total_loss,active_stage,iter_time_s,total_time_s\n")
    
    # Setup signal handler
    try:
        os.setpgrp()
    except OSError:
        pass
    signal.signal(signal.SIGINT, lambda s, f: os.killpg(os.getpgrp(), signal.SIGKILL))

    try:
        # --- Load Config and Initialize Condition Manager ---
        with open(args.hooks, 'r') as f:
            hooks_config = json.load(f)
        
        target_conditions = hooks_config.get('target_conditions', [])
        if not target_conditions:
            raise ValueError("hooks.json must contain a 'target_conditions' list.")

        condition_manager = ConditionManager(
            target_conditions,
            args.success_threshold,
            args.locking_weight_multiplier,
            args.punitive_factor,
            args.stability_patience
        )

        # --- Smart Start: Reconnaissance and Calibration ---
        original_image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
        if original_image is None: raise FileNotFoundError(f"Could not read image: {args.image}")

        is_success_encoding, encoded_image = cv2.imencode(".png", original_image)
        if not is_success_encoding: raise RuntimeError("Failed to encode initial image.")
        
        initial_hooks = run_single_evaluation(encoded_image.tobytes(), args, workdir, "recon_image.png")
        total_queries = 1
        condition_manager.reconnaissance_and_calibrate(initial_hooks)

        attack_image = original_image.copy().astype(np.float32)
        
        # Adam optimizer parameters
        m, v = np.zeros_like(attack_image), np.zeros_like(attack_image)
        beta1, beta2, epsilon_adam = 0.9, 0.999, 1e-8
        adam_step_counter = 0
        start_time = time.time()
        best_loss_so_far = float('inf')

        # --- Main Attack Loop ---
        for i in range(args.iterations):
            iter_start_time = time.time()
            print(f"\n--- Iteration {i+1}/{args.iterations} | Stage: {condition_manager.current_stage} | Total Queries: {total_queries} ---")
            
            # Estimate gradient
            grad = estimate_gradient_nes(attack_image, args, condition_manager, workdir)
            total_queries += args.population_size
            
            # Adam Optimizer Update
            adam_step_counter += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** adam_step_counter)
            v_hat = v / (1 - beta2 ** adam_step_counter)
            update_step = args.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon_adam)
            attack_image -= update_step

            # Clipping and projection
            perturbation = np.clip(attack_image - original_image.astype(np.float32), -args.l_inf_norm, args.l_inf_norm)
            attack_image = np.clip(original_image.astype(np.float32) + perturbation, 0, 255)

            # Verification and State Update
            is_success_encoding, encoded_image = cv2.imencode(".png", attack_image.astype(np.uint8))
            current_hooks = run_single_evaluation(encoded_image.tobytes(), args, workdir, "verify_image.png")
            total_queries += 1
            total_loss = condition_manager.calculate_total_loss(current_hooks)
            
            iter_time = time.time() - iter_start_time
            total_time_so_far = time.time() - start_time
            print(f"Current Total Loss: {total_loss:.6f}. Iter Time: {iter_time:.2f}s.")
            
            detailed_log_file.write(f"{i+1},{total_queries},{total_loss:.6f},{condition_manager.current_stage},{iter_time:.2f},{total_time_so_far:.2f}\n")
            detailed_log_file.flush()

            # Save latest and best images
            cv2.imwrite(os.path.join(args.output_dir, "latest_attack_image.png"), attack_image.astype(np.uint8))
            if total_loss < best_loss_so_far:
                best_loss_so_far = total_loss
                print(f"New best loss found: {total_loss:.6f}. Saving best image.")
                best_image_path = os.path.join(args.output_dir, "best_attack_image.png")
                cv2.imwrite(best_image_path, attack_image.astype(np.uint8))

            # Check for Ratchet Click and Stage Advance
            if condition_manager.check_and_advance_stage(current_hooks):
                # This returns true only when all stages are done
                break

        if condition_manager.is_attack_successful():
            print("\n--- Attack Successful! All conditions are LOCKED. ---")
            successful_image_path = os.path.join(args.output_dir, "successful_attack_image.png")
            cv2.imwrite(successful_image_path, attack_image.astype(np.uint8))
            print(f"Adversarial image saved to: {successful_image_path}")
        else:
            print("\n--- Attack finished (max iterations reached), but not all conditions were locked. ---")

    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
    finally:
        if 'detailed_log_file' in locals() and detailed_log_file:
            detailed_log_file.close()
        if 'workdir' in locals() and workdir and os.path.exists(workdir):
            shutil.rmtree(workdir)
            print("Temporary directory cleaned up.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An advanced grey-box adversarial attack using NES with Ratchet-Locking and Punitive Amplification.")
    # --- File Paths ---
    parser.add_argument("--executable", required=True, help="Path to the target executable.")
    parser.add_argument("--image", required=True, help="Path to the initial image to be attacked.")
    parser.add_argument("--hooks", required=True, help="Path to the JSON file defining target conditions and hook points.")
    parser.add_argument("--model", required=True, help="Path to the model file required by the executable.")
    
    # --- Attack Hyperparameters ---
    parser.add_argument("--iterations", type=int, default=300, help="Maximum number of attack iterations.")
    parser.add_argument("--learning-rate", type=float, default=10.0, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--l-inf-norm", type=float, default=20.0, help="Maximum L-infinity norm for the perturbation.")
    
    # --- NES Hyperparameters ---
    parser.add_argument("--population-size", type=int, default=100, help="Population size for NES. Must be even.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Sigma for NES noise.")

    # --- Ratchet & Punitive Mechanism Parameters ---
    ratchet_group = parser.add_argument_group("Ratchet & Punitive Mechanism")
    ratchet_group.add_argument("--success-threshold", type=float, default=0.01, help="Loss value below which a condition is considered 'satisfied'.")
    ratchet_group.add_argument("--locking-weight-multiplier", type=float, default=1.2, help="Multiplier for a LOCKED condition's weight (> 1.0).")
    ratchet_group.add_argument("--punitive-factor", type=float, default=5.0, help="Huge multiplier for a regressing LOCKED condition's loss.")
    ratchet_group.add_argument("--stability-patience", type=int, default=3, help="Number of consecutive iterations a stage must be satisfied before locking and advancing.")
    
    # --- System ---
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel processes for evaluation.")
    parser.add_argument("--output-dir", type=str, default="attack_outputs_ratchet", help="Directory to save output images and logs.")
    
    cli_args = parser.parse_args()
    main(cli_args) 