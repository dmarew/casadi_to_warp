import warp as wp
import numpy as np
import sys
import os

# Add generated kernels to path
script_dir = os.path.dirname(os.path.abspath(__file__))
kernels_dir = os.path.join(script_dir, "../generated_kernels")
sys.path.append(kernels_dir)

try:
    from state_armature_jac_fixed_kernel import state_armature_jac_fixed_kernel
except ImportError:
    print(f"Could not import state_armature_jac_fixed_kernel from {kernels_dir}")
    sys.exit(1)


def main():
    wp.init()

    # User provided q (converted to numpy)
    q_base = np.array(
        [-0.005744, -0.520502, 1.029688, 0.07328, 0.016227, 0.952728], dtype=np.float32
    )

    nv = 6  # fixed base
    nu = 6

    print("\n--- Fuzzing Search for Singularity ---")

    rng = np.random.default_rng(42)

    num_samples = 10000
    q_batch_np = np.tile(q_base, (num_samples, 1))

    # Add noise to anklepitch (3), ankleroll (4), toepitch (5)
    # Range of +/- 0.2 rad should cover a lot
    noise = rng.uniform(-0.2, 0.2, (num_samples, 3))
    # q_batch_np[:, 3:6] += noise.astype(np.float32)

    q_batch_wp = wp.from_numpy(q_batch_np, dtype=wp.float32)
    v_batch_wp = wp.zeros((num_samples, nv), dtype=wp.float32)
    I_batch_wp = wp.zeros((num_samples, nu), dtype=wp.float32)

    out_jac_fuzz = wp.zeros((num_samples, nu * nu), dtype=wp.float32)

    # dummy outputs
    out_mp = wp.zeros((num_samples, nu), dtype=wp.float32)
    out_mv = wp.zeros((num_samples, nu), dtype=wp.float32)
    out_arm = wp.zeros((num_samples, nv * nv), dtype=wp.float32)

    print(f"Launching kernel with {num_samples} samples...")
    wp.launch(
        kernel=state_armature_jac_fixed_kernel,
        dim=num_samples,
        inputs=[q_batch_wp, v_batch_wp, I_batch_wp],
        outputs=[out_mp, out_mv, out_jac_fuzz, out_arm],
    )
    wp.synchronize()

    jac_res = out_jac_fuzz.numpy()
    # Check for large values
    max_vals = np.max(np.abs(jac_res), axis=1)
    worst_idx = np.argmax(max_vals)
    worst_val = max_vals[worst_idx]

    print(f"Max Jacobian value found: {worst_val}")
    print(f"At configuration: {q_batch_np[worst_idx]}")
    print(f"Jacobian:\n{jac_res[worst_idx].reshape(nu, nv)}")

    if worst_val > 100:
        print("REPRODUCED INSTABILITY!")
    else:
        print("Did not reproduce instability with provided fuzz range.")


if __name__ == "__main__":
    main()
