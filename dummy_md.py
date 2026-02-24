import jax
import jax.numpy as jnp
import nvtx
import time

# 1. Define a heavy computation (Simulating a Force Calculation)
# We use a large matrix multiplication to make the GPU work hard enough to measure.
@jax.jit
def heavy_step(positions, velocities):
    # Create a heavy load: Matrix multiplication of (N, N)
    # This represents the Neural Network inference in GMNN
    with jax.named_scope("FORCES"):
        forces = jnp.dot(positions, positions.T) 
    
    # Simple update (Euler integration)
    with nvtx.annotate("EULER", color="purple"):
        new_vel = velocities + forces * 1e-5
        new_pos = positions + new_vel * 1e-5
    
    return new_pos, new_vel

def main():
    print("Allocating memory...")
    # Create "Atoms": 4000x4000 matrix is approx 64MB, big enough to stress the GPU slightly
    key = jax.random.PRNGKey(0)
    pos = jax.random.normal(key, (2000, 2000), dtype=jnp.float32)
    vel = jax.random.normal(key, (2000, 2000), dtype=jnp.float32)

    # --- WARMUP PHASE ---
    # We run this ONCE to trigger JAX JIT Compilation.
    # We do NOT profile this, because compilation looks like "idle time".
    print("Warming up JIT compilation...")
    pos, vel = heavy_step(pos, vel)
    pos.block_until_ready()
    print("Warmup done. Starting profile loop...")

    # --- PROFILING PHASE ---
    # We run a loop to simulate MD steps.
    steps = 20
    
    # NVTX Range: This marks the entire simulation block in the timeline
    with nvtx.annotate("Simulation_Loop", color="yellow"):
        
        for i in range(steps):
            
            # Marker 1: The "Step" container (Blue)
            with nvtx.annotate(f"Step_{i}", color="blue"):
                
                # Marker 2: Python Dispatch (Green)
                # This measures how fast Python tells JAX what to do
                with nvtx.annotate("Python_Dispatch", color="green"):
                    pos, vel = heavy_step(pos, vel)
                
                # Simulate some CPU-side logic (e.g., trajectory writing overhead)
                # time.sleep(0.005) 

                # Marker 3: Synchronization (Red)
                # This measures how long the CPU waits for the GPU to finish
                with nvtx.annotate("WaitForGPU", color="red"):
                    pos.block_until_ready()

    print("Done! Profile captured.")

if __name__ == "__main__":
    main()