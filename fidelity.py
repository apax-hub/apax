import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax
from functools import partial

# --- Functions to fit ---
def f_lo(x):
    return jnp.sin(x) + 0.1 * x

def f_hi(x):
    return f_lo(x) + 0.2 * jnp.sin(5 * x)

# --- Multi-Fidelity Model ---
class MultiFidelityMLP(nn.Module):
    hidden_sizes: list

    @nn.compact
    def __call__(self, x, fidelity_mask):
        for h in self.hidden_sizes:
            x = nn.silu(nn.Dense(h)(x))
        output = nn.Dense(2)(x)  # [lo, delta_hi]
        return jnp.sum(output * fidelity_mask, axis=1, keepdims=True)

# --- Training setup ---
def create_train_state(rng, model, learning_rate=1e-3):
    params = model.init(rng, jnp.ones((1, 1)), jnp.ones((1, 2)))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def loss_fn(params, model, x, y, mask):
    pred = model({"params": params}, x, mask)
    return jnp.mean((pred - y) ** 2)

@jax.jit
def train_step(state, x, y, mask):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, state.apply_fn, x, y, mask)
    return state.apply_gradients(grads=grads), loss

# --- Flattened gradient of model output ---
def compute_flat_grad(params, model, x, mask):
    def output_fn(p):
        return model.apply({"params": p}, x, mask).squeeze()
    grad = jax.grad(output_fn)(params)
    flat_grad, _ = jax.flatten_util.ravel_pytree(grad)
    return flat_grad

# --- Farthest Point Sampling ---
def farthest_point_sampling(gradients, k):
    first = np.random.randint(0, len(gradients))
    selected = [first]
    distances = np.full(len(gradients), np.inf)
    for _ in range(1, k):
        last = selected[-1]
        dists = np.linalg.norm(gradients - gradients[last], axis=1)
        distances = np.minimum(distances, dists)
        selected.append(np.argmax(distances))
    return selected

# --- Main Experiment ---
def run_experiment():
    rng = jax.random.PRNGKey(0)
    model = MultiFidelityMLP(hidden_sizes=[32, 32])

    # Create training data
    x_lo = np.random.uniform(0, 10, (50, 1))
    y_lo = f_lo(x_lo)
    m_lo = np.tile([1., 0.], (len(x_lo), 1))

    x_hi = np.random.uniform(0, 10, (50, 1))
    y_hi = f_hi(x_hi)
    m_hi = np.tile([1., 1.], (len(x_hi), 1))

    x_train = jnp.array(np.vstack([x_lo, x_hi]))
    y_train = jnp.array(np.vstack([y_lo, y_hi]))
    mask_train = jnp.array(np.vstack([m_lo, m_hi]))

    # Shuffle
    idx = jax.random.permutation(rng, len(x_train))
    x_train, y_train, mask_train = x_train[idx], y_train[idx], mask_train[idx]

    # Train model
    state = create_train_state(rng, model)
    for _ in range(200):
        state, _ = train_step(state, x_train, y_train, mask_train)

    # Duplicate candidate points: each x with lo and hi fidelity masks
    # x_cand_base = jnp.linspace(0, 10, 100).reshape(-1, 1)
    ncand_base = 20
    x_cand_base = np.random.uniform(0, 10, (ncand_base, 1))
    x_cand = jnp.vstack([x_cand_base, x_cand_base])
    mask_lo = jnp.tile(jnp.array([[1.0, 0.0]]), (ncand_base, 1))
    mask_hi = jnp.tile(jnp.array([[1.0, 1.0]]), (ncand_base, 1))
    fidelity_masks = jnp.vstack([mask_lo, mask_hi])

    # Compute flattened gradients
    grads = jnp.stack([
        compute_flat_grad(state.params, model, x_cand[i:i+1], fidelity_masks[i:i+1])
        for i in range(len(x_cand))
    ])
    selected_ids = farthest_point_sampling(np.array(grads), k=4)
    selected_ids = np.array(selected_ids)

    # Collect and report results
    selected_x = np.array(x_cand[selected_ids])
    selected_mask = np.array(fidelity_masks[selected_ids])
    num_lo = np.sum(selected_mask[:, 1] == 0.0)
    num_hi = np.sum(selected_mask[:, 1] == 1.0)

    # Plot
    x_plot = np.linspace(0, 10, 300).reshape(-1, 1)
    y_plot_lo = f_lo(x_plot)
    y_plot_hi = f_hi(x_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot_lo, '--', label="Low Fidelity (f_lo)")
    plt.plot(x_plot, y_plot_hi, '-', label="High Fidelity (f_hi)")

    # Highlight selected points
    x_selected_lo = selected_x[selected_mask[:, 1] == 0.0]
    x_selected_hi = selected_x[selected_mask[:, 1] == 1.0]
    plt.scatter(x_selected_lo, f_lo(x_selected_lo), label='Selected Lo Fidelity', color='blue')
    plt.scatter(x_selected_hi, f_hi(x_selected_hi), label='Selected Hi Fidelity', color='red')

    plt.title(f"Selected Points - Low: {num_lo}, High: {num_hi}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig("selected_points.png")

    return selected_x, selected_mask, num_lo, num_hi

# Run it
# selected_x, selected_ids = run_experiment()
# print("Selected points:", selected_x)
selected_x, selected_mask, num_lo, num_hi = run_experiment()
print(num_lo, num_hi)