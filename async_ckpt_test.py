import os
import threading
import queue
import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict


class AsyncCheckpointManager:
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Initialize the asynchronous checkpoint manager.
        
        Args:
            checkpoint_dir (str): Directory where checkpoints will be saved.
            max_checkpoints (int): Maximum number of checkpoints to keep. Older checkpoints will be deleted.
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoint_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.checkpoints = []  # List of saved checkpoints (filenames)
        
        # Create the directory if it does not exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Start the background thread for saving checkpoints
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def _save_checkpoint(self, params: Dict[str, Any], step: int) -> str:
        """
        Save a single checkpoint file using numpy.savez.
        
        Args:
            params (Dict[str, Any]): Model parameters to save.
            step (int): Training step or epoch for naming the checkpoint.

        Returns:
            str: The path to the saved checkpoint file.
        """
        checkpoint_file = os.path.join(self.checkpoint_dir, f'checkpoint_{step}.npz')

        # Flatten the dictionary of parameters so that it can be saved with numpy.savez
        flat_params = self._flatten_dict(params)

        # Save the flattened parameters to a .npz file
        np.savez(checkpoint_file, **flat_params)

        return checkpoint_file

    def _flatten_dict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten a nested dictionary of parameters for saving with numpy.savez.
        
        Args:
            params (Dict[str, Any]): The nested dictionary of model parameters.
        
        Returns:
            Dict[str, Any]: Flattened dictionary with string keys.
        """
        flat_params = {}
        for key, value in params.items():
            if isinstance(value, dict):
                nested_flat = self._flatten_dict(value)
                for nested_key, nested_value in nested_flat.items():
                    flat_params[f"{key}.{nested_key}"] = nested_value
            else:
                flat_params[key] = np.array(value)
        return flat_params

    def _unflatten_dict(self, flat_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unflatten a dictionary to restore its nested structure after loading.
        
        Args:
            flat_params (Dict[str, Any]): The flattened dictionary.
        
        Returns:
            Dict[str, Any]: Unflattened dictionary of parameters.
        """
        params = {}
        for key, value in flat_params.items():
            keys = key.split('.')
            d = params
            for subkey in keys[:-1]:
                if subkey not in d:
                    d[subkey] = {}
                d = d[subkey]
            d[keys[-1]] = value
        return params

    def save(self, params: Dict[str, Any], step: int):
        """
        Queue a checkpoint for saving.
        
        Args:
            params (Dict[str, Any]): The model parameters to save.
            step (int): The training step or epoch number.
        """
        # Enqueue the checkpoint data and step number
        self.checkpoint_queue.put((params, step))

    def _worker(self):
        """
        Background thread function to save checkpoints asynchronously.
        """
        while not self.stop_event.is_set() or not self.checkpoint_queue.empty():
            try:
                # Get the next checkpoint from the queue (wait up to 1 second)
                params, step = self.checkpoint_queue.get(timeout=1)
                
                # Save the checkpoint
                checkpoint_file = self._save_checkpoint(params, step)
                print(f"Saved checkpoint: {checkpoint_file}")

                # Track the saved checkpoints and remove older ones if needed
                self.checkpoints.append(checkpoint_file)
                self._manage_checkpoints()
            except queue.Empty:
                pass

    def _manage_checkpoints(self):
        """
        Keep only the specified number of checkpoints, deleting the oldest ones.
        """
        if len(self.checkpoints) > self.max_checkpoints:
            # Remove oldest checkpoints to maintain the maximum number of checkpoints
            while len(self.checkpoints) > self.max_checkpoints:
                oldest_checkpoint = self.checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)
                    print(f"Deleted old checkpoint: {oldest_checkpoint}")

    def close(self):
        """
        Signal the checkpoint manager to stop and wait for the thread to finish.
        """
        self.stop_event.set()
        self.thread.join()
        print("Checkpoint manager closed.")

    @staticmethod
    def load_checkpoint(file_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint file using numpy.load.
        
        Args:
            file_path (str): Path to the checkpoint file.

        Returns:
            Dict[str, Any]: The deserialized model parameters.
        """
        with np.load(file_path, allow_pickle=True) as data:
            flat_params = dict(data.items())
        
        checkpoint_manager = AsyncCheckpointManager('')
        params = checkpoint_manager._unflatten_dict(flat_params)
        return jax.tree_util.tree_map(jnp.array, params)


# Example Usage
if __name__ == "__main__":
    # Example model parameters (using JAX arrays)
    params = {
        'layer1': {'weights': jax.numpy.array([1.0, 2.0]), 'biases': jax.numpy.array([0.1])},
        'layer2': {'weights': jax.numpy.array([0.5, -1.5]), 'biases': jax.numpy.array([0.2])}
    }

    # Create an async checkpoint manager
    checkpoint_manager = AsyncCheckpointManager('checkpoints', max_checkpoints=3)

    # Simulate training and saving checkpoints at different steps
    for step in range(10):
        checkpoint_manager.save(params, step)
        time.sleep(0.5)  # Simulate time between training steps

    # Close the manager (ensures all pending checkpoints are saved)
    checkpoint_manager.close()

    # Load the latest checkpoint for testing
    latest_checkpoint = 'checkpoints/checkpoint_9.npz'
    loaded_params = checkpoint_manager.load_checkpoint(latest_checkpoint)
    print(f"Loaded params from {latest_checkpoint}: {loaded_params}")
