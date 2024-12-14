import os
import pytest
import numpy as np
from unittest.mock import MagicMock
from src.callbacks.plot_results_factory import PlotResultsCallback

def test_plot_results_callback(tmpdir):
    # Mock validation data
    validation_data = MagicMock()
    validation_data.__iter__.return_value = iter([
        (np.random.rand(1, 128, 128, 3), np.random.randint(0, 2, size=(1, 128, 128, 1)))
    ])
    
    # Create callback with a short plot_interval so it runs on epoch 1
    callback = PlotResultsCallback(validation_data=validation_data, plot_interval=1)
    
    # Mock model
    mock_model = MagicMock()
    # Set the mock model to return a valid prediction array
    mock_model.return_value = np.random.rand(1, 128, 128, 2)

    # Use set_model method to assign the mock model to the callback
    callback.set_model(mock_model)
    
    # Call on_epoch_end to trigger plotting
    callback.on_epoch_end(epoch=1)
    
    # Check if the plot directory was created
    plot_dir = os.path.join('outputs', 'plots')
    assert os.path.exists(plot_dir), f"Expected directory {plot_dir} does not exist."
    
    # Check if the plot file exists
    plot_file = os.path.join(plot_dir, 'epoch_1_plot.png')
    assert os.path.exists(plot_file), f"Expected file {plot_file} does not exist."
