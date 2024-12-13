import os
import pytest
from unittest.mock import MagicMock
from src.callbacks.plot_results_factory import PlotResultsCallback

def test_plot_results_callback(tmpdir):
    # Mock validation data
    validation_data = MagicMock()
    validation_data.__iter__.return_value = iter([
        (MagicMock(shape=(1, 128, 128, 3)), MagicMock(shape=(1, 128, 128, 1)))
    ])
    
    # Create callback
    callback = PlotResultsCallback(validation_data=validation_data, plot_interval=1)
    
    # Mock model
    mock_model = MagicMock()
    callback.model = mock_model
    
    # Call on_epoch_end
    callback.on_epoch_end(epoch=1)
    
    # Check if the plot directory was created
    assert os.path.exists('plots')
    
    # Check if the plot file exists
    plot_file = os.path.join('plots', 'epoch_1_plot.png')
    assert os.path.exists(plot_file)
