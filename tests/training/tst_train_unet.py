import pytest
from unittest.mock import patch, MagicMock
from src.training.train_unet import main


@pytest.fixture
def mock_keras_backend():
    """Fixture to mock Keras backend clear_session."""
    with patch("keras.backend.clear_session") as mock_clear_session:
        yield mock_clear_session


@pytest.fixture
def mock_environment_variable():
    """Fixture to mock setting environment variables."""
    with patch("os.environ", {}) as mock_env:
        yield mock_env


@pytest.fixture
def mock_mlflow():
    """Fixture to mock MLflow functions."""
    with patch("mlflow.set_tracking_uri") as mock_set_tracking_uri, \
         patch("mlflow.set_experiment") as mock_set_experiment, \
         patch("mlflow.start_run") as mock_start_run, \
         patch("mlflow.keras.autolog") as mock_autolog:
        yield {
            "set_tracking_uri": mock_set_tracking_uri,
            "set_experiment": mock_set_experiment,
            "start_run": mock_start_run,
            "autolog": mock_autolog
        }


@pytest.fixture
def mock_model_fit():
    """Fixture to mock the model training process."""
    with patch("keras.Model.fit") as mock_fit:
        mock_fit.return_value = MagicMock(name="MockedTrainingHistory")
        yield mock_fit


def tst_main_function(
    mock_keras_backend, mock_environment_variable, mock_mlflow, mock_model_fit
):
    """
    Test the main function in the training script.
    Verifies that necessary steps like clearing backend, setting environment variables,
    initializing MLflow, and calling model.fit are executed.
    """
    # Call the main function
    main()

    # Verify that Keras backend clear_session was called
    mock_keras_backend.assert_called_once()

    # Verify that environment variable was set
    assert mock_environment_variable['TF_CUDNN_DETERMINISTIC'] == '1'

    # Verify MLflow functions were called
    mock_mlflow["set_tracking_uri"].assert_called_once()
    mock_mlflow["set_experiment"].assert_called_once_with("unet_experiment_v101")
    mock_mlflow["autolog"].assert_called_once()
    mock_mlflow["start_run"].assert_called_once()

    # Verify that model.fit was called
    mock_model_fit.assert_called_once()
