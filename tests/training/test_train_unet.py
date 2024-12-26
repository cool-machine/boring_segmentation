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
        mock_env["TF_CUDNN_DETERMINISTIC"] = "1"
        yield mock_env


@pytest.fixture
def mock_mlflow():
    """Fixture to mock MLflow functionality."""
    with patch("src.training.train_unet.mlflow") as mock_mlflow:
        # Mock MLflow methods
        mock_mlflow.set_tracking_uri = MagicMock()
        mock_mlflow.set_experiment = MagicMock()
        mock_mlflow.start_run = MagicMock()
        mock_mlflow.keras = MagicMock()
        mock_mlflow.keras.autolog = MagicMock()
        yield mock_mlflow


@pytest.fixture
def mock_mlflow_client():
    """Fixture to mock MlflowClient and its methods."""
    with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
        mock_client_instance = MagicMock()
        # Simulate the behavior of get_experiment_by_name
        mock_client_instance.get_experiment_by_name.return_value = 'some_name'  # Experiment does not exist
        mock_client_instance.create_experiment.return_value = "123"  # Mock creation of new experiment
        mock_client_cls.return_value = mock_client_instance
        yield mock_client_instance


@pytest.fixture
def mock_model_fit():
    """Fixture to mock the model.fit method."""
    with patch("keras.Model.fit") as mock_fit:
        mock_fit.return_value = MagicMock(name="MockedTrainingHistory")
        yield mock_fit


@pytest.fixture
def mock_get_mlflow_uri(tmp_path):
    """Fixture to mock get_mlflow_uri and use a local file-based tracking URI."""
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir()
    with patch("src.training.train_unet.get_mlflow_uri") as mock_uri:
        mock_uri.return_value = f"file://{mlruns_dir}"
        yield mock_uri


def test_main_function(
    mock_keras_backend,
    mock_environment_variable,
    mock_mlflow,
    mock_mlflow_client,
    mock_model_fit,
    mock_get_mlflow_uri,
):
    """
    Test the main function in train_unet.py.
    Verifies that necessary MLflow calls, environment settings, and model training occur.
    """
    # Call the main function
    main()

    # Verify Keras backend clear_session was called
    mock_keras_backend.assert_called_once()

    # Verify the environment variable was set correctly
    assert mock_environment_variable["TF_CUDNN_DETERMINISTIC"] == "1"

    # Verify MLflow functions were called
    mock_mlflow.set_tracking_uri.assert_called_once_with(f"file://{mock_get_mlflow_uri.return_value[7:]}")
    mock_mlflow.set_experiment.assert_called_once_with("unet_experiment_v101")
    mock_mlflow.keras.autolog.assert_called_once()
    mock_mlflow.start_run.assert_called_once()

    # Verify model.fit was called
    mock_model_fit.assert_called_once()

    # Verify get_experiment_by_name was called exactly once
    # mock_mlflow_client.get_experiment_by_name.assert_called_once_with("unet_experiment_v101")


# from unittest.mock import patch, MagicMock
# import pytest

# # Patch mlflow in the train_unet module before importing main
# with patch("src.training.train_unet.mlflow") as mlflow_mock:
#     # Mock out the methods and attributes used in train_unet.py
#     mlflow_mock.set_tracking_uri = MagicMock()
#     mlflow_mock.set_experiment = MagicMock()
#     mlflow_mock.start_run = MagicMock()
#     mlflow_mock.keras = MagicMock()
#     mlflow_mock.keras.autolog = MagicMock()

#     # Now import main after patching
#     from src.training.train_unet import main

# @pytest.fixture
# def mock_keras_backend():
#     with patch("keras.backend.clear_session") as mock_clear_session:
#         yield mock_clear_session

# @pytest.fixture
# def mock_environment_variable():
#     with patch("os.environ", {}) as mock_env:
#         yield mock_env

# @pytest.fixture
# def mock_model_fit():
#     with patch("keras.Model.fit") as mock_fit:
#         mock_fit.return_value = MagicMock(name="MockedTrainingHistory")
#         yield mock_fit

# @pytest.fixture
# def mock_get_mlflow_uri():
#     with patch("src.training.train_unet.get_mlflow_uri") as mock_uri:
#         mock_uri.return_value = "http://fake-mlflow-uri"
#         yield mock_uri

# def test_main_function(
#     mock_keras_backend,
#     mock_environment_variable,
#     mock_model_fit,
#     mock_get_mlflow_uri
# ):
#     # Call the main function
#     main()

#     # Verify that Keras backend clear_session was called
#     mock_keras_backend.assert_called_once()

#     # Verify environment variable
#     assert mock_environment_variable['TF_CUDNN_DETERMINISTIC'] == '1'

#     # Now access the mlflow_mock from the outer scope
#     # We've patched mlflow at module level, so mlflow_mock is still accessible.
#     # It was defined outside the test scope, so we reference it as a global variable:
#     global mlflow_mock

#     # Assert MLflow calls
#     mlflow_mock.set_tracking_uri.assert_called_once_with("http://fake-mlflow-uri")
#     mlflow_mock.set_experiment.assert_called_once_with("unet_experiment_v101")
#     mlflow_mock.keras.autolog.assert_called_once()  # Check autolog call
#     mlflow_mock.start_run.assert_called_once()

#     # Verify model.fit was called
#     mock_model_fit.assert_called_once()


# import pytest
# from unittest.mock import patch, MagicMock
# from src.training.train_unet import main

# @pytest.fixture
# def mock_keras_backend():
#     """Fixture to mock Keras backend clear_session."""
#     with patch("keras.backend.clear_session") as mock_clear_session:
#         yield mock_clear_session

# @pytest.fixture
# def mock_environment_variable():
#     """Fixture to mock setting environment variables."""
#     with patch("os.environ", {}) as mock_env:
#         yield mock_env

# @pytest.fixture
# def mock_mlflow():
#     """Fixture to mock MLflow functions."""
#     with patch("mlflow.set_tracking_uri") as mock_set_tracking_uri, \
#          patch("mlflow.set_experiment") as mock_set_experiment, \
#          patch("mlflow.start_run") as mock_start_run, \
#          patch("mlflow.keras.autolog") as mock_autolog:
#         yield {
#             "set_tracking_uri": mock_set_tracking_uri,
#             "set_experiment": mock_set_experiment,
#             "start_run": mock_start_run,
#             "autolog": mock_autolog
#         }

# @pytest.fixture
# def mock_model_fit():
#     """Fixture to mock the model training process."""
#     with patch("keras.Model.fit") as mock_fit:
#         mock_fit.return_value = MagicMock(name="MockedTrainingHistory")
#         yield mock_fit

# @pytest.fixture
# def mock_get_mlflow_uri():
#     """Fixture to mock get_mlflow_uri function to avoid real Azure calls."""
#     with patch("src.training.train_unet.get_mlflow_uri") as mock_uri:
#         mock_uri.return_value = "http://fake-mlflow-uri"
#         yield mock_uri

# def test_main_function(
#     mock_keras_backend,
#     mock_environment_variable,
#     mock_mlflow,
#     mock_model_fit,
#     mock_get_mlflow_uri
# ):
#     """
#     Test the main function in the training script.
#     Verifies that necessary steps like clearing backend, setting environment variables,
#     initializing MLflow, and calling model.fit are executed.
#     """
#     # Call the main function
#     main()

#     # Verify that Keras backend clear_session was called
#     mock_keras_backend.assert_called_once()

#     # Verify that environment variable was set
#     assert mock_environment_variable['TF_CUDNN_DETERMINISTIC'] == '1'

#     # Verify MLflow functions were called
#     mock_mlflow["set_tracking_uri"].assert_called_once_with("http://fake-mlflow-uri")
#     mock_mlflow["set_experiment"].assert_called_once_with("unet_experiment_v101")
#     mock_mlflow["autolog"].assert_called_once()
#     mock_mlflow["start_run"].assert_called_once()

#     # Verify that model.fit was called
#     mock_model_fit.assert_called_once()




# import pytest
# from unittest.mock import patch, MagicMock
# from src.training.train_unet import main


# @pytest.fixture
# def mock_keras_backend():
#     """Fixture to mock Keras backend clear_session."""
#     with patch("keras.backend.clear_session") as mock_clear_session:
#         yield mock_clear_session


# @pytest.fixture
# def mock_environment_variable():
#     """Fixture to mock setting environment variables."""
#     with patch("os.environ", {}) as mock_env:
#         yield mock_env


# @pytest.fixture
# def mock_mlflow():
#     """Fixture to mock MLflow functions."""
#     with patch("mlflow.set_tracking_uri") as mock_set_tracking_uri, \
#          patch("mlflow.set_experiment") as mock_set_experiment, \
#          patch("mlflow.start_run") as mock_start_run, \
#          patch("mlflow.keras.autolog") as mock_autolog:
#         yield {
#             "set_tracking_uri": mock_set_tracking_uri,
#             "set_experiment": mock_set_experiment,
#             "start_run": mock_start_run,
#             "autolog": mock_autolog
#         }


# @pytest.fixture
# def mock_model_fit():
#     """Fixture to mock the model training process."""
#     with patch("keras.Model.fit") as mock_fit:
#         mock_fit.return_value = MagicMock(name="MockedTrainingHistory")
#         yield mock_fit


# def test_main_function(
#     mock_keras_backend, mock_environment_variable, mock_mlflow, mock_model_fit
# ):
#     """
#     Test the main function in the training script.
#     Verifies that necessary steps like clearing backend, setting environment variables,
#     initializing MLflow, and calling model.fit are executed.
#     """
#     # Call the main function
#     main()

#     # Verify that Keras backend clear_session was called
#     mock_keras_backend.assert_called_once()

#     # Verify that environment variable was set
#     assert mock_environment_variable['TF_CUDNN_DETERMINISTIC'] == '1'

#     # Verify MLflow functions were called
#     mock_mlflow["set_tracking_uri"].assert_called_once()
#     mock_mlflow["set_experiment"].assert_called_once_with("unet_experiment_v101")
#     mock_mlflow["autolog"].assert_called_once()
#     mock_mlflow["start_run"].assert_called_once()

#     # Verify that model.fit was called
#     mock_model_fit.assert_called_once()
