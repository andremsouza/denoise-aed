"""Configuration package for the project.

This package provides access to the configuration system.
"""

from src.config.experiment_config import ExperimentConfig

# Create a default configuration instance
config = ExperimentConfig.create_default_config()

# For backward compatibility, expose the constants at the top level
# Infrastructure constants
USE_MLFLOW = config.infrastructure.use_mlflow

# Data constants
DATA_DIRECTORY = config.data.data_directory
TRANSFORMED_DATA_DIRECTORY = config.data.transformed_data_directory
ANNOTATION_FILE = config.data.annotation_file
TRAIN_ANNOTATION_FILE = config.data.train_annotation_file
VAL_ANNOTATION_FILE = config.data.val_annotation_file
TEST_ANNOTATION_FILE = config.data.test_annotation_file
FEATURE_FILE = config.data.feature_file
SAMPLE_RATE = config.data.sample_rate
NUM_BANDS = config.data.num_bands

# Model constants
# Add model-specific constants from AudioModelConfig here
# For example:
# MODEL_TYPE = config.model.model_type

# Training constants
BATCH_SIZE = config.training.batch_size
MODELS_DIRECTORY = config.training.models_directory
LOG_DIRECTORY = config.training.log_directory
USE_PRETRAINED = config.training.use_pretrained
MAX_EPOCHS = config.training.max_epochs
EARLY_STOPPING_PATIENCE = config.training.early_stopping_patience
NUM_WORKERS = config.training.num_workers
DEVICE = config.training.device
RANDOM_SEED = config.training.random_seed

# Annotation constants
NUM_CLASSES = config.annotation.num_classes
SAMPLE_SECONDS = config.annotation.sample_seconds
ANNOTATION_SECONDS = config.annotation.annotation_seconds

# Evaluation/Utility constants
PRED_THRESHOLD = config.evaluation.pred_threshold
SKIP_TRAINED_MODELS = config.evaluation.skip_trained_models
