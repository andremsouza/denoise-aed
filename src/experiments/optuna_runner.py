from datetime import datetime
from typing import Any, Callable

import mlflow
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.config.experiment_config import ExperimentConfig
from src.experiments.base import (
    ExperimentRunner,
    load_swine_datasets,
    create_data_loaders,
)


class OptunaExperimentRunner(ExperimentRunner):
    """Experiment runner with Optuna integration for hyperparameter optimization."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        experiment_prefix: str,
        optimization_metric: str = "val_auroc",
        direction: str = "maximize",
        n_trials: int = 100,
        n_jobs: int = 1,
        timeout: int | None = None,
        study_name: str | None = None,
        storage: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize Optuna experiment runner.

        Args:
            experiment_config: Base configuration for the experiment
            experiment_prefix: Prefix for the experiment name
            optimization_metric: Metric to optimize ('val_auroc', 'val_loss', etc.)
            direction: Direction of optimization ('maximize' or 'minimize')
            n_trials: Number of optimization trials to run
            timeout: Timeout in seconds for the optimization (optional)
            study_name: Name of the Optuna study (optional)
            storage: Optuna storage URL (optional)
            verbose: Whether to print verbose output
        """
        super().__init__(experiment_config, experiment_prefix, verbose)
        self.optimization_metric = optimization_metric
        self.direction = direction
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout

        # Create or load study
        if study_name is None:
            study_name = f"{experiment_prefix}_optimization"

        self.storage = storage
        self.study_name = study_name

        # Create pruning callback for early stopping poor trials
        self.pruning_callback: PyTorchLightningPruningCallback | None = None

    def _objective(
        self, trial: optuna.Trial, model_class: type[pl.LightningModule]
    ) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial
            model_class: PyTorch Lightning model class to optimize

        Returns:
            Value of the optimization metric
        """
        # Create a configuration from the trial
        trial_config = ExperimentConfig.from_trial(trial, self.experiment_config)

        # Update experiment config temporarily
        original_config = self.experiment_config
        self.experiment_config = trial_config

        if self.verbose:
            print(
                f"[{datetime.now()}]: Trial {trial.number} - Testing parameters: {trial.params}"
            )

        try:
            # Load datasets with the trial config
            dataset_train_wave, dataset_test_wave = load_swine_datasets(
                trial_config, verbose=self.verbose
            )

            # Create data loaders
            train_dataloader, test_dataloader = create_data_loaders(
                trial_config,
                dataset_train_wave,
                dataset_test_wave,
                verbose=self.verbose,
            )

            # Get dataset type and create experiment name
            dataset_type = self._get_dataset_type(train_dataloader)
            run_name = self._create_run_name_name(
                model_class=model_class.__name__, dataset_type=dataset_type
            )
            trial_experiment_name = f"{run_name}_trial{trial.number}"

            # Create model
            model = self._create_model(model_class)

            # Set up loggers (minimal for trials)
            loggers = self._setup_loggers(trial_experiment_name)

            # Create pruning callback for Optuna
            self.pruning_callback = PyTorchLightningPruningCallback(
                trial, monitor=self.optimization_metric
            )

            # Create early stopping callback with shorter patience for trials
            early_stopping = EarlyStopping(
                monitor=self.optimization_metric,
                mode="max" if self.direction == "maximize" else "min",
                patience=trial_config.training.early_stopping_patience // 2,
                verbose=False,
            )

            # Create model checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                dirpath=trial_config.training.models_directory,
                filename=trial_experiment_name
                + "-{"
                + self.optimization_metric
                + ":.4f}-{epoch:02d}",
                monitor=self.optimization_metric,
                verbose=False,
                save_top_k=1,
                mode="max" if self.direction == "maximize" else "min",
            )

            # Set up trainer with fewer epochs for trials
            # Limit epochs for trials
            max_epochs = min(trial_config.training.max_epochs, 20)
            trainer = pl.Trainer(
                callbacks=[checkpoint_callback, early_stopping, self.pruning_callback],
                max_epochs=max_epochs,
                logger=loggers,
                log_every_n_steps=min(50, len(train_dataloader)),
                enable_checkpointing=True,
                enable_progress_bar=self.verbose,
            )

            # Train model
            trainer.fit(model, train_dataloader, test_dataloader)

            # Get best score
            best_score = 0.0 if self.direction == "maximize" else float("inf")
            if checkpoint_callback.best_model_score is not None:
                best_score = float(checkpoint_callback.best_model_score)

            # Log trial results to MLflow if enabled
            if self.experiment_config.infrastructure.use_mlflow:
                with mlflow.start_run(
                    run_name=f"{trial_experiment_name}",
                    nested=True,
                ):
                    mlflow.log_params(trial.params)
                    mlflow.log_metric(self.optimization_metric, float(best_score))
                    mlflow.log_param("trial_number", trial.number)

            if self.verbose:
                print(
                    f"[{datetime.now()}]: Trial {trial.number} completed with score: {best_score}"
                )

            return float(best_score)

        # pylint: disable=broad-exception-caught
        except Exception as e:
            if self.verbose:
                print(
                    f"[{datetime.now()}]: Trial {trial.number} failed with error: {str(e)}"
                )
            # Return a poor value to indicate failure
            return 0.0 if self.direction == "maximize" else float("inf")

        finally:
            # Restore original config
            self.experiment_config = original_config

    def run_optimization(
        self, model_class: type[pl.LightningModule]
    ) -> tuple[ExperimentConfig, float]:
        """Run Optuna optimization to find the best hyperparameters.

        Args:
            model_class: PyTorch Lightning model class to optimize

        Returns:
            Tuple of (best configuration, best value)
        """
        if self.verbose:
            print(
                f"[{datetime.now()}]: Starting optimization with {self.n_trials} trials"
            )
            print(
                f"[{datetime.now()}]: Optimizing {self.optimization_metric} to {self.direction}"
            )

        # Create or load study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(
                seed=self.experiment_config.training.random_seed
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=0, interval_steps=1
            ),
        )

        # Set up MLflow tracking
        if self.experiment_config.infrastructure.use_mlflow:
            with mlflow.start_run(
                run_name=f"{self.experiment_prefix}_optimization_{datetime.now()}",
                tags={"owner": "André Moreira Souza", "type": "optimization"},
            ):
                mlflow.log_param("n_trials", self.n_trials)
                mlflow.log_param("optimization_metric", self.optimization_metric)
                mlflow.log_param("direction", self.direction)

                # Run the optimization
                study.optimize(
                    lambda trial: self._objective(trial, model_class),
                    n_trials=self.n_trials,
                    timeout=self.timeout,
                    n_jobs=self.n_jobs,
                    show_progress_bar=self.verbose,
                )

                # Log best parameters and result
                mlflow.log_params(study.best_params)
                mlflow.log_metric(f"best_{self.optimization_metric}", study.best_value)
        else:
            # Run the optimization without MLflow
            study.optimize(
                lambda trial: self._objective(trial, model_class),
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                show_progress_bar=self.verbose,
            )

        # Create best configuration
        best_config = ExperimentConfig.from_trial(
            optuna.trial.FixedTrial(study.best_params), self.experiment_config
        )

        if self.verbose:
            print(f"[{datetime.now()}]: Optimization complete")
            print(f"[{datetime.now()}]: Best parameters: {study.best_params}")
            print(
                f"[{datetime.now()}]: Best {self.optimization_metric}: {study.best_value}"
            )

        return best_config, study.best_value

    def run_with_best_params(self, model_class: type[pl.LightningModule]) -> None:
        """Run a full experiment with the best parameters from optimization.

        Args:
            model_class: PyTorch Lightning model class to use
        """
        # Run optimization to find best parameters
        best_config, best_value = self.run_optimization(model_class)

        if self.verbose:
            print(f"[{datetime.now()}]: Running final experiment with best parameters")

        # Update experiment config with best parameters
        self.experiment_config = best_config

        # Run full experiment with best parameters
        super().run_experiment(model_class)
