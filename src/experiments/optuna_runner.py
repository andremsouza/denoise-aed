"""Optuna experiment runner for hyperparameter optimization."""

from datetime import datetime
import gc
import logging

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import Logger as PlLogger
import mlflow
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
from torch.utils.data import DataLoader

from src.config.experiment_config import ExperimentConfig
from src.experiments.base import (
    ExperimentRunner,
    load_swine_datasets,
    create_data_loaders,
)

logger = logging.getLogger(__name__)


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
            n_jobs: Number of parallel jobs for Optuna
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

    def _setup_trial_dataloaders(
        self, trial_config: ExperimentConfig
    ) -> tuple[DataLoader, DataLoader, str]:
        """Load datasets and create dataloaders for a trial."""
        dataset_train_wave, dataset_test_wave = load_swine_datasets(
            trial_config, verbose=self.verbose
        )
        train_dataloader, test_dataloader = create_data_loaders(
            trial_config,
            dataset_train_wave,
            dataset_test_wave,
            verbose=self.verbose,
        )
        dataset_type = self._get_dataset_type(train_dataloader)
        return train_dataloader, test_dataloader, dataset_type

    def _setup_trial_model_and_loggers(
        self,
        trial_config: ExperimentConfig,
        model_class: type[pl.LightningModule],
        denoiser_class: type[pl.LightningModule],
        trial_experiment_name: str,
    ) -> tuple[pl.LightningModule, pl.LightningModule, list[PlLogger]]:
        """Create model and loggers for a trial."""
        original_config = self.experiment_config
        self.experiment_config = trial_config
        model = self._create_model(model_class)
        denoiser = self._create_denoiser(denoiser_class)
        self.experiment_config = original_config

        exp_loggers = self._setup_loggers(trial_experiment_name)
        return model, denoiser, exp_loggers

    def _setup_trial_callbacks(
        self,
        trial: optuna.Trial,
        trial_config: ExperimentConfig,
        trial_experiment_name: str,
    ) -> list[pl.Callback]:
        """Setup callbacks for a trial."""
        self.pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor=self.optimization_metric
        )
        early_stopping = EarlyStopping(
            monitor=self.optimization_metric,
            mode="max" if self.direction == "maximize" else "min",
            patience=trial_config.training.early_stopping_patience,
            verbose=False,
        )
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
        return [checkpoint_callback, early_stopping, self.pruning_callback]

    def _setup_trial_trainer(
        self,
        trial_config: ExperimentConfig,
        callbacks: list[pl.Callback],
        exp_loggers: list[PlLogger],
        train_dataloader_len: int,
    ) -> pl.Trainer:
        """Setup PyTorch Lightning Trainer for a trial."""
        max_epochs = min(trial_config.training.max_epochs, 20)
        return pl.Trainer(
            callbacks=callbacks,
            max_epochs=max_epochs,
            logger=exp_loggers,
            log_every_n_steps=min(50, train_dataloader_len),
            enable_checkpointing=True,
            enable_progress_bar=self.verbose,
        )

    def _objective(
        self,
        trial: optuna.Trial,
        model_class: type[pl.LightningModule],
        denoiser_class: type[pl.LightningModule],
        parent_run_id: str | None = None,
    ) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial
            model_class: PyTorch Lightning model class to optimize
            parent_run_id: Parent MLflow run ID for nested runs (optional)

        Returns:
            Value of the optimization metric
        """
        trial_config = ExperimentConfig.from_trial(trial, self.experiment_config)

        if self.verbose:
            logger.info("Trial %s - Testing parameters: %s", trial.number, trial.params)

        model: pl.LightningModule | None = None
        denoiser: pl.LightningModule | None = None
        train_dataloader: DataLoader | None = None
        test_dataloader: DataLoader | None = None
        trainer: pl.Trainer | None = None

        try:
            train_dataloader, test_dataloader, dataset_type = (
                self._setup_trial_dataloaders(trial_config)
            )

            denoiser = self._create_denoiser(denoiser_class)
            denoiser.eval()
            if self.verbose:
                logger.info("Using denoiser: %s", denoiser_class.__name__)
            train_dataloader.dataset.transform = denoiser
            test_dataloader.dataset.transform = denoiser

            run_name = self._create_run_name_name(
                model_class=model_class.__name__,
                denoiser_class=denoiser_class.__name__,
                dataset_type=dataset_type,
            )
            trial_experiment_name = f"{run_name}_trial{trial.number}"

            model, denoiser, exp_loggers = self._setup_trial_model_and_loggers(
                trial_config, model_class, denoiser_class, trial_experiment_name
            )

            callbacks = self._setup_trial_callbacks(
                trial, trial_config, trial_experiment_name
            )
            checkpoint_callback = next(
                (cb for cb in callbacks if isinstance(cb, ModelCheckpoint)), None
            )
            if checkpoint_callback is None:
                raise ValueError(
                    "ModelCheckpoint callback not found in callbacks list."
                )
            if self.verbose:
                logger.info(
                    "Setting up trainer for trial %s with run name: %s",
                    trial.number,
                    trial_experiment_name,
                )

            trainer = self._setup_trial_trainer(
                trial_config, callbacks, exp_loggers, len(train_dataloader)
            )

            original_config = self.experiment_config
            self.experiment_config = trial_config
            self._setup_mlflow_tracking()
            self.experiment_config = original_config

            try:
                if trial_config.infrastructure.use_mlflow:
                    with mlflow.start_run(
                        run_name=f"{trial_experiment_name}",
                        tags={"owner": "André Moreira Souza"},
                        log_system_metrics=(
                            trial_config.infrastructure.mlflow_log_system_metrics
                        ),
                        nested=True,
                        parent_run_id=parent_run_id,
                    ):
                        mlflow.log_param("run_name", trial_experiment_name)
                        mlflow.log_params(trial_config.to_dict(flat=True))
                        mlflow.log_params(trial.params)
                        mlflow.log_param("trial_number", trial.number)
                        trainer.fit(model, train_dataloader, test_dataloader)
                else:
                    trainer.fit(model, train_dataloader, test_dataloader)
            except optuna.exceptions.TrialPruned as e:
                if self.verbose:
                    logger.info("Trial %s pruned: %s", trial.number, str(e))
                raise e

            best_score = 0.0 if self.direction == "maximize" else float("inf")
            if checkpoint_callback.best_model_score is not None:
                best_score = float(checkpoint_callback.best_model_score)

            if self.verbose:
                logger.info(
                    "Trial %s completed with score: %s", trial.number, best_score
                )
            # Clean up resources
            del model, train_dataloader, test_dataloader, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return float(best_score)
        # pylint: disable=broad-except
        except Exception as e:
            if self.verbose:
                logger.error(
                    "Trial %s failed with error: %s",
                    trial.number,
                    str(e),
                    exc_info=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise e

    def run_optimization(
        self,
        model_class: type[pl.LightningModule],
        denoiser_class: type[pl.LightningModule],
    ) -> tuple[ExperimentConfig, float]:
        """Runs the Optuna hyperparameter optimization study.

        This method sets up and executes an Optuna study to find the best
        hyperparameters for the given `model_class`. It configures the study
        with a specific sampler (TPESampler) and pruner (MedianPruner).

        If MLflow integration is enabled in the experiment configuration,
        this method will:
        1. Create a parent MLflow run for the entire optimization study.
        2. Log optimization-level parameters (number of trials, metric, direction, etc.).
        3. Pass the parent MLflow run ID to the objective function so that individual
           trials can be logged as nested runs.
        4. Log the best trial's parameters, metric value, and trial number to the
           parent MLflow run.

        If MLflow is not used, the optimization proceeds without MLflow logging,
        and results are logged to the standard logger.

        Args:
            model_class: The PyTorch Lightning model class for which
                hyperparameters are to be optimized.

        Returns:
            A tuple containing:
            - best_config (ExperimentConfig): An ExperimentConfig object
              populated with the hyperparameters from the best trial.
            - best_value (float): The value of the optimization metric for
              the best trial.
        """
        if self.verbose:
            logger.info("Starting optimization with %s trials", self.n_trials)
            logger.info("Optimizing %s to %s", self.optimization_metric, self.direction)

        study = optuna.create_study(
            study_name=f"{self.study_name}_{model_class.__name__}_{denoiser_class.__name__}",
            storage=self.storage,
            load_if_exists=True,
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(
                seed=self.experiment_config.training.random_seed
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=2, interval_steps=1
            ),
        )

        if self.experiment_config.infrastructure.use_mlflow:
            opt_run_name = (
                f"{self.experiment_prefix}_{model_class.__name__}_{denoiser_class.__name__}"
                f"_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            with mlflow.start_run(
                run_name=opt_run_name,
                tags={"owner": "André Moreira Souza", "type": "optimization_study"},
            ):
                mlflow.log_param("n_trials", self.n_trials)
                mlflow.log_param("optimization_metric", self.optimization_metric)
                mlflow.log_param("direction", self.direction)
                mlflow.log_param("study_name", study.study_name)
                mlflow.log_param("model_class", model_class.__name__)
                mlflow.log_param("denoiser_class", denoiser_class.__name__)

                run_id = None
                active_run = mlflow.active_run()
                if active_run:
                    run_id = active_run.info.run_id
                    logger.info("MLflow run ID: %s", run_id)
                else:
                    logger.warning(
                        "No active MLflow run found for optimization parent run."
                    )

                study.optimize(
                    lambda trial: self._objective(
                        trial, model_class, denoiser_class, parent_run_id=run_id
                    ),
                    n_trials=self.n_trials,
                    timeout=self.timeout,
                    n_jobs=self.n_jobs,
                    catch=(RuntimeError, optuna.exceptions.TrialPruned),
                    show_progress_bar=self.verbose,
                )

                logger.info("Best trial: %s", study.best_trial.number)
                logger.info(
                    "Best value for %s: %s", self.optimization_metric, study.best_value
                )
                logger.info("Best parameters: %s", study.best_params)

                mlflow.log_params(study.best_params)
                mlflow.log_metric(f"best_{self.optimization_metric}", study.best_value)
                mlflow.set_tag("best_trial_number", study.best_trial.number)
        else:
            study.optimize(
                lambda trial: self._objective(trial, model_class, denoiser_class),
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                show_progress_bar=self.verbose,
            )
            logger.info("Best trial: %s", study.best_trial.number)
            logger.info(
                "Best value for %s: %s", self.optimization_metric, study.best_value
            )
            logger.info("Best parameters: %s", study.best_params)

        best_config = ExperimentConfig.from_trial(
            study.best_trial, self.experiment_config
        )

        if self.verbose:
            logger.info("Optimization complete")

        return best_config, study.best_value

    def run_with_best_params(
        self,
        model_class: type[pl.LightningModule],
        denoiser_class: type[pl.LightningModule],
    ) -> None:
        """Runs hyperparameter optimization to find the best parameters and then
        executes a final experiment using these optimal parameters.

        The method first performs an optimization run by calling
        `self.run_optimization(model_class)` to identify the best
        hyperparameter configuration. The best configuration and its
        corresponding metric value are determined based on `self.optimization_metric`.

        Once the best parameters are found, `self.experiment_config` is updated
        with this configuration. A final experiment is then conducted by invoking
        `super().run_experiment(model_class)` with these parameters.

        Verbose logging throughout the process, including the best parameters found
        and the progress of the final run, is enabled if `self.verbose` is True.

        Args:
            model_class (type[pl.LightningModule]): The PyTorch Lightning model class
                to be used for both the optimization phase and the final experimental run.
        """
        if self.verbose:
            logger.info("Running optimization to find best parameters...")
        best_config, best_value = self.run_optimization(model_class, denoiser_class)

        if self.verbose:
            logger.info(
                "Optimization found best %s: %s", self.optimization_metric, best_value
            )
            logger.info("Best config: %s", best_config.to_dict())
            logger.info("Running final experiment with best parameters...")

        self.experiment_config = best_config

        super().run_experiment(model_class, denoiser_class)
        if self.verbose:
            logger.info("Final experiment with best parameters complete.")
