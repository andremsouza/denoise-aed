from datetime import datetime
import gc
import logging

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch

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
        self,
        trial: optuna.Trial,
        model_class: type[pl.LightningModule],
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
        # Create a configuration from the trial
        trial_config = ExperimentConfig.from_trial(trial, self.experiment_config)

        # Update experiment config temporarily
        original_config = self.experiment_config
        self.experiment_config = trial_config

        if self.verbose:
            logger.info("Trial %s - Testing parameters: %s", trial.number, trial.params)

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
            exp_loggers = self._setup_loggers(trial_experiment_name)

            # Create pruning callback for Optuna
            self.pruning_callback = PyTorchLightningPruningCallback(
                trial, monitor=self.optimization_metric
            )

            # Create early stopping callback with shorter patience for trials
            early_stopping = EarlyStopping(
                monitor=self.optimization_metric,
                mode="max" if self.direction == "maximize" else "min",
                patience=trial_config.training.early_stopping_patience,
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
            max_epochs = min(trial_config.training.max_epochs, 20)
            trainer = pl.Trainer(
                callbacks=[checkpoint_callback, early_stopping, self.pruning_callback],
                max_epochs=max_epochs,
                logger=exp_loggers,
                log_every_n_steps=min(50, len(train_dataloader)),
                enable_checkpointing=True,
                enable_progress_bar=self.verbose,
            )

            # Setup MLflow tracking
            self._setup_mlflow_tracking()

            # Run training
            try:
                if self.experiment_config.infrastructure.use_mlflow:
                    with mlflow.start_run(
                        run_name=f"{trial_experiment_name}",
                        tags={"owner": "André Moreira Souza"},
                        log_system_metrics=self.experiment_config.infrastructure.mlflow_log_system_metrics,
                        nested=True,
                        parent_run_id=parent_run_id,
                    ):
                        mlflow.log_param("run_name", trial_experiment_name)
                        mlflow.log_params(self.experiment_config.to_dict(flat=True))
                        mlflow.log_params(trial.params)
                        mlflow.log_param("trial_number", trial.number)
                        trainer.fit(model, train_dataloader, test_dataloader)
                else:
                    trainer.fit(model, train_dataloader, test_dataloader)
            except optuna.exceptions.TrialPruned as e:
                if self.verbose:
                    logger.info("Trial %s pruned: %s", trial.number, str(e))
                # Clean up resources
                local_vars = locals()
                for var in ["model", "train_dataloader", "test_dataloader", "trainer"]:
                    if var in local_vars and local_vars[var] is not None:
                        del local_vars[var]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                raise e

            # Get best score
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

        except (
            FileNotFoundError,
            ValueError,
            RuntimeError,
            torch.cuda.OutOfMemoryError,
            OSError,
            TypeError,
            IndexError,
        ) as e:
            if self.verbose:
                logger.error(
                    "Trial %s failed with error: %s",
                    trial.number,
                    str(e),
                    exc_info=True,
                )
            local_vars = locals()
            for var in ["model", "train_dataloader", "test_dataloader", "trainer"]:
                if var in local_vars and local_vars[var] is not None:
                    del local_vars[var]
            return 0.0 if self.direction == "maximize" else float("inf")

        finally:
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
            logger.info("Starting optimization with %s trials", self.n_trials)
            logger.info("Optimizing %s to %s", self.optimization_metric, self.direction)

        # Create or load study
        study = optuna.create_study(
            study_name=f"{self.study_name}_{model_class.__name__}",
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

        # Set up MLflow tracking
        if self.experiment_config.infrastructure.use_mlflow:
            opt_run_name = (
                f"{self.experiment_prefix}_{model_class.__name__}"
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

                run_id = mlflow.active_run().info.run_id
                logger.info("MLflow run ID: %s", run_id)

                study.optimize(
                    lambda trial: self._objective(
                        trial, model_class, parent_run_id=run_id
                    ),
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

                mlflow.log_params(study.best_params)
                mlflow.log_metric(f"best_{self.optimization_metric}", study.best_value)
                mlflow.set_tag("best_trial_number", study.best_trial.number)
        else:
            study.optimize(
                lambda trial: self._objective(trial, model_class),
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
            optuna.trial.FixedTrial(study.best_params), self.experiment_config
        )

        if self.verbose:
            logger.info("Optimization complete")

        return best_config, study.best_value

    def run_with_best_params(self, model_class: type[pl.LightningModule]) -> None:
        """Run a full experiment with the best parameters from optimization.

        Args:
            model_class: PyTorch Lightning model class to use
        """
        if self.verbose:
            logger.info("Running optimization to find best parameters...")
        best_config, best_value = self.run_optimization(model_class)

        if self.verbose:
            logger.info(
                "Optimization found best %s: %s", self.optimization_metric, best_value
            )
            logger.info("Best config: %s", best_config.to_dict())
            logger.info("Running final experiment with best parameters...")

        self.experiment_config = best_config

        super().run_experiment(model_class)
        if self.verbose:
            logger.info("Final experiment with best parameters complete.")
