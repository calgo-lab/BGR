import pandas as pd
from bgr.soil.training_args import TrainingArgs

class ExperimentRunner:
    """
    The ExperimentRunner class is responsible for managing and executing the training, validation, and testing 
    of a machine learning model. It handles the creation of the model, the execution of the training process, 
    the evaluation on the validation set, and the final testing.
    """
    
    def __init__(
            self,
            experiment_type: str,
            train_data: pd.DataFrame, 
            val_data: pd.DataFrame, 
            test_data: pd.DataFrame,
            target: str,
            wandb_project_name : str,
            num_experiment_runs: int = 1,
            seed: int = None,
            wandb_image_logging: bool = False
        ):
        """
        Initializes the ExperimentRunner with the given parameters.
        """
        
        self.experiment_type = experiment_type
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.target = target
        self.wandb_project_name = wandb_project_name
        self.num_experiment_runs = num_experiment_runs
        self.seed = seed
        self.wandb_image_logging = wandb_image_logging
        
    def run_train_val_test(
        training_args: TrainingArgs,
        model_output_dir: str,
        wandb_offline: bool = False
    ):
        # TODO: Implement this function
        
        pass