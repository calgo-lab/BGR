import argparse
from pathlib import Path
from ast import literal_eval
import sys
from datetime import datetime

import torch

from bgr.soil.data.horizon_tabular_data import HorizonDataProcessor
from bgr.soil.experiment_runner import ExperimentRunner
from bgr.soil.training_args import TrainingArgs
import pytz

def create_parser() -> argparse.ArgumentParser:
    """
    The `create_parser` function defines and returns an ArgumentParser object with default parameters
    for a machine learning model training script. It includes arguments related to data, directories, 
    model, training, and hyperparameter optimization.

    Returns
    -------
    argparse.ArgumentParser
        An ArgumentParser object with default parameters for a machine learning model training script.
    """
    # For lists
    def parse_list(arg, length=None, dtype=None):
        value = literal_eval(arg)
        
        if not isinstance(value, list):
            raise ValueError('Argument must be a list.')
        if length is not None and len(value) != length:
            raise ValueError(f'List must have length {length}.')
        if dtype is not None and not all(isinstance(i, dtype) for i in value):
            raise ValueError(f'List must contain {dtype} values.')
        
        return value
    
    parser = argparse.ArgumentParser()
    
    # arguments from file
    parser.add_argument('--args_file', type=str, default=None, help='Path to a file containing arguments')

    # data-related parameters
    parser.add_argument('--data_folder_path', type=str, default='../data/BGR/')
    parser.add_argument('--target', type=str, default='Horizontsymbol_relevant') #TODO: Do we need multiple targets?
    parser.add_argument('--train_val_test_frac', type=lambda arg: parse_list(arg, length=3, dtype=float),
        default=[0.6, 0.2, 0.2])
    parser.add_argument('--label_embedding_path', type=str, default='./BGR/label_embeddings/all_horizons_embeddings.pickle')

    # dir-related parameters
    parser.add_argument('--model_output_dir', type=str, default='model_output')
    parser.add_argument('--inference_model_file', type=str, default=None)

    # experiment-related parameters
    parser.add_argument('--experiment_type', type=str, default='depth_experiment') #TODO: Add default experiment type
    parser.add_argument('--wandb_online', dest='wandb_offline', action='store_false') # wandb_offline defaults to True if not specified
    parser.add_argument('--wandb_project_name', type=str, default='BGR_debugging')
    parser.add_argument('--wandb_plot_logging', action='store_true') # Defaults to false if not specified

    # training-related parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--no_save_checkpoints', dest='save_checkpoints', action='store_false') # save_checkpoints defaults to true if not specified
    parser.add_argument('--no_early_stopping', dest='use_early_stopping', action='store_false') # use_early_stopping defaults to true if not specified
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4)
    
    # hpo-related parameters
    
    return parser

def parse_unknown_args(unknown_args: list) -> dict:
    """
    Parses a list of unknown arguments and converts them into a dictionary.

    Parameters
    ----------
    unknown_args : list
        A list of unknown arguments, typically from the command line.

    Returns
    -------
    dict
        A dictionary where the keys are argument names (without the '--' prefix) and the values
        are the corresponding argument values.
    """
    unknown_dict = {}
    key = None
    for arg in unknown_args:
        if arg.startswith('--'):
            key = arg.lstrip('--').split('=')[0]
            value = arg.split('=')[-1]
            
            try:
                # Use ast.literal_eval to convert the value to its appropriate type
                unknown_dict[key] = literal_eval(value)
            except:
                # If literal_eval fails, keep the value as a string
                unknown_dict[key] = value
            
    return unknown_dict

def main(args : argparse.Namespace):
    """
    The main function performs various tasks such as creating model output path, loading and splitting data, 
    creating TimeSeriesDataset, executing train, validation, and test for different model arguments, 
    performing hyperparameter optimization if enabled, and training, validating and testing the model.

    Args:
        args (argparse.Namespace): Namespace object containing all the arguments required for the execution of the main function.
    """
    # Create model output path
    berlin_tz = pytz.timezone('Europe/Berlin')
    timestamp = datetime.now(berlin_tz).strftime("%Y-%m-%d_%H-%M-%S")
    args.model_output_dir = f"{args.model_output_dir}/{args.experiment_type}_{timestamp}"
    Path(args.model_output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    dataprocessor = HorizonDataProcessor(args.label_embedding_path, args.data_folder_path)
    horizon_data = dataprocessor.load_processed_data()
    
    # Split data
    train_data, val_data, test_data = dataprocessor.multi_label_stratified_shuffle_split(horizon_data, train_val_test_frac=args.train_val_test_frac, random_state=args.seed)
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Get TrainingArgs object
    training_args = TrainingArgs.create_from_args(args)
    
    # Experimenter for executing train, val, test for different model arguments
    experimenter = ExperimentRunner(
        args.experiment_type,
        train_data, 
        val_data, 
        test_data,
        dataprocessor,
        target = args.target,
        seed = args.seed,
        wandb_project_name = args.wandb_project_name,
        wandb_plot_logging = args.wandb_plot_logging
    )
    
    if args.inference_model_file:
        # Run inference on the test data using a pre-trained model
        # TODO: Will probably not work because of hyperparameters
        test_metrics = experimenter.run_inference(training_args, args.inference_model_file, timestamp, wandb_offline=args.wandb_offline)
    else:
        # Train, validate and test the model according to the model arguments
        metrics = experimenter.run_train_val_test(training_args, args.model_output_dir, timestamp, wandb_offline=args.wandb_offline)

def read_and_handle_args():
    """
    This function reads the args, parses them. It also handles args to read from file and unknown args.
    Args given in by the command line will override the args given in the file.

    Returns
    -------
    Namespace
        The argument namespace object containing all the given arguments for the script.
    """
    parser = create_parser()
    args, unknown = parser.parse_known_args(sys.argv[1:])
    
    # If args_file is provided, read the argument from the file
    if args.args_file and args.args_file != "":
        with open(args.args_file, 'r') as file:
            file_args = [line.strip() for line in file if line.strip() and not line.startswith("#")]
        args, unknown = parser.parse_known_args(file_args + sys.argv[1:])
        print('Arguments read from file.')

    # Parse unknown args and add them to the args namespace
    unknown_dict = parse_unknown_args(unknown)
    for key, value in unknown_dict.items():
        setattr(args, key, value)
    
    return args

if __name__ == "__main__":
    args = read_and_handle_args()
    main(args)