import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for training")

    # Data and Directories
    parser.add_argument('--data_root', type=str, default=None, help='Path to dataset root')
    parser.add_argument('--audio_dir', type=str, default=None, help='Path to audio data')
    parser.add_argument('--text_dir', type=str, default=None, help='Path to text data')

    # Training Parameters
    parser.add_argument('--random_seed', type=int, default=999, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='Learning rate decay ratio')
    parser.add_argument('--lr_decay_step', type=int, default=70, help='Number of steps before decaying learning rate')

    # Model and Optimization
    parser.add_argument('--language_model', type=str, default='bert', help='Language model to use')
    parser.add_argument('--mj_model', type=str, default='bert', help='MJ model type')
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adamw', 'adam'], help='Optimizer type')
    
    # Hyperparameters
    parser.add_argument('--lam', type=float, default=1.0, help='Lambda hyperparameter')
    parser.add_argument('--version', type=int, default=1, help='Version identifier')

    # Misc
    parser.add_argument('--pca', type=bool, default=True, help='AVMNSIT PCA projection or not')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated list of GPU IDs')
    parser.add_argument('--n_times', type=int, default=1, help='Number of times to repeat the experiment')

    return parser.parse_args()
