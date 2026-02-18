import argparse

def build_arg_parser():
    """
    Build argument parser for training.
    
    Returns:
        argparse.ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description='Train MLP and TabM models on medical tabular data')
    parser.add_argument('--exp-path', type=str, default='configs/lvef/__base__/exp_path.yaml',
                       help='Experiment paths YAML')
    parser.add_argument('--train-args', type=str, default='configs/lvef/__base__/train_argument.yaml',
                       help='Training arguments YAML')
    parser.add_argument('--subset', type=str, default=None,
                       help='Run only this subset (e.g. full, with_intervention, without_intervention, lab_only). If not set, all subsets are run.')
    parser.add_argument('--gw-weight', type=float, default=None,
                       help='Override training gw_weight (Gromov-Wasserstein loss weight). Example: --gw-weight 0.01')
    parser.add_argument('--lambda-ot', type=float, default=None,
                       help='Override WAE lambda_ot (OT regularization weight). Example: --lambda-ot 0.01')
    return parser
