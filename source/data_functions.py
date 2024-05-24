import pandas as pd

def print_df_stats(df: pd.DataFrame, save_path=None) -> None:
    n_samples, n_features = df.shape
    if 'class' in df.columns:
        pass
    elif 'target' in df.columns:
        pass
    else:
        raise ValueError("Bad task! Check columns for: 'class' or 'target'")
    task = 'Classification' if 'class' in df.columns else 'Regression'
    print(
        f'# Samples = {n_samples}\n'
        f'# Features = {n_features - 1}\n'
        f'Task: {task}\n'
    )
    print(df.head(3))
    if save_path is not None:
        pd.Series(
            {
                'n_samples': n_samples,
                'n_features': n_features,
                'task': task,
            }
        ).to_csv(save_path)
