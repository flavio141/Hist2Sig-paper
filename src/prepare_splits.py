import os
import argparse
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, train_test_split

parser = argparse.ArgumentParser(description='Train-Val-Test Cross-Validation split')
parser.add_argument('--csv_path', type=str, default='signature/Signatures_DB.csv', help='Path to the Signatures_DB.csv file')
parser.add_argument('--output_dir', type=str, default='splits', help='Directory to save the splits')
parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split')
parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for cross-validation')
parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')


def split_dataset(csv_path, output_dir, test_size=0.2, n_splits=5, random_state=42):
    signature = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    case_id_mut = signature.groupby('Case ID').first().reset_index()
    case_id_feat = ["-".join(feature.split('-')[:3])for feature in os.listdir('features') if feature.endswith('.pt')]

    case_id_tmp = set(case_id_mut['Case ID']).intersection(set(case_id_feat))
    case_id = case_id_mut[case_id_mut['Case ID'].isin(case_id_tmp)].reset_index(drop=True)

    train_cases, test_cases = train_test_split(
        case_id,
        test_size=test_size,
        stratify=case_id['Project ID'],
        random_state=random_state
    )

    test_samples = test_cases['Case ID']
    test_samples.to_frame(name='test').to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"Test set: {len(test_samples)} samples")

    train_df = signature[signature['Case ID'].isin(train_cases['Case ID'])].reset_index(drop=True)

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['Project ID'], groups=train_df['Case ID'])):
        train_samples = train_df.loc[train_idx, 'Case ID']
        val_samples = train_df.loc[val_idx, 'Case ID']

        split_df = pd.DataFrame({
            'train': pd.Series(train_samples.tolist()),
            'val': pd.Series(val_samples.tolist())
        })

        split_df.to_csv(os.path.join(output_dir, f'{fold}.csv'), index=False)

        print(f"Fold {fold}: {len(train_samples)} train samples, {len(val_samples)} val samples")

    print(f"\nSplits saved in {output_dir}")


if __name__ == '__main__':
    args = parser.parse_args()

    split_dataset(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        n_splits=args.n_splits,
        random_state=args.random_state
    )
