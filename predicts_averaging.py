import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_args(args):
    parser = argparse.ArgumentParser(description='averaging several predicts')
    parser.add_argument('--source', type=str)
    parser.add_argument('--out_name', type=str)
    parser.add_argument('--n_classes', type=int, default=26)

    return parser.parse_args(args)

def main(args):
    names = args.source.split(';')
    for i_name, name in enumerate(tqdm(names, desc='keel calm')):
        if i_name == 0:
            Id = df.Id.tolist()
            final_scores = np.zeros((len(Id), args.n_classes))
        for i_score, sample_score in enumerate(df.scores):
            final_scores[i_score] += np.array(eval(sample_score))

    Category = np.argmax(final_scores, axis=1)
    out_df = pd.DataFrame({'Id': Id, 'Category': Category})
    out_df.to_csv(args.out_name, index=None)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
