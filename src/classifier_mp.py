'''
Run emotion classifier on binned subtitles
'''

import os
import ndjson
from tqdm import tqdm
from wasabi import msg
from codecarbon import OfflineEmissionsTracker
from mpire import WorkerPool

from transformers import pipeline


def setup_pipeline():
    '''load pretrained model from huggingface'''

    classifier = pipeline(
        "text-classification",
        model='bhadresh-savani/distilbert-base-uncased-emotion',
        return_all_scores=True
    )

    return classifier


# def classify_file(classifier, texts):
#     '''process file'''

#     predictions = []
#     for doc in texts:
#         pred = classifier(doc)
#         predictions.append(pred)

#     return predictions


def main(classifier, paths, outdir):
    '''process all subtitles & dump results into outdir'''

    for path in tqdm(paths):
        with open(path) as fin:
            file = ndjson.load(fin)
            texts = [doc['text'] for doc in file]

        try:
            predictions = [classifier(doc) for doc in texts]

            fname = os.path.basename(path)
            outpath = os.path.join(outdir, fname)

            with open(outpath, 'w') as fout:
                ndjson.dump(predictions, fout)

        # catch an unknown error with episode 372.ndjson
        except IndexError:
            msg.fail(f'Episode failed: {path}')


def parallel_main(paths, outdir='data/test_run/representations'):
    '''
    process all subtitles & dump results into outdir
    modification of main method for parallel processing
    '''
    import os
    import ndjson
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model='bhadresh-savani/distilbert-base-uncased-emotion',
        return_all_scores=True
    )

    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        with open(path) as fin:
            file = ndjson.load(fin)
            texts = [doc['text'] for doc in file]

        try:
            predictions = [classifier(doc) for doc in texts]

            fname = os.path.basename(path)
            outpath = os.path.join(outdir, fname)

            with open(outpath, 'w') as fout:
                ndjson.dump(predictions, fout)

        # catch an unknown error with episode 372.ndjson
        except IndexError:
            msg.fail(f'Episode failed: {path}')


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--datadir')
    ap.add_argument('-o', '--outdir')
    ap.add_argument('-j', '--njobs', required=False, default=1)
    args = vars(ap.parse_args())

    found_paths = sorted(os.listdir(args['datadir']))
    found_paths = [os.path.join(args['datadir'], path) for path in found_paths]
    msg.info(f'Sentiment: found {len(found_paths)} input files')

    # # init emission tracker
    # tracker = OfflineEmissionsTracker(
    #     country_iso_code="DNK",
    #     project_name='classifier'
    # )

    # tracker.start()

    n_jobs = int(args['njobs'])
    msg.info(f'starting {n_jobs} jobs')
    # run in parallel
    # def parallel_main_fixed_args(paths):
    #     parallel_main(outdir=args['outdir'], paths=paths)

    with WorkerPool(n_jobs=n_jobs, start_method='spawn') as pool:
        pool.map(parallel_main, found_paths, progress_bar=True)

    # emissions = tracker.stop()

    msg.info(f'Files generated in {args["outdir"]}')
    msg.good('Job completed!')
