'''
Run emotion classifier on binned subtitles
'''

import os
import ndjson
from tqdm import tqdm
from wasabi import msg
from codecarbon import OfflineEmissionsTracker

from transformers import pipeline


def setup_pipeline():
    '''load pretrained model from huggingface'''

    classifier = pipeline(
        "text-classification",
        model='bhadresh-savani/distilbert-base-uncased-emotion',
        return_all_scores=True
    )

    return classifier


def classify_file(classifier, texts):
    '''process file'''

    predictions = []
    for doc in texts:
        pred = classifier(doc)
        predictions.append(pred)

    return predictions


def main(classifier, paths, outdir):
    '''process all subtitles & dump results into outdir'''

    for path in tqdm(paths):
        with open(path) as fin:
            file = ndjson.load(fin)
            texts = [doc['text'] for doc in file]

        try:
            predictions = classify_file(
                classifier=classifier,
                texts=texts
            )

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
    args = vars(ap.parse_args())

    found_paths = sorted(os.listdir(args['datadir']))
    found_paths = [os.path.join(args['datadir'], path) for path in found_paths]
    msg.info(f'Sentiment: found {len(found_paths)} input files')

    classifier = setup_pipeline()

    tracker = OfflineEmissionsTracker(
        country_iso_code="DNK",
        project_name='classifier'
        )

    tracker.start()
    main(
        classifier=classifier,
        paths=found_paths,
        outdir=args['outdir']
    )
    emissions = tracker.stop()

    msg.info(f'Files generated in {args["outdir"]}')
    msg.good('Job completed!')
