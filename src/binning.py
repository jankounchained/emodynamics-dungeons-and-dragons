'''
Resampling subtitles
'''

import os
import numpy as np
import pandas as pd
import ndjson
from tqdm import tqdm
from wasabi import msg
from codecarbon import OfflineEmissionsTracker


def timebin(texts, timestamps, bin_freq='10s'):
    '''Concatenate documents into desired timebins.

    Parameters
    ----------
    texts : list
        either untokenized documents as List[Str]
        or tokenized documents as List[List[Str]]
    timestamps : list
        one timebin per document = List[Str]. Same lengths as texts.
    bin_freq : str, optional
        pandas timebin alias, by default '10s'

    Returns
    -------
    dict
        serializable concatenated documents
    '''

    assert len(texts) == len(timestamps)

    if isinstance(texts[0], str):
        # add a trailing space if untokenized documents are passed.
        # otherwise words get merged when concatening with pd.sum
        texts = [doc + ' ' for doc in texts]

    # resample
    df_resampled = (
        pd.DataFrame(texts, index=pd.to_timedelta(timestamps))
        .resample(bin_freq)
        .sum()
        .rename(columns={0: 'text'})
    )

    # get rid of 0 (no document in time bin)
    df_resampled = (df_resampled
                    .replace(0, np.nan)
                    .dropna()
                    )

    # get rid of Timedelta() values in text (no text at timestamp)
    # .sum() copies the index to text if no text at timestamp
    timedelta_index = []
    timedelta_mask = []
    for i, row in df_resampled.iterrows():
        if isinstance(row['text'], pd._libs.tslibs.timedeltas.Timedelta):
            timedelta_index.append(i)
            timedelta_mask.append(True)
        else:
            timedelta_index.append(i)
            timedelta_mask.append(False)

    # print(f'Invalid rows in the mask: {sum(timedelta_mask)}; Total length: {len(timedelta_mask)}')
    if sum(timedelta_mask) > 0:
        timedelta_mask = pd.Series(timedelta_mask, index=timedelta_index)
        df_resampled = df_resampled[~timedelta_mask]

    # get rid of [] (there is a document but no features in time bin)
    df_resampled['text'] = df_resampled.text[df_resampled.text.apply(len) > 0]
    df_resampled = (
        df_resampled
        .dropna()
        .reset_index()
    )

    # get timestamp as str
    # df_resampled['time'] = df_resampled['index'].astype(str).str.extract('days (.*?)\.')
    df_resampled['time'] = df_resampled['index'].astype(
        'str').str.split().str[-1]

    # serialize
    times_bin = df_resampled['time'].tolist()
    texts_bin = df_resampled['text'].tolist()
    file_res = []
    for time, text in zip(times_bin, texts_bin):
        res = dict()
        res.update({
            'time': time,
            'text': text,
        })
        file_res.append(res)

    return file_res


def load_subtitle_file(path):
    '''load parsed subtitles'''
    with open(path) as fin:
        file = ndjson.load(fin)
        timestamps = [doc['start'] for doc in file]
        texts = [doc['text'] for doc in file]

    return texts, timestamps


def main(paths, bin_freq, outdir):
    '''Run timebinning over subtitle files, save resampled files

    Parameters
    ----------
    paths : list
    bin_freq : str
        pandas timebin alias
    outdir : str
        folder to dump resampled files to
    '''

    for path in tqdm(paths):

        # extract features
        texts, timestamps = load_subtitle_file(path)

        # timebins
        binned = timebin(texts, timestamps, bin_freq)

        fname = os.path.basename(path)
        outpath = os.path.join(outdir, fname)

        # very cautios writer
        with open(outpath, 'w') as fout:
            writer = ndjson.writer(fout)
            for line in binned:
                try:
                    writer.writerow(line)
                except TypeError as e:
                    msg.fail(
                        f'Problem serializing the following line: \n {line}')
                    raise e


if __name__ == '__main__':

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--datadir')
    ap.add_argument('-t', '--binfreq')
    ap.add_argument('-o', '--outdir')
    args = vars(ap.parse_args())

    found_paths = sorted(os.listdir(args['datadir']))
    found_paths = [os.path.join(args['datadir'], path) for path in found_paths]
    msg.info(f'Data loader: found {len(found_paths)} input files')

    tracker = OfflineEmissionsTracker(
        country_iso_code="DNK",
        project_name='binning'
        )

    tracker.start()
    main(
        paths=found_paths,
        bin_freq=args['binfreq'],
        outdir=args['outdir']
    )
    emissions = tracker.stop()

    msg.info(f'Files generated in {args["outdir"]}')
    msg.good('Job completed!')
