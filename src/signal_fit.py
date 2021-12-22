import os

import ndjson
import numpy as np
from wasabi import msg
from tqdm import tqdm
import matplotlib.pyplot as plt
from codecarbon import OfflineEmissionsTracker

from nolds import dfa
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from entropies import InfoDynamics
from entropies import jsd
from util import load_transform_sentiment


def calc_ntr(emotion_matrix, window, visualize=False):
    '''Calculate Novelty, Transience & Resonance in a given window 

    Parameters
    ----------
    emotion_matrix : np.array
        array of shape (n_documents, n_labels)
    window : int
        n documents to look before/after document[i]
    visualize : bool, optional
        enable diagnostics plot for novelty? By default False

    Returns
    -------
    entropies.InfoDynamics
        trained instance of infodynamics class
    '''

    idmdl = InfoDynamics(
        data=emotion_matrix,
        time=None,
        window=window,
        sort=False
    )

    idmdl.novelty(meas=jsd)
    idmdl.transience(meas=jsd)
    idmdl.resonance(meas=jsd)

    if visualize:
        plt.plot(idmdl.nsignal)

    return idmdl


def curb_incomplete_signal(timeseries, window):
    '''remove first & last {window} documents'''
    return timeseries[window:-window]


def calculate_resonance_novelty_slope(resonance, novelty):
    '''get slope of resonance ~ novelty linear model
    a) standardize
    b) fit a simple linear regression
    c) extract beta coefficient

    Parameters
    ----------
    resonance : np.array-like
    novelty : np.array-like

    Returns
    -------
    float
        slope of lm(resonance ~ novelty)
    '''

    # reshape
    novelty = novelty.reshape(-1, 1)
    resonance = resonance.reshape(-1, 1)

    # standardize resonance & novelty
    z_novelty = StandardScaler().fit_transform(
        novelty
    )

    z_resonance = StandardScaler().fit_transform(
        resonance
    )

    # fit model
    lm = LinearRegression(fit_intercept=False)
    lm.fit(X=z_novelty, y=z_resonance)

    # capture slope
    return lm.coef_[0][0]


def main(paths, window, length_threshold):
    '''Get scores for files, generate results file

    Parameters
    ----------
    paths : list
        paths to emotion classification matrices
    window : int
        n documents to look before/after document[i]
    length_threshold : int
        minimum number of datapoints the timeseries should have
        to be processed

    Returns
    -------
    List[dict]
        where rows are documents, key-value pairs are variable-values.
    '''

    results = []
    for path in tqdm(paths):
        emotions, tag = load_transform_sentiment(path)

        if len(emotions) >= length_threshold:
            model = calc_ntr(emotions, window=window)

            novelty = curb_incomplete_signal(model.nsignal, window=window)
            resonance = curb_incomplete_signal(model.rsignal, window=window)

            slope = calculate_resonance_novelty_slope(resonance, novelty)

            H = dfa(novelty, overlap=True)

            report_episode = {
                'episode': int(tag),
                'hurst': H,
                'rn_slope': slope,
                'N_mean': np.mean(novelty),
                'N_std': np.std(novelty),
                'n_datapoints': len(model.nsignal)
            }

            results.append(report_episode)

        # catch error with episodes {2444, 396, 516}.ndjson
        else:
            msg.fail(f'episode too short: {path}')

    return results


if __name__ == "__main__":

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--signaldir', type=str)
    ap.add_argument('-w', '--window', type=int)
    ap.add_argument('-t', '--lengththreshold', type=int)
    ap.add_argument('-o', '--outpath', type=str)
    args = vars(ap.parse_args())

    if not os.path.exists(args['signaldir']):
        raise FileNotFoundError(f"{args['signaldir']} doesn't exist!")

    signal_paths = sorted(
        [os.path.join(args['signaldir'], path)
         for path in os.listdir(args['signaldir'])]
    )

    if len(signal_paths) < 1:
        raise FileNotFoundError(f"{args['signaldir']} seems to be empty!")


    tracker = OfflineEmissionsTracker(
        country_iso_code="DNK",
        project_name='entropy_signal'
        )

    tracker.start()
    results = main(
        paths=signal_paths,
        window=args['window'],
        length_threshold=args['lengththreshold']
    )
    emissions = tracker.stop()

    with open(args['outpath'], 'w') as fout:
        ndjson.dump(results, fout)
