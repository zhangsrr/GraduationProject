"""Computation of the dissimilarity representation of a set of objects
(streamlines) from a set of prototypes (streamlines) given a distance
function. Some prototype selection algorithms are available.

See Olivetti E., Nguyen T.B., Garyfallidis, E., The Approximation of
the Dissimilarity Projection, http://dx.doi.org/10.1109/PRNI.2012.13
"""

from __future__ import division
import numpy as np
from dipy.tracking.distances import bundles_distances_mam
try:
    from joblib import Parallel, delayed, cpu_count
    joblib_available = True
except:
    joblib_available = False


def furthest_first_traversal(tracks, k, distance, permutation=True):
    """This is the farthest first traversal (fft) algorithm which
    selects k streamlines out of a set of streamlines (tracks). This
    algorithms is known to be a good sub-optimal solution to the
    k-center problem, i.e. the k streamlines are sequentially selected
    in order to be far away from each other.

    Parameters
    ----------

    tracks : list or array of objects
        an iterable of streamlines.
    k : int
        the number of streamlines to select.
    distance : function
        a distance function between groups of streamlines, like
        dipy.tracking.distances.bundles_distances_mam
    permutation : bool
        True if you want to shuffle the streamlines first. No
        side-effect.

    Return
    ------
    idx : array of int
        an array of k indices of the k selected streamlines.

    Notes
    -----
    - Hochbaum, Dorit S. and Shmoys, David B., A Best Possible
    Heuristic for the k-Center Problem, Mathematics of Operations
    Research, 1985.
    - http://en.wikipedia.org/wiki/Metric_k-center

    See Also
    --------
    subset_furthest_first


    """
    print("pre tracks: ", tracks)

    if permutation:
        idx = np.random.permutation(len(tracks))  # shuffle tracks
        tracks = tracks[idx]
        print("tracks: ", tracks)
    else:
        idx = np.arange(len(tracks), dtype=np.int)
        print("idx: ", idx)

    T = [0]
    while len(T) < k:
        z = distance(tracks, tracks[T]).min(1).argmax()  # min(1) 返回每一行最小值组成的数组, argmax()返回数组中的最大值所在的下标
        T.append(z)

    return idx[T]  # 一个嵌套的array


def subset_furthest_first(tracks, k, distance, permutation=True, c=2.0):
    """The subset furthest first (sff) algorithm is a stochastic
    version of the furthest first traversal (fft) algorithm. Sff
    scales well on large set of objects (streamlines) because it
    does not depend on len(tracks).

    Parameters
    ----------

    tracks : list or array of objects
        an iterable of streamlines.
    k : int
        the number of streamlines to select.
    distance : function
        a distance function between groups of streamlines, like
        dipy.tracking.distances.bundles_distances_mam
    permutation : bool
        True if you want to shuffle the streamlines first. No
        side-effect.
    c : float
        Parameter to tune the probability that the random subset of
        streamlines is sufficiently representive of tracks. Typically
        2.0-3.0.

    Return
    ------
    idx : array of int
        an array of k indices of the k selected streamlines.

    See Also
    --------
    furthest_first_traversal

    Notes
    -----
    See: E. Olivetti, T.B. Nguyen, E. Garyfallidis, The Approximation
    of the Dissimilarity Projection, Proceedings of the 2012
    International Workshop on Pattern Recognition in NeuroImaging
    (PRNI), pp.85,88, 2-4 July 2012 doi:10.1109/PRNI.2012.13
    """
    size = int(max(1, np.ceil(c * k * np.log(k))))
    if permutation:
        idx = np.random.permutation(len(tracks))[:size]
    else:
        idx = range(size)

    return idx[furthest_first_traversal(tracks[idx],
                                        k, distance,
                                        permutation=False)]


def dissimilarity(tracks, prototypes, distance=bundles_distances_mam,
                  n_jobs=-1, verbose=False):
    """Compute the dissimilarity (distance) matrix between tracks and
    given prototypes. This function supports parallel (multicore)
    computation.

    Parameters
    ----------
    tracks : list or array of objects
           an iterable of streamlines.
    prototypes : iterable of objects
           The prototypes.
    distance : function
           Distance function between groups of streamlines.
    prototype_policy : string
           Shortname for the prototype selection policy. The default
           value is 'sff'.
    n_jobs : int
           If joblib is available, split the dissimilarity computation
           in n_jobs. If n_jobs is -1, then all available cpus/cores
           are used. The default value is -1.
    verbose : bool
           If true prints some messages. Deafault is True.

    Return
    ------
    dissimilarity_matrix : array (N, num_prototypes)

    See Also
    --------
    furthest_first_traversal, subset_furthest_first

    Notes
    -----
    """
    if verbose:
        print("Computing the dissimilarity matrix.")

    if joblib_available and n_jobs != 1:
        if n_jobs is None or n_jobs == -1:
            n_jobs = cpu_count()

        if verbose:
            print("Parallel computation of the dissimilarity matrix: %s cpus." % n_jobs)

        if n_jobs > 1:
            tmp = np.linspace(0, len(tracks), n_jobs + 1).astype(np.int)
        else:  # corner case: joblib detected 1 cpu only.
            tmp = (0, len(tracks))

        chunks = zip(tmp[:-1], tmp[1:])
        dissimilarity_matrix = np.vstack(Parallel(n_jobs=n_jobs)(delayed(distance)(tracks[start:stop], prototypes) for start, stop in chunks))
    else:
        dissimilarity_matrix = distance(tracks, prototypes)  # tracks是原始流线数据，prototypes是选择后的流线数据，distance的实现呢？

    if verbose:
        print("Done.")

    return dissimilarity_matrix


def compute_dissimilarity(tracks, num_prototypes=40,
                          distance=bundles_distances_mam,
                          prototype_policy='sff',
                          single_thread = False,
                          n_jobs=-1,
                          verbose=False):
    """Compute the dissimilarity (distance) matrix between tracks and
    prototypes, where prototypes are selected among the tracks with a
    given policy.

    Parameters
    ----------
    tracks : list or array of objects
           an iterable of streamlines.
    num_prototypes : int
           The number of prototypes. In most cases 40 is enough, which
           is the default value.
    distance : function
           Distance function between groups of streamlines. The
           default is bundles_distances_mam
    prototype_policy : string
           Shortname for the prototype selection policy. The default
           value is 'sff'.
    n_jobs : int
           If joblib is available, split the dissimilarity computation
           in n_jobs. If n_jobs is -1, then all available cpus/cores
           are used. The default value is -1.
    verbose : bool
           If true prints some messages. Deafault is True.

    Return
    ------
    dissimilarity_matrix : array (N, num_prototypes)

    See Also
    --------
    furthest_first_traversal, subset_furthest_first

    Notes
    -----
    """
    global joblib_available
    joblib_available = not single_thread
    
    if verbose:
        print("Generating %s prototypes with policy %s." % (num_prototypes, prototype_policy))

    if prototype_policy == 'random':
        prototype_idx = np.random.permutation(len(tracks))[:num_prototypes]
    elif prototype_policy == 'fft':
        prototype_idx = furthest_first_traversal(tracks,
                                                 num_prototypes, distance)
    elif prototype_policy == 'sff':
        prototype_idx = subset_furthest_first(tracks, num_prototypes, distance)  # num_prototypes = 要生成的流线数目
    else:
        if verbose:
            print("Prototype selection policy not supported: %s" % prototype_policy)

        raise Exception

    prototypes = [tracks[i] for i in prototype_idx]  # prototype_idx是list，里面存的是选择的流线下标，则prototypes存的就是经distance处理选择的流线
    dissimilarity_matrix = dissimilarity(tracks, prototypes, distance,
                                         n_jobs=n_jobs, verbose=verbose)
    return dissimilarity_matrix, prototype_idx
