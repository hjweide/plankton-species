#!/usr/bin/env python

import cPickle as pickle
import numpy as np

from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from PIL import Image

# the images will be resized to these dimensions
# before computing the code vectors
HEIGHT = 32
WIDTH = 32
CHANNELS = 3


# loads images from disk, computes a PCA and returns the code vectors
def compute_code_vectors(filepaths, n_components, cachefile=None):
    X = np.empty((len(filepaths), HEIGHT * WIDTH * CHANNELS), dtype=np.float32)
    print('loading %d images from disk' % len(filepaths))
    for i, fpath in enumerate(filepaths):
        img = Image.open(fpath).resize((WIDTH, HEIGHT), Image.ANTIALIAS)
        X[i] = np.asarray(img, dtype=np.float32).flatten()

    print('computing code vectors for data of shape %r' % (X.shape,))
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)

    if cachefile is not None:
        print('pickling code vectors to %s' % (cachefile))
        with open(cachefile, 'wb') as f:
            data_dict = {'codes': Z, 'filepaths': filepaths}
            pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)

    return Z


# assigns images to clusters based on their code vectors
def compute_clusters(n_clusters, image_ids, Z):
    mbk = MiniBatchKMeans(
        init='k-means++',
        n_clusters=n_clusters,
        max_iter=1000,
        batch_size=1024, n_init=10, max_no_improvement=None,
        verbose=1, random_state=0
    )

    print('fitting to the code vector matrix of shape %r' % (Z.shape,))
    distances = mbk.fit_transform(Z)
    assignments = mbk.predict(Z)

    print('assigning %d image_ids to %d clusters' % (
        len(image_ids), n_clusters))
    # dict mapping cluster_id to list of (img_id, dist_to_cluster)
    assignment_dict = defaultdict(list)
    image_ids = np.array(image_ids)
    for i in xrange(n_clusters):
        # get the images whose code vectors belong to this cluster
        cluster_ids = image_ids[assignments == i]
        cluster_dists = distances[assignments == i]
        for img_id, img_dist in zip(cluster_ids, cluster_dists):
            #assert np.allclose(img_dist.min(), img_dist[i])
            assignment_dict[i].append((
                img_id,
                img_dist[i],
            ))

    return assignment_dict


# builds the values needed to update the database
def get_query_values(cluster_dict):
    values = []
    for key in cluster_dict:
        for assignment_tuple in cluster_dict[key]:
            image_id = assignment_tuple[0]
            image_dist = assignment_tuple[1]

            # need to be in the same order as the eventual query
            values.append((
                key,
                image_dist,
                image_id,
            ))

    return values
