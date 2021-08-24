# Code taken from Samuel Kolb at https://github.com/samuelkolb/incal/releases
# Given a number of points:
# - Train a DT (scale points?)
# - For every point compute distance to the decision boundary
import sklearn.tree as tree
import pysmt.shortcuts as smt
import numpy as np


def convert(domain, data):
    def _convert(var, val):
        if domain.var_types[var] == smt.BOOL:
            return 1 if val else 0
        elif domain.var_types[var] == smt.REAL:
            return float(val)

    feature_matrix = []
    labels = []
    for instance, label in data:
        feature_matrix.append([_convert(v, instance[v]) for v in domain.variables])
        labels.append(1 if label else 0)
    return feature_matrix, labels


def learn_dt(feature_matrix, labels, **kwargs):
    # noinspection PyArgumentList
    estimator = tree.DecisionTreeClassifier(**kwargs)
    estimator.fit(feature_matrix, labels)
    return estimator


def get_leafs(tree, domain):

    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [domain.variables[i] for i in tree.tree_.feature]
    value = tree.tree_.value
    le = '<='
    g = '>'

    idx = np.argwhere(left == -1)[:, 0]

    IDS=value[idx]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'
        lineage.append((parent, split, threshold[parent], features[parent]))
        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    rules=[]
    for j, child in enumerate(idx):
        rules.append([])
        for node in recurse(left, right, child):
            if len(str(node)) < 3:
                continue
            i = node
            if i[1] == 'l':
                sign = le
            else:
                sign = g
            rules[-1].append([i[3] , sign , i[2]] )
        rules[-1].append(np.argmax(IDS[j]))

    return rules


def get_distances_dt(dt, domain, feature_matrix):
    # Include more features than trained with?

    leave_id = dt.apply(feature_matrix)
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold
    node_indicator = dt.decision_path(feature_matrix)

    distances = []

    for sample_id in range(len(feature_matrix)):
        distance = dict()
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]]
        for node_id in node_index:
            variable = domain.variables[feature[node_id]]
            if leave_id[sample_id] != node_id and domain.var_types[variable] == smt.REAL:
                new_distance = abs(feature_matrix[sample_id][feature[node_id]] - threshold[node_id])
                if variable not in distance or new_distance < distance[variable]:
                    distance[variable] = new_distance
        distances.append(distance)

    return distances


def get_distances(domain, data):
    feature_matrix, labels = convert(domain, data)
    dt = learn_dt(feature_matrix, labels)
    return get_distances_dt(dt, domain, feature_matrix)
    #return get_distances_dtNEW(dt, domain, feature_matrix,labels)

