import numpy as np
import scipy.sparse as sp
import torch
import json
import os
import pickle
import os.path


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    if dataset == "cora":
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        print(features.shape)
        labels = encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test

    if dataset == "reddit":
        return process_reddit()


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def process_reddit():
    ##################################################
    ########### FROM GraphSage CODE ##################
    ##################################################
    import networkx as nx
    from networkx.readwrite import json_graph
    version_info = list(map(int, nx.__version__.split('.')))
    major = version_info[0]
    minor = version_info[1]
    assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

    print("Loading Reddit Data ...")
    G_data = json.load(open("../data/reddit/reddit-G.json"))

    print("Building Graph ...")
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    print("Build Features ...")

    if os.path.exists("../data/reddit/reddit-feats.npy"):
        feats = np.load("../data/reddit/reddit-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open("../data/reddit/reddit-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open("../data/reddit/reddit-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data... now preprocessing...")

    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    ##################################################
    ########### END GraphSage CODE ##################
    ##################################################

    print("Prepare for GCN")
    N = len(feats) # number of nodes

    print("Process Adj Matrix")

    # Just build symmetric adjacency matrix from scratch
    i = []
    j = []
    for edge in G.edges():
        src_idx = id_map[edge[0]]
        dest_idx = id_map[edge[1]]

        i.append(src_idx)
        j.append(dest_idx)

        i.append(dest_idx)
        j.append(src_idx)

    d = [1 for x in range(len(i))]

    adj = sp.coo_matrix((d, (i, j)))
    adj = normalize(adj + sp.eye(N))

    if not os.path.exists("../data/reddit/reddit-features.pickle") or not os.path.exists("../data/reddit/reddit-labels.pickle"):
        print("Process Features and Labels")

        features = []
        labels = []
        for key in id_map:
            features.append(feats[id_map[key]])
            labels.append(class_map[key])

        with open('../data/reddit/reddit-features.pickle', 'wb') as f:
            pickle.dump(list(features), f, pickle.HIGHEST_PROTOCOL)

        with open('../data/reddit/reddit-labels.pickle', 'wb') as f:
            pickle.dump(list(labels), f, pickle.HIGHEST_PROTOCOL)
    else:
        print("Load Features and Labels")     
        
        with open('../data/reddit/reddit-features.pickle', 'rb') as f:
            features = np.array(pickle.load(f))

        with open('../data/reddit/reddit-labels.pickle', 'rb') as f:
            labels = np.array(pickle.load(f))
        
    features = normalize(sp.csr_matrix(features))

    # 5% of data marked for training only
    # 20% of data for validation
    # remaining 70% of data analyzed at the end
    idx_train = range(int(0.05*N))
    idx_val = range(int(0.05*N), int(0.25*N))
    idx_test = range(int(0.25*N), N)

    print("Convert to PyTorch Data Structures")

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.array(labels))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print("Done Loading Data")

    return adj, features, labels, idx_train, idx_val, idx_test