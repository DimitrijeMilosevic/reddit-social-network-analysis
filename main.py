import collections
from itertools import chain, combinations
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import powerlaw
from scipy.cluster.hierarchy import dendrogram

submission_data_path = 'data/submissions'
comment_data_path = 'data/comments'
cleansed_submission_data_path = 'data/submissions_cleansed/submissions'
cleansed_comment_data_path = 'data/comments_cleansed/comments'
graphs_path = 'graphs'
dendrograms_path = 'dendrograms'


def create_secondary_dataset():
    submissions_secondary_dataset = pd.DataFrame()
    comments_secondary_dataset = pd.DataFrame()
    for file in os.listdir(submission_data_path):
        submission_data = pd.read_csv(f"{submission_data_path}/{file}")
        comment_data = pd.read_csv(f"{comment_data_path}/{file}")
        cleansed_submission_data = \
            submission_data[submission_data['author'] != '[deleted]']
        submissions_secondary_dataset = \
            pd.concat([submissions_secondary_dataset,
                       cleansed_submission_data])
        cleansed_comment_data = \
            comment_data[comment_data['author'] != '[deleted]']
        comments_secondary_dataset = \
            pd.concat([comments_secondary_dataset, cleansed_comment_data])
    with open(cleansed_submission_data_path, 'wb') as cleansed_file:
        pickle.dump(submissions_secondary_dataset, cleansed_file)
    with open(cleansed_comment_data_path, 'wb') as cleansed_file:
        pickle.dump(comments_secondary_dataset, cleansed_file)


def number_of_unique_subreddits():
    with open(cleansed_submission_data_path, 'rb') as cleansed_submission_data, \
            open(cleansed_comment_data_path, 'rb') as cleansed_comment_data:
        submission_data = pickle.load(cleansed_submission_data)
        comment_data = pickle.load(cleansed_comment_data)
        subreddit_data = pd.concat([submission_data['subreddit'], comment_data['subreddit']])
        print(f"Number of unique subreddits: {subreddit_data.nunique()}.")


def subreddits_with_most_users():
    with open(cleansed_submission_data_path, 'rb') as cleansed_submission_data, \
            open(cleansed_comment_data_path, 'rb') as cleansed_comment_data:
        submission_data = pickle.load(cleansed_submission_data)
        comment_data = pickle.load(cleansed_comment_data)
        users_data = pd.concat([submission_data[['subreddit', 'author']], comment_data[['subreddit', 'author']]])
        users_by_subreddit = users_data.groupby('subreddit') \
            .agg('nunique') \
            .rename(columns={'author': 'number_of_authors'})
        print(users_by_subreddit.sort_values('number_of_authors', ascending=False).head(5))


def subreddits_with_most_comments():
    with open(cleansed_submission_data_path, 'rb') as cleansed_submission_data:
        submission_data = pickle.load(cleansed_submission_data)
        comments_by_submission = submission_data[['subreddit', 'num_comments']] \
            .groupby('subreddit') \
            .agg('sum')
        print(comments_by_submission.sort_values('num_comments', ascending=False).head(5))


def mean_number_of_subreddit_users():
    with open(cleansed_submission_data_path, 'rb') as cleansed_submission_data, \
            open(cleansed_comment_data_path, 'rb') as cleansed_comment_data:
        submission_data = pickle.load(cleansed_submission_data)
        comment_data = pickle.load(cleansed_comment_data)
        user_data = pd.concat([submission_data[['subreddit', 'author']], comment_data[['subreddit', 'author']]])
        users_by_subreddit = user_data.groupby('subreddit') \
            .agg('nunique') \
            .rename(columns={'author': 'number_of_authors'})
        print(f"Mean number of subreddit users is: {users_by_subreddit['number_of_authors'].mean()}.")


def users_with_most_submissions():
    with open(cleansed_submission_data_path, 'rb') as cleansed_submission_data:
        submission_data = pickle.load(cleansed_submission_data)
        submissions_by_user = submission_data[['id', 'author']] \
            .groupby('author') \
            .agg(np.size) \
            .rename(columns={'id': 'number_of_submissions'})
        print(submissions_by_user.sort_values('number_of_submissions', ascending=False).head(5))


def users_with_most_comments():
    with open(cleansed_comment_data_path, 'rb') as cleansed_comment_data:
        comment_data = pickle.load(cleansed_comment_data)
        comments_by_user = comment_data[['id', 'author']] \
            .groupby('author') \
            .agg(np.size) \
            .rename(columns={'id': 'number_of_comments'})
        print(comments_by_user.sort_values('number_of_comments', ascending=False).head(5))


def users_active_on_most_subreddits():
    with open(cleansed_submission_data_path, 'rb') as cleansed_submission_data, \
            open(cleansed_comment_data_path, 'rb') as cleansed_comment_data:
        submission_data = pickle.load(cleansed_submission_data)
        comment_data = pickle.load(cleansed_comment_data)
        user_data = pd.concat([submission_data[['subreddit', 'author']], comment_data[['subreddit', 'author']]])
        subreddits_by_user = user_data.groupby('author') \
            .agg('nunique') \
            .rename(columns={'subreddit': 'number_of_subreddits'})
        print(subreddits_by_user.sort_values('number_of_subreddits', ascending=False).head(5))


def pearson_correlation_coefficient():
    with open(cleansed_submission_data_path, 'rb') as cleansed_submission_data, \
            open(cleansed_comment_data_path, 'rb') as cleansed_comment_data:
        submission_data = pickle.load(cleansed_submission_data)
        comment_data = pickle.load(cleansed_comment_data)
        submission_data_filtered = submission_data[['id', 'author']].rename(columns={'id': 'submission_id'})
        comment_data_filtered = comment_data[['id', 'author']].rename(columns={'id': 'comment_id'})
        submissions_by_user = submission_data_filtered.groupby('author') \
            .agg(np.size) \
            .rename(columns={'submission_id': 'number_of_submissions'})
        comments_by_user = comment_data_filtered.groupby('author') \
            .agg(np.size) \
            .rename(columns={'comment_id': 'number_of_comments'})
        df = pd.concat([submissions_by_user, comments_by_user], axis=1)
        df['number_of_submissions'] = df['number_of_submissions'].fillna(0)
        df['number_of_comments'] = df['number_of_comments'].fillna(0)
        print(df.corr())


def submissions_with_most_comments():
    with open(cleansed_submission_data_path, 'rb') as cleansed_submission_data:
        submission_data = pickle.load(cleansed_submission_data)
        submission_data_filtered = submission_data[submission_data['over_18'] == False]
        submission_data_filtered = submission_data_filtered[['id', 'subreddit', 'num_comments', 'domain']]
        print(submission_data_filtered.sort_values('num_comments', ascending=False).head(5))


def active_users(submissions, comments, subreddit):
    submissions = submissions[submissions['subreddit'] == subreddit]
    comments = comments[comments['subreddit'] == subreddit]
    return set(submissions['author'].unique()).union(set(comments['author'].unique()))


def model_snet_graph():
    with open(cleansed_submission_data_path, 'rb') as cleansed_submission_data, \
            open(cleansed_comment_data_path, 'rb') as cleansed_comment_data:
        submission_data = pickle.load(cleansed_submission_data)
        comment_data = pickle.load(cleansed_comment_data)
        subreddits = set(submission_data['subreddit'].unique())
        subreddits = subreddits.union(set(comment_data['subreddit'].unique()))
        active_users_by_subreddit = {}
        for subreddit in subreddits:
            active_users_by_subreddit[subreddit] = active_users(submission_data, comment_data, subreddit)
        print(active_users_by_subreddit)
        snet_graph = nx.Graph()
        snet_graph.add_nodes_from(subreddits)
        subreddits_list = list(subreddits)
        pairs_of_subreddits = [(subreddits_list[i], subreddits_list[j])
                               for i in range(len(subreddits_list))
                               for j in range(i + 1, len(subreddits_list))
                               ]
        for subreddit_i, subreddit_j in pairs_of_subreddits:
            weight = len(active_users_by_subreddit[subreddit_i]
                         .intersection(active_users_by_subreddit[subreddit_j]))
            if weight > 0:
                snet_graph.add_edge(subreddit_i, subreddit_j, weight=weight)
        nx.write_gml(snet_graph, f"{graphs_path}/snet.gml")


def clustering_analysis_erdos_renyi(graph):
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    p = (2 * float(m)) / (n * (n - 1))
    er_graph = nx.erdos_renyi_graph(n, p)
    print(f"Avg. Clustering Coefficient: {nx.average_clustering(er_graph)}")
    er_graph_clustering = nx.clustering(er_graph)
    axs = plt.subplot()
    axs.hist(er_graph_clustering.values(), bins=10)
    axs.set_xlabel('Clustering Coefficient')
    axs.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def assortative_analysis(graph):
    # Assortativity coefficient (by node degree)
    print(f"Assortativity coefficient (by node degree): {nx.degree_assortativity_coefficient(graph)}.")
    # Assortativity coefficient (by weighted node degree)
    print(f"Assortativity coefficient (by node degree): {nx.degree_assortativity_coefficient(graph, weight='weight')}.")


def draw_degree_histogram(graph, weighted=False, xscale='linear', yscale='linear'):
    if weighted:
        degrees = graph.degree(weight='weight')
    else:
        degrees = graph.degree()
    _, degree_list = zip(*degrees)
    degree_counts = collections.Counter(degree_list)
    x, y = zip(*degree_counts.items())
    plt.figure(1)
    if weighted:
        plt.xlabel('Weighted Degree')
    else:
        plt.xlabel('Degree')
    plt.xscale(xscale)
    plt.xlim(1, max(x))
    plt.ylabel('Frequency')
    plt.yscale(yscale)
    plt.ylim(1, max(y))
    plt.scatter(x, y, marker='.')
    plt.show()


def powerlaw_fit_analysis(graph):
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    results = powerlaw.Fit(degree_sequence)
    print(results.power_law.alpha)
    print(results.power_law.xmin)
    print(results.power_law.sigma)
    r, p = results.distribution_compare('power_law', 'exponential')
    print(f"Loglikelihood ratio: {r}")
    print(f"Statistical significance: {p}")


def hits_analysis(graph):
    hubs, authorities = nx.hits(graph)
    print(sorted(hubs.items(), key=lambda x: x[1], reverse=True))
    print(sorted(authorities.items(), key=lambda x: x[1], reverse=True))


def extract_dominant_cc(graph):
    dominant_cc = max(nx.connected_components(graph), key=len)
    graph_dominant_cc = graph.subgraph(dominant_cc).copy()
    nx.write_gml(graph_dominant_cc, f"{graphs_path}/snetf-dom.gml")


def centrality_analysis(graph):
    dc_dict = nx.degree_centrality(graph)
    cc_dict = nx.closeness_centrality(graph)
    bc_dict = nx.betweenness_centrality(graph)
    evc_dict = nx.eigenvector_centrality(graph, weight='weight')
    lambda_max = max(nx.adjacency_spectrum(graph))
    katz_centrality_dict = nx.katz_centrality(graph, alpha=(1 / (2 * lambda_max)), weight='weight')
    # Form a DataFrame
    df1 = pd.DataFrame.from_dict(dc_dict, orient='index', columns=['DC'])
    df2 = pd.DataFrame.from_dict(cc_dict, orient='index', columns=['CC'])
    df3 = pd.DataFrame.from_dict(bc_dict, orient='index', columns=['BC'])
    df4 = pd.DataFrame.from_dict(evc_dict, orient='index', columns=['EVC'])
    df5 = pd.DataFrame.from_dict(katz_centrality_dict, orient='index', columns=['KC'])
    df = pd.concat([df1, df2, df3, df4, df5], axis=1)
    df['composite_rank'] = df['DC'] + df['CC'] + df['BC'] + df['EVC'] + df['KC']
    # Degree centrality
    df_dc = pd.DataFrame({'DC': df['DC']})
    df_dc.sort_values(by='DC', ascending=False, inplace=True)
    print(df_dc.head(5))
    # Closeness centrality
    df_cc = pd.DataFrame({'CC': df['CC']})
    df_cc.sort_values(by='CC', ascending=False, inplace=True)
    print(df_cc.head(5))
    # Betweenness centrality
    df_bc = pd.DataFrame({'BC': df['BC']})
    df_bc.sort_values(by='BC', ascending=False, inplace=True)
    print(df_bc.head(5))
    # Eigenvector centrality
    df_evc = pd.DataFrame({'EVC': df['EVC']})
    df_evc.sort_values(by='EVC', ascending=False, inplace=True)
    print(df_evc.head(5))
    # Katz centrality
    df_kc = pd.DataFrame({'KC': df['KC']})
    df_kc.sort_values(by='KC', ascending=False, inplace=True)
    print(df_kc.head(5))
    # Composite rank
    df_cr = pd.DataFrame({'composite_rank': df['composite_rank']})
    df_cr.sort_values(by='composite_rank', ascending=False, inplace=True)
    print(df_cr.head(5))
    return df


def katz_centrality_analysis(graph):
    lambda_max = max(nx.adjacency_spectrum(graph))
    with open(cleansed_submission_data_path, 'rb') as cleansed_submission_data, \
            open(cleansed_comment_data_path, 'rb') as cleansed_comment_data:
        submission_data = pickle.load(cleansed_submission_data)
        comment_data = pickle.load(cleansed_comment_data)
        submission_data_mask = submission_data['subreddit'] == 'reddit.com'
        reddit_dot_com_submissions = submission_data[submission_data_mask]
        comment_data_mask = comment_data['subreddit'] == 'reddit.com'
        reddit_dot_com_comments = comment_data[comment_data_mask]
        reddit_dot_com_subreddits = set(reddit_dot_com_submissions['subreddit'].unique()) \
            .union(set(reddit_dot_com_comments['subreddit'].unique()))
        beta_dict = {}
        for subreddit in graph.nodes():
            if subreddit in reddit_dot_com_subreddits:
                beta_dict[subreddit] = 10.0
            else:
                beta_dict[subreddit] = 1.0
        katz_centrality_dict = nx.katz_centrality(graph, alpha=(1 / (2 * lambda_max)), beta=beta_dict, weight='weight')
        df_katz_centrality = pd.DataFrame.from_dict(katz_centrality_dict, orient='index', columns=['KC'])
        df_katz_centrality.sort_values(by='KC', ascending=False, inplace=True)
        print(df_katz_centrality.head(5))


def plot_and_save_dendrogram(G):
    plt.rcParams["figure.figsize"] = (24, 13)
    # get Girvan-Newman communities list
    communities = list(nx.community.girvan_newman(G))

    # building initial dict of node_id to each possible subset:
    node_id = 0
    init_node2community_dict = {node_id: communities[0][0].union(communities[0][1])}
    for comm in communities:
        for subset in list(comm):
            if subset not in init_node2community_dict.values():
                node_id += 1
                init_node2community_dict[node_id] = subset

    # turning this dictionary to the desired format in @mdml's answer
    node_id_to_children = {e: [] for e in init_node2community_dict.keys()}
    for node_id1, node_id2 in combinations(init_node2community_dict.keys(), 2):
        for node_id_parent, group in init_node2community_dict.items():
            if len(init_node2community_dict[node_id1].intersection(
                    init_node2community_dict[node_id2])) == 0 and group == init_node2community_dict[node_id1].union(
                init_node2community_dict[node_id2]):
                node_id_to_children[node_id_parent].append(node_id1)
                node_id_to_children[node_id_parent].append(node_id2)

    # also recording node_labels dict for the correct label for dendrogram leaves
    node_labels = dict()
    for node_id, group in init_node2community_dict.items():
        if len(group) == 1:
            node_labels[node_id] = list(group)[0]
        else:
            node_labels[node_id] = ''

    # also needing a subset to rank dict to later know within all k-length merges which came first
    subset_rank_dict = dict()
    rank = 0
    for e in communities[::-1]:
        for p in list(e):
            if tuple(p) not in subset_rank_dict:
                subset_rank_dict[tuple(sorted(p))] = rank
                rank += 1
    subset_rank_dict[tuple(sorted(chain.from_iterable(communities[-1])))] = rank

    # my function to get a merge height so that it is unique (probably not that efficient)
    def get_merge_height(sub):
        sub_tuple = tuple(sorted([node_labels[i] for i in sub]))
        n = len(sub_tuple)
        other_same_len_merges = {k: v for k, v in subset_rank_dict.items() if len(k) == n}
        min_rank, max_rank = min(other_same_len_merges.values()), max(other_same_len_merges.values())
        range = (max_rank - min_rank) if max_rank > min_rank else 1
        return float(len(sub)) + 0.8 * (subset_rank_dict[sub_tuple] - min_rank) / range

    # finally using @mdml's magic, slightly modified:
    G = nx.DiGraph(node_id_to_children)
    nodes = G.nodes()
    leaves = set(n for n in nodes if G.out_degree(n) == 0)
    inner_nodes = [n for n in nodes if G.out_degree(n) > 0]

    # Compute the size of each subtree
    subtree = dict((n, [n]) for n in leaves)
    for u in inner_nodes:
        children = set()
        node_list = list(node_id_to_children[u])
        while len(node_list) > 0:
            v = node_list.pop(0)
            children.add(v)
            node_list += node_id_to_children[v]
        subtree[u] = sorted(children & leaves)

    inner_nodes.sort(key=lambda n: len(subtree[n]))  # <-- order inner nodes ascending by subtree size, root is last

    # Construct the linkage matrix
    leaves = sorted(leaves)
    index = dict((tuple([n]), i) for i, n in enumerate(leaves))
    Z = []
    k = len(leaves)
    for i, n in enumerate(inner_nodes):
        children = node_id_to_children[n]
        x = children[0]
        for y in children[1:]:
            z = tuple(sorted(subtree[x] + subtree[y]))
            i, j = index[tuple(sorted(subtree[x]))], index[tuple(sorted(subtree[y]))]
            Z.append([i, j, get_merge_height(subtree[n]), len(z)])  # <-- float is required by the dendrogram function
            index[z] = k
            subtree[z] = list(z)
            x = z
            k += 1

    # dendrogram
    plt.figure()
    dendrogram(Z, labels=[node_labels[node_id] for node_id in leaves])
    plt.savefig(f"{dendrograms_path}/dendrogram.png")


def draw_edge_weight_histogram(graph):
    edge_weights = nx.get_edge_attributes(graph, name='weight').values()
    edge_weight_counts = collections.Counter(edge_weights)
    x, y = zip(*edge_weight_counts.items())
    plt.figure(1)
    plt.xlabel('Edge Weight')
    plt.xscale('linear')
    plt.xlim(1, max(x))
    plt.ylabel('Frequency')
    plt.yscale('linear')
    plt.ylim(1, max(y))
    plt.scatter(x, y, marker='.')
    plt.show()


def model_snetf_graph(snet_graph):
    w_threshold = 25
    snetf_graph = snet_graph.copy()
    snetf_graph.remove_edges_from(
        [(n1, n2)
         for n1, n2, w in snet_graph.edges(data='weight')
         if w < w_threshold])
    nx.write_gml(snetf_graph, f"{graphs_path}/snetf.gml")


def model_snett_graph(snet_graph):
    targeted_subreddits = [
        'reddit.com',
        'pics',
        'worldnews',
        'programming',
        'business',
        'politics',
        'obama',
        'science',
        'technology',
        'WTF',
        'AskReddit',
        'netsec',
        'philosophy',
        'videos',
        'offbeat',
        'funny',
        'entertainment',
        'linux',
        'geek',
        'gaming',
        'comics',
        'gadgets',
        'nsfw',
        'news',
        'environment',
        'atheism',
        'canada',
        'math',
        'Economics',
        'scifi',
        'bestof',
        'cogsci',
        'joel',
        'Health',
        'guns',
        'photography',
        'software',
        'history',
        'ideas'
    ]
    snett_graph = snet_graph.subgraph(nodes=targeted_subreddits)
    nx.write_gml(snett_graph, f"{graphs_path}/snett.gml")


def model_usernet_graph():
    with open(cleansed_submission_data_path, 'rb') as cleansed_submission_data, \
            open(cleansed_comment_data_path, 'rb') as cleansed_comment_data:
        submission_data = pickle.load(cleansed_submission_data)
        comment_data = pickle.load(cleansed_comment_data)
        authors = pd.concat([submission_data[['id', 'author']],
                             comment_data[['id', 'author']]])
        actual_comments_data = comment_data[['author', 'parent_id']]
        actual_comments_data['parent_id'] = \
            actual_comments_data['parent_id'].str[3:]
        usernet_data = actual_comments_data \
            .merge(authors, left_on='parent_id', right_on='id',
                   suffixes=('_child', '_parent'))
        usernet_data = usernet_data[['author_child', 'author_parent']] \
            .groupby(by=['author_child', 'author_parent']) \
            .agg(np.size)
        usernet_graph = nx.DiGraph()
        usernet_graph.add_nodes_from(set(authors['author'].unique()))
        for index, value in usernet_data.items():
            usernet_graph.add_edge(index[0], index[1], weight=value)
        nx.write_gml(usernet_graph, f"{graphs_path}/usernet.gml")


if __name__ == '__main__':
    usernet_graph = nx.read_gml(f"{graphs_path}/usernet.gml")
    print(nx.average_shortest_path_length(usernet_graph))
    print(nx.diameter(usernet_graph))
