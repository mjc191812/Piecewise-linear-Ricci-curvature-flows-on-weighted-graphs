
import numpy as np
import math
import time
from sklearn import preprocessing, metrics
from multiprocessing import Pool, cpu_count
import community.community_louvain as community_louvain
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
import importlib
import cvxpy as cvx
import io
import zipfile
from concurrent.futures import ProcessPoolExecutor
from networkx.algorithms import community
from networkx.algorithms.community import girvan_newman, label_propagation_communities
from infomap import Infomap
import igraph as ig
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import networkx as nx
import networkit as nk
import heapq
from copy import deepcopy
_nbr_topk = 3000
_apsp = {}
_max_ari={}
_max_nmi={}
_max_modularity={}






class RhoCurvature:
    
    def __init__(self, G, alpha=0.5, weight="weight", proc=cpu_count()):
        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self.proc = proc
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary
        self.EPSILON = 1e-7  # to prevent divided by zero
        self.base = math.e
    
        


   


    def _get_all_pairs_shortest_path(self):
        # Construct the all pair shortest path lookup
        self.lengths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight=self.weight))
        return self.lengths

        
    def _get_single_node_neighbors_distributions(self, node):
        neighbors = list(self.G.neighbors(node))
        heap_weight_node_pair = []
        for nbr in neighbors:
            w=self.G[node][nbr][self.weight]
            if len(heap_weight_node_pair) < _nbr_topk:
                heapq.heappush(heap_weight_node_pair, (w, nbr))
            else:
                heapq.heappushpop(heap_weight_node_pair, (w, nbr))
        nbr_edge_weight_sum = sum([x[0] for x in heap_weight_node_pair])
        if not neighbors:
            return [1],[node]
        if nbr_edge_weight_sum > self.EPSILON:
            distributions = [(1.0 - self.alpha) * w / nbr_edge_weight_sum for w, _ in heap_weight_node_pair]

        else:
            # Sum too small, just evenly distribute to every neighbors
           
            distributions = [(1.0 - self.alpha) / len(heap_weight_node_pair)] * len(heap_weight_node_pair)
        nbr = [x[1] for x in heap_weight_node_pair]
        return distributions + [self.alpha], nbr + [node]
    def _get_edge_density_distributions(self):
        densities = dict()

        
        for x in self.G.nodes():
            densities[x] = self._get_single_node_neighbors_distributions(x)

    def _optimal_transportation_distance(self, x, y, d):
        rho = cvx.Variable((len(y), len(x)))  
        
        # objective function d(x,y) * rho * x, need to do element-wise multiply here
        obj = cvx.Minimize(cvx.sum(cvx.multiply(np.multiply(d.T, x.T), rho)))

        # \sigma_i rho_{ij}=[1,1,...,1]
        source_sum = cvx.sum(rho, axis=0, keepdims=True)
        constrains = [rho @ x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
        prob = cvx.Problem(obj, constrains)

        m = prob.solve(solver="ECOS")  # change solver here if you want
        return m

    def _distribute_densities(self, source, target):
        # Append source and target node into weight distribution matrix x,y
        x,source_topknbr=self._get_single_node_neighbors_distributions(source)
        y,target_topknbr=self._get_single_node_neighbors_distributions(target)
        d = []
        for src in source_topknbr:
            tmp = []
            for tgt in target_topknbr:
                tmp.append(self.lengths[src][tgt])
            d.append(tmp)
        d = np.array(d)
        x=np.array(x)
        y=np.array(y)
        return x,y,d    
        

    def _compute_ricci_curvature_single_edge(self, source, target):

        assert source != target, "Self loop is not allowed."  # to prevent self loop

        # If the weight of edge is too small, return 0 instead.
        if self.lengths[source][target] < self.EPSILON:
            print("Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead." % (source, target))
            return {(source, target): 0}

        # compute transportation distance
        m = 1  # assign an initial cost

        x, y, d = self._distribute_densities(source, target)
        m = self._optimal_transportation_distance(x, y, d)

        # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
        result =1- m / self.lengths[source][target]  # Divided by the length of d(i, j)
        #print("Ricci curvature (%s,%s) = %f" % (source, target, result))

        return {(source, target): result}


   
    def compute_ricci_curvature_edges(self, edge_list=None):
        
        if not edge_list:
            edge_list = []

        # Construct the all pair shortest path dictionary
        #if not self.lengths:
        self.lengths = self._get_all_pairs_shortest_path()

        # Construct the density distribution
        if not self.densities:
            self.densities = self._get_edge_density_distributions()
        
       

            
        # Compute Ricci curvature for edges
        args = [(source, target) for source, target in edge_list]

        result = [self._compute_ricci_curvature_single_edge(*arg) for arg in args]
            


        return result


    def compute_ricci_curvature(self):
        
        if not nx.get_edge_attributes(self.G, self.weight):
            print('Edge weight not detected in graph, use "weight" as edge weight.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = 1.0

                
        edge_ricci = self.compute_ricci_curvature_edges(self.G.edges())
        
        # Assign edge Ricci curvature from result to graph G
        for rc in edge_ricci:
            for k in list(rc.keys()):
                source, target = k
                self.G[source][target]['ricciCurvature'] = rc[k]
    
    def LLY_get_edge_density_distributions(self):
        densities = dict()

        def Gamma(i, j):
            return self.lengths[i][j]

        # Construct the density distributions on each node
        def get_single_node_neighbors_distributions(neighbors):
            # Get sum of distributions from x's all neighbors
            nbr_edge_weight_sum = sum([Gamma(x,nbr) for nbr in neighbors])

            if nbr_edge_weight_sum > self.EPSILON:
                result = [Gamma(x,nbr) / nbr_edge_weight_sum for nbr in neighbors]
            elif len(neighbors) == 0:
                return []
            else:
                result = [1.0 / len(neighbors)] * len(neighbors)
            result.append(0)
            return result

        for x in self.G.nodes():
            densities[x] = get_single_node_neighbors_distributions(list(self.G.neighbors(x)))

        return densities


    def Lin_Lu_Yau_optimal_transportation_distance(self, x, y, d):
        star_coupling = cvx.Variable((len(y), len(x)))  # the transportation plan B
        # objective function sum(star_coupling(x,y) * d(x,y)) , need to do element-wise multiply here
        obj = cvx.Maximize(cvx.sum(cvx.multiply(star_coupling, d.T)))
        # constrains
        constrains = [cvx.sum(star_coupling)==0]

        constrains += [cvx.sum(star_coupling[:, :-1], axis=0, keepdims=True) == np.multiply(-1, x.T[:,:-1])]
        constrains += [cvx.sum(star_coupling[:-1, :], axis=1, keepdims=True) == np.multiply(-1, y[:-1])]

        constrains += [0 <= star_coupling[-1, -1], star_coupling[-1, -1] <= 2]
        constrains += [star_coupling[:-1,:-1] <= 0]
        constrains += [star_coupling[-1,:-1] <= 0]
        constrains += [star_coupling[:-1,-1] <= 0]

        prob = cvx.Problem(obj, constrains)

        m = prob.solve(solver="ECOS")  # change solver here if you want
        # solve for optimal transportation cost
        return m

    def LLY_distribute_densities(self, source, target):

        # Append source and target node into weight distribution matrix x,y
        source_nbr = list(self.G.neighbors(source))
        target_nbr = list(self.G.neighbors(target))

        # Distribute densities for source and source's neighbors as x
        if not source_nbr:
            source_nbr.append(source)
            x = [1]
        else:
            source_nbr.append(source)
            x = self.densities[source]

        # Distribute densities for target and target's neighbors as y
        if not target_nbr:
            target_nbr.append(target)
            y = [1]
        else:
            target_nbr.append(target)
            y = self.densities[target]

        # construct the cost dictionary from x to y
        d = np.zeros((len(x), len(y)))

        for i, src in enumerate(source_nbr):
            for j, dst in enumerate(target_nbr):
                assert dst in self.lengths[src], "Target node not in list, should not happened, pair (%d, %d)" % (src, dst)
                d[i][j] = self.lengths[src][dst]

        x = np.array([x]).T  # the mass that source neighborhood initially owned
        y = np.array([y]).T  # the mass that target neighborhood needs to received

        return x, y, d


    def calculate_Lin_Lu_yau_ricci_curvature(self, source, target):
        assert source != target, "Self loop is not allowed."  # to prevent self loop

        # If the weight of edge is too small, return 0 instead.
        if self.lengths[source][target] < self.EPSILON:
            print("Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead." % (source, target))
            return {(source, target): 0}

        # compute transportation distance
        m = 1  # assign an initial cost

        x, y, d = self.LLY_distribute_densities(source, target)
        m = self.Lin_Lu_Yau_optimal_transportation_distance(x, y, d)

      
        result =m / self.lengths[source][target]  # Divided by the length of d(i, j)
        
        return result

    def compute_Lin_Lu_yau_ricci_curvature(self):
        curvatures = {}
        if not self.densities:
            self.densities = self.LLY_get_edge_density_distributions()

        for n1, n2 in self.G.edges():
            curvatures[(n1, n2)] = self.calculate_Lin_Lu_yau_ricci_curvature(n1, n2)
            self.G[n1][n2]['LLY_curvature'] = curvatures[(n1, n2)]

    def forman_ricci_curvature(self, G, edge):
    
        u, v = edge       
        w_uv = self.G[u][v].get('weight', 1.0)
    
        
        deg_u = sum(self.G[u][neighbor].get('weight', 1.0) for neighbor in self.G.neighbors(u))
        deg_v = sum(self.G[v][neighbor].get('weight', 1.0) for neighbor in self.G.neighbors(v))
    
        S1=0
        for neighbor in self.G.neighbors(u):
            w_uw=self.G[u][neighbor].get('weight')
            S1+=math.sqrt(w_uv/w_uw)

        S2=0
        for neighbor in self.G.neighbors(v):
            w_vw=self.G[v][neighbor].get('weight')
            S2+=math.sqrt(w_uv/w_vw)
    
        curvature = deg_u*(2-S1)+deg_v*(2-S2)
        return curvature
    
    def compute_forman_ricci_curvatures(self):
        edge_ricci = [self.forman_ricci_curvature(self.G, (u, v)) for (u, v) in self.G.edges()]
        
        # Assign edge Ricci curvature from result to graph G
        for (u, v), curvature in zip(self.G.edges(), edge_ricci):
            self.G[u][v]['forman_ricci_curvature'] = curvature
        

    def Menger_ricci_curvature(self, G, edge):
        u, v = edge

        
        #lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=self.weight))

        M_T = 0

        connected_nodes = [x for x in G.nodes() if x in self.lengths[u] and x in self.lengths[v]]

        for x in connected_nodes:
            a = self.lengths[u][x]
            b = self.lengths[u][v]
            c = self.lengths[v][x]
            if a == 0 or b == 0 or c == 0:
                continue
            p = (a + b + c) / 2
            if (p - a) * (p - b) * (p - c) < 0:
                continue
            s = math.sqrt(p * (p - a) * (p - b) * (p - c)) / (a * b * c)
            M_T += s
        
        return M_T

    def compute_Menger_ricci_curvatures(self):
        curvatures = {}
        for n1, n2 in self.G.edges():
           
            curvatures[(n1, n2)] = self.Menger_ricci_curvature(self.G, (n1, n2))
            self.G[n1][n2]['Menger_ricci_curvature'] = curvatures[(n1, n2)]
        return curvatures
        


    def dijkstra(self, start, end):
       
        distances = {node: float('inf') for node in self.G.nodes()}
        distances[start] = 0
        pq = [(0, start)]
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            if current_node == end: 
                return current_dist
            if current_dist > distances[current_node]:
                continue
            
            for neighbor in self.G.neighbors(current_node):
                weight = self.G[current_node][neighbor].get(self.weight, 1.0)
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
        
        return float('inf')  

    def find_all_paths(self, start, end, path=None, current_weight=0, steps=0, max_steps=4):
        if path is None:
            path = []
        path = path + [start]
        
       
        if  steps > max_steps:
            return []
        
        if start == end:
            return [(path.copy(), current_weight)]
        
        if start not in self.G.nodes:
            return []
        
        all_paths = []
        for neighbor in self.G.neighbors(start):
            if neighbor not in path:
                edge_weight = self.G[start][neighbor].get('weight', 1.0)
                
                new_paths = self.find_all_paths(
                    neighbor, end, path, current_weight + edge_weight, 
                    steps + 1, max_steps
                )
                all_paths.extend(new_paths)
        return all_paths


    def calculate_haantjes(self, start, end):
        all_paths = self.find_all_paths(start, end)
        shortest_dist = self.dijkstra(start, end)
        
        if shortest_dist == 0 or shortest_dist == float('inf'):
            return 0.0
        
        H = 0.0
        for path, total_weight in all_paths:
            delta = total_weight - shortest_dist
            if delta > 0: 
                H += math.sqrt(delta / shortest_dist) / shortest_dist
        
        return H

    def compute_haantjes_curvature(self):
        curvatures = {}
        for n1, n2 in self.G.edges():
            curvatures[(n1, n2)] = self.calculate_haantjes(n1, n2)
            self.G[n1][n2]['haantjes_curvature'] = curvatures[(n1, n2)]

        
  

 

def show_results(G, curvature="ricciCurvature"):


    # Plot the histogram of Ricci curvatures
    plt.subplot(2, 1, 1)
    ricci_curvtures = nx.get_edge_attributes(G, curvature).values()
    plt.hist(ricci_curvtures,bins=20)
    plt.xlabel('Ricci curvature')
    plt.title("Histogram of Ricci Curvatures ")

    # Plot the histogram of edge weights
    plt.subplot(2, 1, 2)
    weights = nx.get_edge_attributes(G, "weight").values()
    plt.hist(weights,bins=20)
    plt.xlabel('Edge weight')
    plt.title("Histogram of Edge weights ")

    plt.tight_layout()
    plt.show()


    return G



    

def ARI(G, clustering, clustering_label="club"):

    complex_list = nx.get_node_attributes(G, clustering_label)
    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(list(complex_list.values()))

    if isinstance(clustering, dict):
        # python-louvain partition format
        y_pred = np.array([clustering[v] for v in complex_list.keys()])
    elif isinstance(clustering[0], set):
        # networkx partition format
        predict_dict = {c: idx for idx, comp in enumerate(clustering) for c in comp}
        y_pred = np.array([predict_dict[v] for v in complex_list.keys()])
    elif isinstance(clustering, list):
        # sklearn partition format
        y_pred = clustering
    else:
        return -1

    return metrics.adjusted_rand_score(y_true, y_pred)


def NMI(G, clustering, clustering_label="club"):
    
    
    complex_list = nx.get_node_attributes(G, clustering_label)
    
    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(list(complex_list.values()))
    
    if isinstance(clustering, dict):
        # python-louvain partition format
        y_pred = np.array([clustering[v] for v in complex_list.keys()])
    elif isinstance(clustering[0], set):
        # networkx partition format
        predict_dict = {c: idx for idx, comp in enumerate(clustering) for c in comp}
        y_pred = np.array([predict_dict[v] for v in complex_list.keys()])
    elif isinstance(clustering, list):
        # sklearn partition format
        y_pred = clustering
    else:
        return -1
    
    return metrics.normalized_mutual_info_score(y_true, y_pred)

#my_surgery(_rc.G, weight="weight", cut=1.0)

def check_accuracy(G_origin, weight="weight", clustering_label="club"):
    """To check the clustering quality while cut the edges with weight using different threshold

    Parameters
    ----------
    G_origin : NetworkX graph
        A graph with ``weight`` as Ricci flow metric to cut.
    weight: float
        The edge weight used as Ricci flow metric. (Default value = "weight")
    clustering_label : str
        Node attribute name for ground truth.

    """
    G = G_origin.copy()
    modularity, ari ,nmi = [], [], []
    c_communities=list(nx.connected_components(G))
    deg_sum = sum(dict(G.degree(weight=weight)).values())
    if deg_sum == 0:
        modularity.append(0)
    else:
        modularity.append(nx.community.modularity(G, c_communities))
    edge_weights = nx.get_edge_attributes(G, weight).values()
    if edge_weights:
        maxw = max(edge_weights)
        minw = min(edge_weights)
    else:
        return 0,0,0

    cutoff_range = np.arange(maxw, minw , -0.01)
    for cutoff in cutoff_range:
        
        edge_trim_list = []
        for n1, n2 in G.edges():
            if G[n1][n2][weight] > cutoff:
                edge_trim_list.append((n1, n2))
        G.remove_edges_from(edge_trim_list)
        if G.number_of_edges() == 0:
            cutoff_range=np.arange(maxw, minw , -0.01)
            print("No edges left in the graph. Exiting the loop.")
            break
        # Get connected component after cut as clustering
        clustering = {c: idx for idx, comp in enumerate(nx.connected_components(G)) for c in comp}
       

        # Compute modularity and ari 
        c_communities=list(nx.connected_components(G))
        deg_sum = sum(dict(G.degree(weight=weight)).values())
        if deg_sum == 0:
            modularity.append(0)
        else:
            modularity.append(nx.community.modularity(G, c_communities))
        
        ari.append(ARI(G, clustering, clustering_label=clustering_label))
        nmi.append(NMI(G, clustering, clustering_label=clustering_label))
    # if maxw != 0:
    #     plt.xlim(maxw, 0)
    # else:
    #     plt.xlim(-1, 1) 
    # plt.xlim(maxw, 0)
    # plt.xlabel("Edge weight cutoff")
    # plt.plot(cutoff_range, modularity, alpha=0.8)
    # plt.plot(cutoff_range, ari, alpha=0.8)
    # plt.plot(cutoff_range, nmi, alpha=0.8)

    
    # plt.legend(['Modularity', 'Adjust Rand Index',"NMI"])
    
    print("max ari:", max(ari))
    print("max nmi:", max(nmi))
    print("max modularity:", max(modularity))
    
    #plt.show()
    return max(ari), max(nmi), max(modularity)

def draw_graph(G, clustering_label="club"):
    """
    A helper function to draw a nx graph with community.
    """
    complex_list = nx.get_node_attributes(G, clustering_label)
    le = preprocessing.LabelEncoder()
    node_color = le.fit_transform(list(complex_list.values()))
    pos=nx.spring_layout(G,seed=42)
    nx.draw(
            G,
            pos,
            node_color=node_color,
            cmap=plt.cm.rainbow,  
            node_size=300,
            with_labels=True,
            font_size=8
        )
    plt.show()





def main(): 


        n = 100
        tau1 = 3
        tau2 = 1.5
        mu = 0.3
        G = nx.LFR_benchmark_graph(
            n, tau1, tau2, mu, average_degree=20,  max_degree=50,min_community=20,  max_community=50
        )
        G.remove_edges_from(nx.selfloop_edges(G))
        Y = {node: list(G.nodes[node]['community'])[0] for node in G.nodes()}
        Y= [Y[i] for i in range(len(Y))]

        
        node_index_map = {node: i for i, node in enumerate(G.nodes)}     
        for (v1, v2) in G.edges():
                G[v1][v2]['weight'] = 1.0
        maxw=max(nx.get_edge_attributes(G, "weight").values())
        minw=min(nx.get_edge_attributes(G, "weight").values())
        A=2*maxw/minw
        c_i=False
        ricci_G = RhoCurvature(G)
        ricci_G._get_all_pairs_shortest_path()
        ricci_G.compute_ricci_curvature()
        def surgery(G_origin, weight='weight'):
            G = G_origin.copy()    
            c_i=False
            to_cut=[]
            for n1,n2 in G.edges():
                if G[n1][n2][weight]/minw>A:
                    to_cut.append((n1,n2))
                    c_i=True
            G.remove_edges_from(to_cut)
            # print("cutted edges:",to_cut)
            # print("cutted edges number:",len(to_cut))
            return G,c_i

    
    
        def safe_exponential(x):
            try:
                result = math.exp(x)
                return result
            except OverflowError:
                print(f"输入值 {x} 超出了可以计算的范围。")
                return None
        minw=min(nx.get_edge_attributes(ricci_G.G, "weight").values())
        last_t = 0
        max_iter=4
        time_interval=50

        for iter in range(1,max_iter):
            t = iter / time_interval
            for n1, n2 in ricci_G.G.edges():
            
                result=safe_exponential(-(t-last_t) * ricci_G.G[n1][n2]['ricciCurvature'])
                
                if result is not None:
                    ricci_G.G[n1][n2]['weight'] = ricci_G.G[n1][n2]['weight'] * (result)
                else:
                    break
            
            minw=min(nx.get_edge_attributes(ricci_G.G, "weight").values())
            
            if (iter+1) % 1== 0:
                ricci_G.G ,c_i=surgery(ricci_G.G, weight='weight')
            #nx.write_gexf(old_G.G, save_path + f"\_{iter+1}.gexf")
                
            last_t = t
            if c_i==True:
                ricci_G.compute_ricci_curvature()
                c_i=False

        a,b,c=check_accuracy(ricci_G.G, weight="weight", clustering_label=Y)





        # Louvain
        partition = community_louvain.best_partition(G)
        louvain_modularity = community_louvain.modularity(partition, G)
        louvain_labels = [partition[i] for i in G.nodes]  
        true_labels = Y
        nmi_louvain = normalized_mutual_info_score(true_labels, louvain_labels)
        

        # Girvan-Newman
        communities_gn = next(girvan_newman(G))
        community_list_gn = [list(c) for c in communities_gn]
        gn_modularity = nx.algorithms.community.quality.modularity(G, community_list_gn)
        node_index_map = {node: i for i, node in enumerate(G.nodes)}

        gn_labels = [0] * n
        for i, comm in enumerate(community_list_gn):
            for node in comm:
                gn_labels[node_index_map[node]] = i
        gn_nmi = normalized_mutual_info_score(Y, gn_labels)

        


        # LPA
        community_list_lpa = list(label_propagation_communities(G))
        lpa_modularity = nx.algorithms.community.quality.modularity(G, community_list_lpa)
        
        lpa_labels = [0]*n
        for i, comm in enumerate(community_list_lpa):
            for node in comm:
                lpa_labels[node_index_map[node]] = i
        lpa_nmi = normalized_mutual_info_score(Y, lpa_labels)
        

        
        
        # Infomap
        im = Infomap("--two-level")  
        for edge in G.edges(data=True):
            im.addLink(node_index_map[(edge[0])],node_index_map[(edge[1])])  
        im.run()  
        community_result = {node.physicalId: node.module_id for node in im.iterTree() if node.isLeaf}
        Y_infomap = [community_result[node_index_map[node]] for node in G.nodes]
        infomap_nmi = normalized_mutual_info_score(Y, Y_infomap)
        community_dict_infomap = {node: community_result[node_index_map[node]] for node in G.nodes}
        community_set_list_infomap = [{node for node, community in community_dict_infomap.items() if community == i} for i in set(community_result.values())]
        mod_infomap = nx.algorithms.community.quality.modularity(G, community_set_list_infomap)

        

        #  Walktrap 
        g = ig.Graph.from_networkx(G)
        walktrap_partition = g.community_walktrap()
        community_result = walktrap_partition.as_clustering().membership
        community_dict_walktrap = {node: community_result[node_index_map[node]] for node in G.nodes}
        community_set_list_walktrap = [{node for node, community in community_dict_walktrap.items() if community == i} for i in set(community_result)]
        walktrap_nmi = normalized_mutual_info_score(Y, community_result)
        mod_walktrap = nx.algorithms.community.quality.modularity(G, community_set_list_walktrap)


        print(f"Louvain Modularity: {louvain_modularity}")
        print(f"Louvain NMI: {nmi_louvain}")
        print(f"Girvan-Newman Modularity: {gn_modularity}")
        print(f"Girvan-Newman NMI: {gn_nmi}")
        print(f"LPA Modularity: {lpa_modularity}")
        print(f"LPA NMI: {lpa_nmi}")
        print(f"Infomap Modularity:", mod_infomap)
        print(f"Infomap NMI: {infomap_nmi}")
        print(f"Walktrap Modularity:", mod_walktrap)
        print(f"Walktrap NMI: {walktrap_nmi}")
        print(f"our_algorithm_Modularity:",a)
        print(f"our_algorithm_NMI:",c)

if __name__ == "__main__":
    main()

