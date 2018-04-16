"""
% Dynamic Community Detection Algorithm
% Version 1.0 (09/04/18)
%
% Original author:
% Tyrone Naidoo (naidootea@gmail.com)
%
% License:
% CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
%
"""

import networkx as nx
import community
import operator
import itertools
import numpy as np
from copy import deepcopy

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def _compute_e(node, community, Graph, partition):
    """
    This function calculates the "e" value between a specified node and
    community. The "e" value of a node is the number of edges from
    that node to a specific community.

    Args:
        node (any):       A node in a graph. May be an integer or string
                          or object, etc.
        community (int):  The community to check if edges are connected
                          to from the node.
        Graph (graph):    A graph consisting of nodes and edges.
        partition (dict): Keys represent nodes, and values represent the
                          community that each node belongs to.

    Returns:
        e (int): The number of edges between a node and a community.
    """

    neighbors = Graph.neighbors(node)

    e = 0

    # Check if neighbors are in the specified community
    for node_obj in neighbors:
        community_neighbor = partition.get(node_obj)
        if community_neighbor == community:
            e += 1

    return e


def _compute_degree_of_community(community, Graph, partition):
    """
    This function calculates the degree of a community. The degree of a
    community is the sum of the degrees of each node belonging to that
    community.

    Args:
        community (int):  The community whose degree is to be calculated
        Graph (graph):    A graph consisting of nodes and edges.
        partition (dict): Keys represent nodes, and values represent the
                          community that each node belongs to.

    Returns:
        degree_community (int): The degree of the community
    """

    # Get all the nodes that are in the community
    keys = []

    for node, community_node in partition.items():
        if community_node == community:
            keys.append(node)

    degree_community = 0

    # Get each nodes degree and sum
    for node in keys:
        degree_community += Graph.degree(node)

    return degree_community


def _compute_fin(node, Graph, partition):
    """
    This function computes the force F_in for a node, which signifies
    how strongly a node is attracted to the current community that it is
    in. The force F_in is used to determine if a node should stay in
    it's current community.

    Args:
        node (any):       A node in a graph. May be an integer or string
                          or object, etc.
        Graph (graph):    A graph consisting of nodes and edges.
        partition (dict): Keys represent nodes, and values represent the
                          community that each node belongs to.

    Returns:
        fin (float): The force that a node's community exerts on that
                     node.
    """

    community_node = partition.get(node)

    e = _compute_e(node, community_node, Graph, partition)
    degree_node = Graph.degree(node)
    degree_community = _compute_degree_of_community(community_node,
                                                   Graph, partition)
    num_edges = Graph.number_of_edges()

    fin = e - ( (degree_node * (degree_community - degree_node)) / (2 * num_edges) )

    return fin


def _compute_fout(node, Graph, partition, community):
    """
    This function computes the force F_out for a node, which signifies
    how strongly a node is attracted to another community. The force
    F_out is used to determine if a node should move to a different
    community.

    Args:
        node (any):       A node in a graph. May be an integer or string
                          or object, etc.
        Graph (graph):    A graph consisting of nodes and edges.
        partition (dict): Keys represent nodes, and values represent the
                          community that each node belongs to.
        community (int):  The external community to check if node "node"
                          has edges connecting to it.

    Returns:
        fout (float): The force that an external community exerts on a
                      node.
    """

    e = _compute_e(node, community, Graph, partition)
    degree_node = Graph.degree(node)
    degree_community = _compute_degree_of_community(community, Graph,
                                                   partition)
    num_edges = Graph.number_of_edges()

    fout = e - ((degree_node * degree_community) / (2 * num_edges))

    return fout


def _find_neighbour_communities(community, Graph, partition):
    """
    This function finds the neighbour communities of a specified
    community. It first finds all the nodes inside the specified
    community then finds the neighbours of each node and determines
    the communities that they belong to.

    Args:
        community (int):  The neighbour communities of this "community"
                          will be found.
        Graph (graph):    A graph consisting of nodes and edges.
        partition (dict): Keys represent nodes, and values represent the
                          community that each node belongs to.


    Returns:
        neighbor_communities (list): A list of communities that are
                                     neighbors to the community provided
                                     as a parameter.
    """

    # Find all nodes in "community"
    community_nodes = []
    all_neighbors = []
    for node, community_node in partition.items():
        if community_node == community:
            neighbors = Graph.neighbors(node)
            all_neighbors.append(neighbors)
            community_nodes.append(node)

    # all_neighbors is a list of lists, convert to one list
    # and then remove duplicates
    all_neighbors_list = list(itertools.chain(*all_neighbors))
    unique_neighbors = set(all_neighbors_list)

    # Remove all the neighbors that are within the same community
    # that was provided as a parameter.
    # Hence, we have only those nodes external to the community
    # ie. nodes belonging to neighbor communities
    external_nodes = [x for x in unique_neighbors if
                      x not in community_nodes]

    # Now collect each of the external nodes communities
    all_communities = []
    for node in external_nodes:
        neighbor_community = partition[node]
        all_communities.append(neighbor_community)

    # Get the unique list of neighbor communities
    # ie. remove duplicates
    neighbor_communities = set(all_communities)

    return neighbor_communities


def _compute_q_uv(u, v, Graph, partition):
    """
    This function computes delta q for nodes u and v. When a new edge is
    introduced, delta q is used to determine whether node u should move
    to node v's community (C(v)), or if node v should move to node u's
    community (C(u)), or neither.

    Args:
        u (any):          A node in a graph. May be an integer or string
                          or object, etc.
        v (any):          A node in a graph. May be an integer or string
                          or object, etc.
        Graph (graph):    A graph consisting of nodes and edges.
        partition (dict): Keys represent nodes, and values represent the
                          community that each node belongs to.

    Returns:
        delta_qu (int): The delta q value pertaining to node u.
        delta_qv (int): The delta q value pertaining to node v.
    """

    num_edges = Graph.number_of_edges()
    degree_u = Graph.degree(u)
    degree_v = Graph.degree(v)
    community_u = partition.get(u)
    degree_community_u = _compute_degree_of_community(community_u,
                                                      Graph,
                                                      partition)
    community_v = partition.get(v)

    degree_community_v = _compute_degree_of_community(community_v,
                                                      Graph,
                                                      partition)

    # Compute the e values (Edges from node to community)
    e_u_Cu = _compute_e(u, community_u, Graph, partition)
    e_u_Cv = _compute_e(u, community_v, Graph, partition)
    e_v_Cu = _compute_e(v, community_u, Graph, partition)
    e_v_Cv = _compute_e(v, community_v, Graph, partition)

    A = 4 * (num_edges + 1) * (e_u_Cv + 1 - e_u_Cu)
    B = e_u_Cu
    C = (2 * degree_community_v) - (2 * degree_community_u) - e_u_Cu
    D = 2 * (degree_u + 1)
    E = degree_u + 1 + degree_community_v - degree_community_u

    delta_qu = A + (B * C) - (D * E)

    A = 4 * (num_edges + 1) * (e_v_Cu + 1 - e_v_Cv)
    B = e_v_Cv
    C = (2 * degree_community_u) - (2 * degree_community_v) - e_v_Cv
    D = 2 * (degree_v + 1)
    E = degree_v + 1 + degree_community_u - degree_community_v

    delta_qv = A + (B * C) - (D * E)

    return delta_qu, delta_qv


def _compute_modularity(Graph, partition):
    """
    This function computes the modularity of the given community
    structure. A higher modularity indicates a better community
    structure. The modularity is used to determine whether a node should
    move from one community to another.

    Args:
        Graph (graph):    A graph consisting of nodes and edges.
        partition (dict): Keys represent nodes, and values represent the
                          community that each node belongs to.

    Returns:
        modularity (float): The modularity of the given community
                            structure.
    """

    num_edges = Graph.number_of_edges()
    all_communities = set(partition.values())
    all_c_degrees = []

    # Get the degree of each community
    for community in all_communities:
        degree = _compute_degree_of_community(community,
                                              Graph,
                                              partition)
        all_c_degrees.append(degree)

    # Get the number of edges inside each community
    all_num_edges = []
    for community in all_communities:
        all_nodes_in_c = []
        for node, comm in partition.items():
            if comm == community:
                all_nodes_in_c.append(node)

        # At this stage we have all nodes belonging to "community"
        # Now create subgraph from these nodes (Easier to count number
        # of edges). Also, excludes the edges that connects to external
        # nodes (ie. outside of the community)
        G_temp = Graph.subgraph(all_nodes_in_c)
        all_num_edges.append(G_temp.number_of_edges())

    # Convert to numpy arrays - makes the modularity computation easier
    degrees_array = np.array(all_c_degrees)
    degrees_array_sqrd = np.square(degrees_array)

    com_num_edges_array = np.array(all_num_edges)
    overall_num_edges_sqrd = np.square(num_edges)

    A = np.divide(com_num_edges_array, num_edges)
    B = np.divide( degrees_array_sqrd, (4 * overall_num_edges_sqrd) )

    modularity = sum(A - B)

    return modularity


def _find_best_community_for_neighbors(neighbors, Graph, partition,
                                       modularity_initial):
    """
    This function finds the best community for the neighbors of node
    w (where w is a node that moved to a new community).

    The neighbors of node w are found and each neighbor is processed.
    The neighbor communities of each node is found and the modularity
    of the community structure is computed when the neighbor node is
    placed in each of its neighbor communities.

    If the modularity increases when moving the neighbor to a neighbor
    community, then the neighbor is placed in that community.

    Args:
        neighbors (list): A list of neighbor nodes
        Graph (graph):    A graph consisting of nodes and edges.
        partition (dict): Keys represent nodes, and values represent the
                          community that each node belongs to.
        modularity_initial (float): The initial modularity prior to
                                    moving a neighbor node to other
                                    communities.

    Returns:
        changes (dict): A dictionary where keys are nodes, and the
                        values are the new communities that each node is
                        to be assigned to.
    """

    changes = {}
    for node in neighbors:
        neighbors_of_node = Graph.neighbors(node)

        # Get communities of neighbors of "node" (different communities)
        communities = []
        for node_x in neighbors_of_node:
            # If they are in different communities
            if partition.get(node_x) != partition.get(node):
                communities.append(partition.get(node_x))

        # Place node in each community and compute modularity
        modularities = {}

        temp_partition = deepcopy(partition)
        for c in set(communities):
            temp_partition[node] = c

            # Compute modularity when node is placed in each neighbour
            # community
            modularities[c] = _compute_modularity(Graph, temp_partition)

        # If modularity increases record the change that needs to me
        # made ie. move node to other community.
        if modularities:
            if max(modularities.values()) > modularity_initial:
                # store all changes that need to be made
                for community, mod in modularities.items():
                    if mod == max(modularities.values()):
                        changes[node] = community

    return changes


def _find_missing_elements(element_list):
    """
    This function finds elements that are missing in a list of
    a sequence of numbers.
    
    Eg: List = [1,2,3,4,6,7,8,9,10]
        Return = 5

    Args:
        element_list (list): A list of integers.
        
    Returns:
        missing_elements (list): A list of the integer values that
                                 are missing in 'element_list'.
    """
        
    start, end = element_list[0], element_list[-1]
    
    missing_elements = sorted(set(range(start, end + 1)).difference(element_list))
    
    return missing_elements


def _reset_community_numbers(partition, community_list, missing_community):
    """
    This function moves the nodes of the last community into the 
    community that is currently empty. It is used to reset the community
    numbers when a community becomes empty due to nodes moving to
    other communities.
    
    Eg: Communities = [1,2,3,4,6,7,8,9,10]
        Returns = [1,2,3,4,5,6,7,8,9]

    Args:
        partition (dict): Keys represent nodes, and values represent the
                          community that each node belongs to.
        community_list (list): A list of integers representing community
                               numbers
        missing_community (int): The empty community
        
    Returns:
        partition (dict): Keys represent nodes, and values represent the
                          community that each node belongs to. The
                          partition that results from moving nodes in the
                          last community to the empty community.
    """
    
    last_community = community_list[-1]
    
    for node, community in partition.items():
        if community == last_community:
            partition[node] = missing_community
        
    return partition


def _process_new_node(Graph, partition_1, node_u):
    """
    This function is used to determine whether a new node should create
    it's own community, along with neighbor nodes, or whether the new
    node should be assigned to another community.

    Args:
        Graph (graph):      A graph consisting of nodes and edges.
        partition_1 (dict): Keys represent nodes, and values represent
                            the community that each node belongs to.
        node_u (any):       A new node that was added to the graph. May
                            be an integer or string or object, etc.

    Returns:
        partition_1 (dict): A dictionary where keys are nodes, and the
                            values are the communities that each node is
                            assigned to. Partition_1 is the community
                            structure that results from a node and some
                            of its neighbors being assigned to an
                            entirely new community.

        OR

        partition_2 (dict): A dictionary where keys are nodes, and the
                            values are the communities that each node is
                            assigned to. Partition_2 is the community
                            structure that results from a node being
                            assigned to an already existing community.
    """

    # Add node u to it's own community
    community_u = max(list(partition_1.values())) + 1
    partition_1[node_u] = community_u

    # The while loop alters the partition, but the for loop
    # afterwards may need to undo those changes
    partition_2 = deepcopy(partition_1)

    done = False

    non_visited_neighbors = Graph.neighbors(node_u)

    while not done:

        F_in_neighbors = {}
        F_out_neighbors = {}
        qualifying_nodes = {}

        # Compute Fin and Fout for each of u's neighbours v
        for v in non_visited_neighbors:

            F_in_neighbors[v] = _compute_fin(v, Graph, partition_1)
            F_out_neighbors[v] = _compute_fout(v, Graph, partition_1,
                                               community_u)
            
            if F_out_neighbors[v] > F_in_neighbors[v]:
                qualifying_nodes[v] = F_out_neighbors[v]

        # Now sort the neighbors (v's) according to their fout score
        # and add the first node to the new community
        # If there are no qualifying nodes, then end the while loop.
        if qualifying_nodes:

            # Sort in descending order
            sorted_nodes = sorted(qualifying_nodes.items(),
                                  key=operator.itemgetter(1),
                                  reverse=True)

            # Add first element from sorted_nodes to the community of u
            # Remove element from non_visited_neighbors
            partition_1[sorted_nodes[0][0]] = community_u
            non_visited_neighbors.remove(sorted_nodes[0][0])

            # If the last neighbor has been processed and removed
            if not non_visited_neighbors:
                done = True

        # If there are no qualifying nodes
        else:
            done = True

    # Find F_in for u inside community C(u)
    F_in_u = _compute_fin(node_u, Graph, partition_1)

    # Find neighbor communities of C(u)
    neighbor_communities = _find_neighbour_communities(community_u,
                                                       Graph,
                                                       partition_1)
    
    if neighbor_communities:
    
        # Find f_out for u exerted by each neighbor community
        all_F_out_u = {}
        for neighbor_community in neighbor_communities:
            all_F_out_u[neighbor_community] = _compute_fout(node_u, Graph,
                                                            partition_1,
                                                            neighbor_community)

        max_fout_u = max(all_F_out_u.values())

        # Test if the maximum f_out is greater than f_in for node u,
        # if so, then place u in that community, while all other nodes
        # of C(u) go back to their original communities
        if max_fout_u > F_in_u:
            # Get the community of the maximum f_out
            for community, fout in all_F_out_u.items():
                if fout == max_fout_u:
                    key = community

            # Assign u to the other community.
            # Other nodes should remain in their previous communities
            # Partition_2 has all the v nodes back in their original
            # communities
            partition_2[node_u] = key
            return partition_2

        # Else the force that community C(u) exerts on node u
        # is the strongest and hence community C(u), remains as is,
        # after the modifications in the while loop
        else:
            return partition_1
        
    else:
        return partition_1


def _process_new_edge(Graph, partition, node_1, node_2):
    """
    This function takes in a new edge and determines whether either of
    the nodes attached to that edge, should move to the other nodes
    community. If a node should move to the other nodes community,
    the function then determines if any of that nodes neighbors should
    move to any other community.

    Args:
        Graph (graph):      A graph consisting of nodes and edges.
        partition (dict):   Keys represent nodes, and values represent
                            the community that each node belongs to.
        node_1 (any):       A node that represents one point of the new
                            edge that was added to the graph. May be an
                            integer or string or object, etc.
        node_2 (any):       A node that represents the other point of
                            the new edge that was added to the graph.
                            May be an integer or string or object, etc.

    Returns:
        partition_1 (dict): A dictionary where keys are nodes, and the
                            values are the communities that each node is
                            assigned to. Partition is the community
                            structure that results from a node in an
                            edge moving to the other nodes community,
                            which may include that nodes neighbors
                            moving as well.

    """

    # Check if the nodes are not in the same community
    if partition[node_1] != partition[node_2]:

        # Now Compute the two delta(q) values
        delta_qu, delta_qv = _compute_q_uv(node_1, node_2, Graph,
                                           partition)

        # Modularity does not change, so leave as is
        if (delta_qu == delta_qv) or \
                ((delta_qu <= 0) and (delta_qv <= 0)):
            return partition

        # Decide which node to move to the other community
        elif (delta_qu > 0) or (delta_qv > 0):
            if delta_qu > delta_qv:
                w = node_1
                new_community = partition.get(node_2)
            elif delta_qv > delta_qu:
                w = node_2
                new_community = partition.get(node_1)

        # Move node (w) to the new community
        partition[w] = new_community

        # The modularity of the current community structure,
        # prior to moving any of w's neighbors
        modularity_initial = _compute_modularity(Graph, partition)
        
        # Decide if w's neighbours should also move
        # Get neighbours of w
        neighbors_w = Graph.neighbors(w)
        
        changes = _find_best_community_for_neighbors(neighbors_w, Graph,
                                                    partition,
                                                    modularity_initial)

        # Make changes
        for n, c in changes.items():
            partition[n] = c

        return partition

    else:
        return partition


def community_detection(G_new, edge_list, previous_partition):
    """
    This function performs community assignment on an edge list, one
    row/edge at a time, or on a singular node. The function determines
    whether the edge already exists within the current graph, or if
    either node exists in the current graph. Based on this outcome, the
    function processes the new node or the new edge using the
    appropriate functions. It then returns an updated community
    structure once all the edges in the edge list have been processed.
    It also takes in a singular node with no edges and adds it to its
    own community.

    Args:
        G_new (graph):  A graph consisting of nodes and edges. This
                        graph evolves as each node / edge is added to
                        the graph.

        edge_list (list): A list of 2-tuples or a list consisting
                          of one new node with no edges (neighbours).
                          New nodes and edges are added to graph G_new.

        previous_partition (dict):  A dictionary where keys are nodes,
                                    and the values are the communities
                                    that each node is assigned to. The
                                    previous partition prior to adding
                                    new nodes and edges.

    Returns:
        new_partition (dict):   A dictionary where keys are nodes, and
                                the values are the communities that each
                                node is assigned to. The new partition
                                where new nodes and edges are added to
                                the graph and nodes are assigned to
                                communities.

        G_new (graph):  A graph consisting of nodes and edges. This
                        is the graph that results from adding new nodes
                        and edges.
    """

    new_partition = deepcopy(previous_partition)

    # If a singular node with no edges is sent to the function
    if not isinstance(edge_list[0], tuple):
        new_node = edge_list[0]
        has_node = G_new.has_node(new_node)

        if has_node:
            return new_partition, G_new

        G_new.add_node(new_node)
        new_partition[new_node] = max(previous_partition.values()) + 1

        return new_partition, G_new

    # Process the list of 2-tuples (edges)
    for row in edge_list:

        # Three cases occur depending on whether:
        # 1) One node is not in the graph,
        # 2) Both nodes are not in the graph,
        # 3) Both nodes are in the graph but the edge between
        #    them is not

        node_1 = row[0]
        node_2 = row[1]

        # In the event that some users send tweets to themselves,
        # ignore these
        if node_1 == node_2:
            continue

        has_edge = G_new.has_edge(node_1, node_2)
        has_node_1 = G_new.has_node(node_1)
        has_node_2 = G_new.has_node(node_2)

        if (not has_edge) and ((not has_node_1) or (not has_node_2)):

            if (not has_node_1) and (not has_node_2):
                # Add BOTH new nodes to one new community. If we add
                # the first node to its own community, then process the
                # second node by adding it to its own community, and
                # then determining if its neighbors should be added to
                # that community, then both node 1 and node 2 will end
                # up in the same community (node_2 will only have one
                # neighbour, ie node_1, which will "pull" node_2 into
                # its community). F_out will always be > F_in

                G_new.add_edge(node_1, node_2)

                new_community = max(list(new_partition.values())) + 1

                new_partition[node_1] = new_community
                new_partition[node_2] = new_community

            elif (not has_node_1) and (has_node_2):
                G_new.add_edge(node_1, node_2)

                new_partition = _process_new_node(G_new,
                                                  new_partition,
                                                  node_1)

            elif (has_node_1) and (not has_node_2):
                G_new.add_edge(node_1, node_2)

                new_partition = _process_new_node(G_new,
                                                  new_partition,
                                                  node_2)
                
        elif (not has_edge) and (has_node_1) and (has_node_2):
            G_new.add_edge(node_1, node_2)
            
            new_partition = _process_new_edge(G_new,
                                              new_partition,
                                              node_1,
                                              node_2)
            
            # Check if an empty community resulted from the 
            # processing of the new edge
            community_list = list(set(new_partition.values()))
            missing_community = _find_missing_elements(community_list)
            
            # Move the nodes in the last community to the empty
            # community
            if missing_community:
                new_partition = _reset_community_numbers(new_partition,
                                                         community_list,
                                                         missing_community[0])
                       
        # Skip duplicates
        elif (has_edge) and (has_node_1) and (has_node_2):
            continue

    return new_partition, G_new


def main(training_set_size=500, testing_set_size=500):
    """
    This is as example of how to use the "community_detection" function.
    Executing the main function will perform community detection on a
    random graph of 1000 nodes. The Louvain community detection
    algorithm will be applied to the first 500 nodes (and accompanying
    edges) in order to obtain the initial community structure.
    Thereafter, the dynamic community detection algorithm will be
    applied to the remaining 500 nodes (and accompanying edges).

    Args:
        training_set_size (integer):    Number of nodes in the initial
                                        training set

        testing_set_size (integer):     Number of nodes in the testing
                                        set

    """

    test_set_range = list(range(training_set_size,
                                training_set_size+testing_set_size))

    G_training = nx.erdos_renyi_graph(1000, 0.01)
    G_testing = G_training.subgraph(test_set_range)
    G_training.remove_nodes_from(test_set_range)

    # Perform Louvain community detection on the initial data set
    # "initial_partition" is a list of key value pairs. Key = node,
    # Value = community the node is assigned to
    initial_partition = community.best_partition(G_training)

    print("number edges", G_testing.number_of_edges())

    dynamic_partition = community_detection(G_training,
                                            G_testing.edges(),
                                            initial_partition)

    logger.info("Final Partition %s %s %s",
                 '\n', dynamic_partition, '\n')

if __name__ == "__main__":
    main()



