'''
how can we accomadate this to NOT visualise nodes that are not in the network (but nonetheless in the adjacency matrices)
'''


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class visualise_genome:
    def __init__(self, genome_toplot):
        self.genome_toplot = genome_toplot

    def see_network(self):
        # Create a weighted adjacency matrix that IGNORES disabled edges
        adj_mat = np.multiply(self.genome_toplot.connection_weights, self.genome_toplot.connection_enable_bits)
        #round the weights for neat plotting
        adj_mat=np.round(adj_mat,2)
        #convert all nan values to zeros for plotting
        adj_mat[np.isnan(adj_mat)]=0
        # Create DiGraph from the adjacency matrix
        G = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph)

        # Use circular_layout to arrange nodes in a circle
        layout = nx.circular_layout(G)

        # Get the number of nodes in the graph
        num_nodes = len(G.nodes)

        # Use a list for node sizes
        sizes = [1000] * num_nodes

        # Use a list for node colors
        color_map = ['pink'] * num_nodes

        # Create a new figure and axes
        fig, ax = plt.subplots()

        # Draw the graph using the layout - with_labels=True if you want node labels
        nx.draw(G, layout, with_labels=True, node_size=sizes, node_color=color_map, ax=ax)

        # Get weights of each edge and assign to labels
        labels = nx.get_edge_attributes(G, "weight")

        # Draw edge labels using the axes object and the list of labels
        nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels, ax=ax)

        plt.show()
        

    def plot_network(self):
        self.see_network()
        # Show plot
        plt.show()
        pass
