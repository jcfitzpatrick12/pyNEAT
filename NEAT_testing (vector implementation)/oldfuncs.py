#first attempt at "add edge no cycles"


#returns true if there exists a path from the edge indexed by (start,end)
#our "adjacency matrix" is the matrix of enable bits. In this way, we ignore explicatly disabled nodes
#we use a recursive function
def DFS(self,proposed_genome,start,end,visited=None):
    sliced_genome=proposed_genome.return_enabled_genome()
    adj_matrix = np.copy(sliced_genome.connection_enable_bits) 
    print(adj_matrix) 
    if visited is None:
        visited=set()
    #add the index of the starting node!
    visited.add(start)
    # If we've reached the end node, we've found a path
    if start == end:
        return True
    # For adjacency matrices, we iterate over the column of the current row
    # If adj_matrix[start][i] is not zero, there's an edge from 'start' to 'i'
    for i, edge_exists in enumerate(adj_matrix[start]):
        if edge_exists and i not in visited:
            if self.DFS(proposed_genome, i, end, visited):
                return True
    return False

'''
to add an edge randomly, making sure there is no cycle, we simply add an edge then determine whether it creates a cycle
using a Randomized Depth-First Search (DFS). If it creates a cycle, then we select a new edge.
For simplicity currently, we do "keep track" of which edges have been proposed.
'''
def add_link_no_cycles(self,arb_genome):
    #add any random edge, and return the mutated genome and the edge we activated
    proposed_genome,edge_to_activate = self.add_link_allow_cycles(arb_genome)
    #determine whether the proposed edge_to_activate creates a cycle
    cycle_exists = self.DFS(proposed_genome,edge_to_activate[0],edge_to_activate[1])
    #cycle_exists==True
    #check whether the edge we introduced introduced a cycle or not
    if cycle_exists==True:
        #loose upper bound on the amount of times we should be able to iterate
        #we estimate this as the number of enabled edges
        N = np.nansum(arb_genome.connection_enable_bits)
        #start the increment at n=0
        n=0
        #while we have a cycle, repeat the above until we don't have a cycle
        while cycle_exists == True and n<=N:
            #print(n)
            #add any random edge
            proposed_genome,edge_to_activate = self.add_link_allow_cycles(arb_genome)
            #determine whether the proposed edge_to_activate creates a cycle
            cycle_exists = self.DFS(proposed_genome,edge_to_activate[0],edge_to_activate[1])
            #increment our n to keep track of how many times we've tried to add an edge
            n+=1
        if n>N:
            print('We could not add an edge which did not introduce a cycle. Returning genome unchanged.')
            return arb_genome
    #use a depth first check
    #only once cycle_exists == False, can we return the proposed_genome
    return proposed_genome