Overview:

This algorithm performs community detection on an edge list one edge at a time, instead of batch processing an entire
edge list all at once (like Louvain). The algorithm works by first performing Louvain community detection on a subset
of the edge list, to determine the initial set of communities. A subsequent edge list is input into the 
"community_detection" function where each edge is processed one at a time, and nodes are either assigned to existing 
communities, assigned to new communities, or moved from one community to another.


Source:

The source of the dynamic community detection algorithm implemented here is:

	Dynamic Social Community Detection and Its Applications (2014) - Nam P. Nguyen, Thang N. Dinh, Yilin Shen, My T. Thai
	

Requirements:

python-louvain==0.10

networkx==1.11

numpy==1.13.1


Execution:

Sample execution in main (Returns final_partition):

	python dynamic_community_detection.py

To use community detection algorithm:

	final_partition = community_detection(G_new, edge_list, previous_partition)


Full documentation can be found in the source file.
