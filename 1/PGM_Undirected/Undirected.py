# THIS CODE HAS TO BE RUN ON PYTHON 2
# Otherwise, you will get wrong results

from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
import numpy as np

# Construct a graph
PGM = MarkovModel()
PGM.add_nodes_from(['w1', 'w2', 'w3'])
PGM.add_edges_from([('w1', 'w2'), ('w2', 'w3')])
tr_matrix = np.array([1,10,3,2,1,5,3,3,2])
phi = [DiscreteFactor(edge, [3, 3], tr_matrix) for edge in PGM.edges()]
PGM.add_factors(*phi)

# Calculate partition funtion
Z= PGM.get_partition_function()
print('The partition function is:', Z)

# Calibrate the click
belief_propagation = BeliefPropagation(PGM)
belief_propagation.calibrate()

# Output calibration result, which you should get
query=belief_propagation.query(variables=['w2'])
print('After calibration you should get the following mu(S):\n',query*Z)

# Get marginal distribution over third word
query=belief_propagation.query(variables=['w3'])
print('Marginal distribution over the third word is:\n',query)

#Get conditional distribution over third word
query=belief_propagation.query(variables=['w3'], evidence = {'w1':0}) # 0 stays for "noun"
print('Conditional distribution over the third word, given that the first word is noun is:\n', query)
