'''

This code template belongs to
"
PGM-TUTORIAL: EVALUATION OF THE 
PGMPY MODULE FOR PYTHON ON BAYESIAN PGMS
"

Created: Summer 2017
@author: miker@kth.se

Refer to https://github.com/pgmpy/pgmpy
for the installation of the pgmpy module

See http://pgmpy.org/models.html#module-pgmpy.models.BayesianModel
for examples on fitting data

See http://pgmpy.org/inference.html
for examples on inference

'''
from pgmpy.inference import VariableElimination
from pgmpy.estimators.StructureScore import K2Score


def separator():
    input('Enter to continue')
    print('-'*70, '\n')
    
# Generally used stuff from pgmpy and others:
import math
import random
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel

# Specific imports for the tutorial
import pgm_tutorial_data
from pgm_tutorial_data import ratio, get_random_partition

RAW_DATA = pgm_tutorial_data.RAW_DATA
FEATURES = [f for f in RAW_DATA]

# Task 1 ------------ Setting up and fitting a naive Bayes PGM
data = pd.DataFrame(data=RAW_DATA)
model = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])
model.fit(data) # Uses the default ML-estimation

STATE_NAMES = model.cpds[0].state_names
print('State names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

print('')
print(model.cpds[0])
print(ratio(data, lambda t: t['age']=='>23', lambda t: t['delay']=='>=2'))


print(model.cpds[3])
print("delay | -")
print(ratio(data, lambda t: t['delay']=='0'))
print(ratio(data, lambda t: t['delay']=='1'))
print(ratio(data, lambda t: t['delay']=='>=2'))
print(ratio(data, lambda t: t['delay']=='NA'))


print(model.cpds[2])
print("avg_mat | delay")
print(ratio(data, lambda t: t['avg_mat']=='4<5', lambda t: t['delay']=='0'))
print(ratio(data, lambda t: t['avg_mat']=='4<5', lambda t: t['delay']=='1'))
print(ratio(data, lambda t: t['avg_mat']=='4<5', lambda t: t['delay']=='>=2'))
print(ratio(data, lambda t: t['avg_mat']=='4<5', lambda t: t['delay']=='NA'))

separator()

# End of Task 1


# Task 2 ------------ Probability queries (inference)

data = pd.DataFrame(data=RAW_DATA)
model = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])
model.fit(data) # Uses the default ML-estimation

print(len(model.cpds))

STATE_NAMES = model.cpds[0].state_names
print('State names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

ve = VariableElimination(model)
#q = ve.query(variables = ['delay'], evidence = {'age': 1})
# alternative way to call:

print("2.1")
q = ve.query(variables = ['delay'], evidence = {'age': '<=20'})
print(q)


print("2.2")
q = ve.query(variables = ['age'], evidence = {'delay': '0'})
print(q)


print("2.3")
print(ratio(data, lambda t: t['age']=='20-23', lambda t: t['delay']=='0'))
print(ratio(data, lambda t: t['age']=='<=20', lambda t: t['delay']=='0'))
print(ratio(data, lambda t: t['age']=='>23', lambda t: t['delay']=='0'))


print("2.4")
q = ve.map_query(variables = ['age'], evidence = {'delay': '0'})
print(q)


separator()
# End of Task 2



# Task 3 ------------ Reversed PGM

data = pd.DataFrame(data=RAW_DATA)
model = BayesianModel([('age', 'delay'),
                       ('gender', 'delay'),
                       ('avg_mat', 'delay'),
                       ('avg_cs', 'delay')])
model.fit(data) # Uses the default ML-estimation

STATE_NAMES = model.cpds[0].state_names
print('State names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

print("3.2")
print(np.prod(model.cpds[3].cardinality))

q = ve.query(variables = ['delay'])
print(q)

print("3.5 - Errors")
print(q.values[0] - ratio(data, lambda t: t['delay']=='0'))
print(q.values[1] - ratio(data, lambda t: t['delay']=='1'))
print(q.values[2] - ratio(data, lambda t: t['delay']=='>=2'))
print(q.values[3] - ratio(data, lambda t: t['delay']=='NA'))

separator()

# End of Task 3



# Task 4 ------------ Comparing accuracy of PGM models
from scipy.stats import entropy
'''
data = pd.DataFrame(data=RAW_DATA)

model1 = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])

model2 = BayesianModel([('age', 'delay'),
                        ('gender', 'delay'),
                        ('avg_mat', 'delay'),
                        ('avg_cs', 'delay')])

models = [model1, model2]

[m.fit(data) for m in models] # ML-fit

# this is a hack beter if you extract the names from the model.

STATE_NAMES = {'delay': ['0', '1', '>=2', 'NA'], 'age': ['20-23', '<=20', '>23'], 'avg_cs': ['2<3', '3<4', '4<5', '<2'], 'avg_mat': ['2<3', '3<4', '4<5', '<2'], 'gender': ['0', '1']} 

print('\nState names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

S = STATE_NAMES
VARIABLES = list(S.keys())


def random_query(variables, target):
    # Helper function, generates random evidence query
    n = random.randrange(1, len(variables)+1)
    evidence = {v: random.randrange(len(S[v])) for v in random.sample(variables, n)}
    if target in evidence: del evidence[target]
    return (target, evidence)

queries = []

#for target in ['delay']:  # 4.4
#for target in ['age']:  # 4.5
for target in ['age', 'delay']:  # 4.6
    variables = [v for v in VARIABLES if v != target]
    queries.extend([random_query(variables, target) for i in range(200)])

divs = []
# divs will be filled with lists on the form
# [query, distr. in data, distr. model 1, div. model 1, distr. model 2, div. model 2]
for v, e in queries:
    try:
        # Relative frequencies, compared below
        rf = [ratio(RAW_DATA, lambda t: t[v]==s,
                    lambda t:all(t[w] == S[w][e[w]] for w in e)) \
              for s in S[v]]
        # Special treatment for missing samples
        #### if sum(rf) == 0: rf = [1/len(rf)]*len(rf) # Commented out on purpose

        print(len(divs), '-'*20)
        print('Query:', v, 'given', e)
        print('rf: ', rf)
         
        div = [(v, e), rf]
        for m in models:
            print('\nModel:', m.edges())
            ve = VariableElimination(m)
            q = ve.query(variables = [v], evidence = e)
            div.extend([q.values, entropy(rf, q.values)])
            print('PGM:', q.values, ', Divergence:', div[-1])
        divs.append(div)
    except:
        # Error occurs if variable is both target and evidence. We can ignore it.
        # (Also, this case should be avoided with current code)
        pass

divs2 = [r for r in divs if math.isfinite(r[3]) and math.isfinite(r[5])]
# What is the meaning of what is printed below?
for n in range(1,5):
    print([n,
           len([r for r in divs2 if len(r[0][1])==n]),
           len([r for r in divs2 if len(r[0][1])==n and r[3] < r[5]]),
           len([r for r in divs2 if len(r[0][1])==n and r[3] > r[5]]),
           sum(r[3] for r in divs2 if len(r[0][1])==n),
           sum(r[5] for r in divs2 if len(r[0][1])==n),
           len([r for r in divs if len(r[0][1])==n and \
                not(math.isfinite(r[3]) and math.isfinite(r[5]))]),]
          )
'''

# End of Task 4




# Task 5 ------------ Finding a better structure

data = pd.DataFrame(data=RAW_DATA)

model1 = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])

model2 = BayesianModel([('age', 'delay'),
                        ('gender', 'delay'),
                        ('avg_mat', 'delay'),
                        ('avg_cs', 'delay')])

models = [model1, model2]

[m.fit(data) for m in models] # ML-fit

# STATE_NAMES = model1.cpds[0].state_names
# print('\nState names:')
# for s in STATE_NAMES:
#    print(s, STATE_NAMES[s])

# Information for the curious:
# Structure-scores: http://pgmpy.org/estimators.html#structure-score
# K2-score: for instance http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
# Additive smoothing and pseudocount: https://en.wikipedia.org/wiki/Additive_smoothing
# Scoring functions: https://www.cs.helsinki.fi/u/bmmalone/probabilistic-models-spring-2014/ScoringFunctions.pdf
k2 = K2Score(data)
print('Structure scores:', [k2.score(m) for m in models])

separator()

print('\n\nExhaustive structure search based on structure scores:')

from pgmpy.estimators import ExhaustiveSearch

# Warning: Doing exhaustive search on a PGM with all 5 variables
# takes more time than you should have to wait. Hence
# re-fit the models to data where some variable(s) has been removed
# for this assignement.
raw_data2 = {'age': data['age'],
             'avg_cs': data['avg_cs'],
             'avg_mat': data['avg_mat'],
             'delay': data['delay'], # Don't comment out this one
             #'gender': data['gender'],
             }

data2 = pd.DataFrame(data=raw_data2)

import time
t0 = time.time()
# Uncomment below to perform exhaustive search
searcher = ExhaustiveSearch(data2, scoring_method=K2Score(data2))
search = searcher.all_scores()
print('time:', time.time() - t0)

# Uncomment for printout:
for score, model in search:
    print("{0}        {1}".format(score, model.edges()))

separator()


# End of Task 5
