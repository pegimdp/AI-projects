#!/usr/bin/env python
# coding: utf-8

# # Genetic Algorithm for Logic Gates
# In this problem, a truth-table with n columns of input and one column of output is given and the goal is to find a set of logic gates (n-1 gates specifically) that produce the output. <br>
# Here we will use a method inspired by nature and natural selection which will result in the best set of logic gates and the output of the final gate will give the correct output (the same output as the truth table).<br>
#     In each row, the first two columns are given to the first gate and the consecutive gates take the output of the previous gate and the next row as input.Six basic logic gates i.e. AND, OR, XOR, NAND, NOR, XNOR can be used as each gate.The output of the model should be a specific comibination of the available gates.<br>
#     The problem is solved in 4 phases using a general genetic algorithm.
#      <img src="logic_gates.png"> 

# For 5 input values and 4 gates we would have the following circuit:
# <img src="example.png"> 

# ## Packages ##
# 
# First, let's run the cell below to import all the packages that you will need during this assignment. 
# - [numpy](www.numpy.org) is the fundamental package for scientific computing with Python.
# - [pandas](https://pandas.pydata.org) is a fast and easy to use open source data analysis and manipulation tool built on top of Python. 
# - [random](https://docs.python.org/3/library/random.html) This module implements pseudo-random number generators for various distributions.
# -[itertools](https://docs.python.org/2/library/itertools.html) This module implements a number of iterator building blocks inspired by constructs from APL, Haskell, and SML.
# - [operator](https://docs.python.org/3/library/operator.html) The operator module exports a set of efficient functions corresponding to the intrinsic operators of Python.

# In[103]:


import numpy as np 
import pandas as pd
import random
import itertools
import operator


# ## Phase 1: Specifying Gene and Chromosome Concepts in the Problem

# A chromosome consists of a set of genes and this set is a suggested solution for the problem.Here we define the genes and the chromosomes as follows:<br>
#   >  <b>Gene</b>: Every gene is a single gate.<br>
#      <b>Chromosome</b>: Every chromosome is a permuation of genes and is a suggested solution.<br>
#      <b>Population</b>: Every population consists of different chromosomes i.e. different possible solutions to the problem.</b><br>

# ### Implementation note 
# Genes are considered to be functions (gates) that are applied to the binary input and chromosomes a list (vector) of numbers from 0 to 5 that correspond these digits to AND, OR, XOR, NAND, NOR and XNOR gates respectively.Population is an m x n matrix where m is the size of the population and n the number of input values (columns of the given truth table minus 1).

# Let us first define the logic gates aka genes:

# In[125]:


gate_types = 6
gates_labels = [] #corresponding to AND,OR,XOR,NAND,NOR,XNOR respectively in this problem
for gate_num in range(gate_types):
    gates_labels.append(gate_num)
    
'''defining AND, OR and XOR gates, the rest of the gates can be constructed by taking logical complements, 
which is possible by applying the not operator on them.
'''
def AND_gene(a, b): #NAND is not AND
    if a == 1 and b == 1:
        return True
    else:
        return False

def OR_gene(a, b): #NOR is not OR
    if a == 1 or b == 1:
        return True
    else:
        return False
    
def XOR_gene(a, b): #XNOR is not XOR
    if a == 1 and b == 1:
        return False
    return OR_gene(a, b)


# ### Loading CSV Truth Table ### 
# To know the dimension of the chromosomes and hence the dimension of the population matrix we first need to read the the truth tabel.For n+1 total columns in the csv file there will be n-1 gates as explained above.

# In[40]:


truth_table = pd.read_csv("truth_table.csv")
truth_table = truth_table.astype(int) #replacing True and False values with 1 and 0 respectively
truth_table.describe() #basic statistical detail about the truth table


# In[41]:


number_of_columns = len(truth_table.columns)
number_of_gates = number_of_columns-2 # 1 less than the number of input columns


# ## Phase 2: Initial Population Production

# We first define an empty population matrix as an array with no array (chromosome) inside, declaring the rows' dimension.We then add the initial chromosomes and create the primary population.<br>
# Our primary chromosomes are considered to be sequences of the same repeated gene each.In this case, if they become combined for the next generations we could have different genes inside the children.

# In[84]:


#defining the population matrix rows as large as all the possible combinations of gates
all_populations = [] #defining the empty population matrix
current_population = []
for gate in gates_labels:
    this_chromosome = [gate] * number_of_gates
    all_populations.append(this_chromosome)
    current_population.append(this_chromosome)
all_populations #we have initialized the primary population with len(gates_labels) chromosomes


# Hence, the initial population has 6 (#number of types of gates) members.

# ## Phase 3: Implementation and Fitness Criterion

# ### Implementation of Gates/Genes
# To apply the gates in a possible solution (chromosome) to a row, we define a function that matches the gate label in the chromosome to the respective gate function and produces the expected output based on the given input (as explained in problem formulation).<br>
# Firstly, we have a function which matches every gate label to it's corresponding function:

# In[127]:


#0,1,2,3,4,5 are matched to AND,OR,XOR,NAND,NOR,XNOR respectively in this problem
def apply_gate(value1, value2, gate_label):
    if gate_label == 0:
        return AND_gene(value1, value2)
    elif gate_label == 1:
        return OR_gene(value1, value2)
    elif gate_label == 2:
        return XOR_gene(value1, value2)
    elif gate_label == 3:
        return not AND_gene(value1, value2)
    elif gate_label == 4:
        return not OR_gene(value1, value2)
    elif gate_label == 5:
        return not XOR_gene(value1, value2)
    else: 
        raise Exception("Invalide gene!")


# Therefore, we will have a circuit function which applies the gates one-by-one:

# In[121]:


def apply_circuit(given_row_input, given_chromosome):
    input_index = 0 #keeping track of the indices we have visited in the input row
    input1 = given_row_input[input_index]
    input_index +=1
    input2 = given_row_input[input_index]
    for gate in given_chromosome:
        input1 = apply_gate(input1,input2,gate) #the result of the gate is used as input for the next gate
        if input_index == len(given_row_input)-1: #we have reached the last input value
            return input1 #the last gate result was the final result
        input_index += 1
        input2 = given_row_input[input_index] #reading the next input from the truth table


# ### Fitness Criterion
# To evaluate how a specific solution (circuit/chromosome) is suitable for the given truth table, we apply the circuit to all the rows in the table and calculate a $fitness$ $ratio$ function that shows the ratio of the correctly calcultaed rows to all the rows in the given truth table.The chromosomes whose outputs matches that of the table the most, are the fittest.Hence, the chromosomes with a fitness ratio of more than 70% are <b> selected </b> as parents for mutation and reproduction (crossover) of the next generation in the following phase. 

# In[123]:


def fitness_ratio(given_chromosome):
    num_of_corrects = 0 #number of correct results
    for index, line in truth_table.iterrows():
        input_array = [line["Input1"], line["Input2"], line["Input3"], line["Input4"], line["Input5"]
                           ,line["Input6"], line["Input7"], line["Input8"], line["Input9"], line["Input10"]]
        this_result = apply_circuit(input_array, given_chromosome)
        if this_result == line["Output"]:
            num_of_corrects +=1
    return num_of_corrects/(truth_table.shape[0])

def is_solution(given_population):
    for chromosome in given_population:
        if fitness_ratio(chromosome) == 1:
            return True, chromosome
    return False, []       


def select_parents(given_population):
    parents = []
    for chromosome in given_population:
        if fitness_ratio(chromosome) >= 0.7:
            parents.append(chromosome)
    return parents       


# ## Phase 4: Mutation and Crossover Implementation and Reproduction

# Based on natural selection the fittest chromosomes survive and move on to the next generation.These chromosomes are the parents for the next generation.<br>
# That means the next generation contains 3 types of chromosomes:
# - Parents (chromosomes from the previous generation with a fitness ratio of 0.7 and above)
# - Children produced by crossover (combination of every two parents)
# - Children produced by mutation (of every single parent)

# ### Genetic Algorithm Reproduction Cycle
# 1.Select parents for the mating pool <br>
# 2.Shuffle the mating pool.<br>
# 3.For each consecutive pair apply crossover with probability $p_c$ , otherwise copy parents.<br>
# 4.For each offspring apply mutation (bit-flip with probability $p_m$ independently for each bit).<br>
# 5.Replace the whole population with the resulting offspring.

# ### Crossover 
# A crossover child is created by splitting two chromosomes at $m$ random points and joining the split parts from every other chromosome.Here $m = 3$ seems a logical choice as we have 9 gates in total.

# In[133]:


def crossover_child(chrom1, chrom2):
    chrom_len = len(chrom1) #both chromosomes have the same length
    indices = random.sample(range(1, chrom_len-1), 3) #3 random numbers for splitting from the first to one to last index
    child = [chrom1[:indices[0]]+chrom2[indices[0]:indices[1]]+chrom1[indices[1]:indices[2]]+chrom2[indices[2]:]]
    return


# ### Mutation
# Mutation children are created by applying random changes to a single parent in the previous generation.Here we randomly select 4 genes and replace the gene (gate) by the next gene (gate) in the list defined in the beginning.This lets us try different combinations of gates through iterations.

# In[99]:


def mutated_child(chromosome):
    rand_indices = random.sample(range(0, len(chromosome)), 4)
    for index in rand_indices:
        chromosome[index] = (chromosome[index]+1)%gate_types
    return chromosome


# ### Next Generation
# Here we define a function which takes the parents for the next generation and ultimately changes the current and all populations (that are defined as global variables).

# In[ ]:


def produce_next_generation(given_parents):
    random.shuffle(given_parents)
    new_generation = []
    p_c = 0.4
    p_m = 0.5
    random_c = np.random.uniform(low = 0.0, high = 1.0, size = None)
    random_m = np.random.uniform(low = 0.0, high = 1.0, size = None)
    if random_c >= p_c:
        for parent1, parent2 in itertools.combinations(given_parents,2):
            new_child = crossover_child(parent1, parent2)
            if random_m < p_m and new_child not in all_populations: #if it is in previous populations it's not an optimal solution
                new_generation.append(new_child)
            else:
                new_child = mutated_child(new_child)
                if new_child not in all_populations: #if it is in previous populations it's not an optimal solution
                    new_generation.append(new_child)
    current_population = new_generation
    all_populations.extend(current_population)


# ## Phase 5: Possible Questions
# 

# 1) What is the reason behind choosing our fitness criterion?
# > In our fitness criterion we chose the chromosomes their outputs were correct for more that $7/10$ of all input lines.Being correct for these many inputs gets us much closer to the final correct circuit(chromosome) and there are less gates(genes) that have to be changed in comparison with the unfit chromosomes.These changes are done in crossover and mutation.

# 2) What is the method for choosing the next generation? Why did we choose this method?
# > As mentioned above, the fittest parents from the previous generation and the result of their mutation and crossover creat the next generation.<br>
# By using this method we choose the chromosomes closest to the actual solution and by mutating and combining them (by crossover) it is more likely to reach the solution faster.

# 3) What is the affect of mutation and crossover? What is their probability?
# > In our fitness criterion we chose the chromosomes their outputs were correct for more that $7/10$ of all input lines.We have two random numbers that are produced by random uniform distributions (random_c and random_m).If these are at least the given probabilities (p_c and p_m respectively).By mutation and crossover we change and combine the chromosomes closest to the solution and it becomes much more possible to reach the solution.

# 4) Given the methods above, there is still the possibility of the chromosomes not changing (the same chromosomes being copied in every generation).Why does this happen? Is there any solution?
# > If we don't arrive to the solution early on, it is likely for the model to stay with the same chromosomes in geneartions.The reason for that is that the change in chromosomes happens in crossover and mutation and these two algorithms are only executed with p_c and p_m probabilities.Which means crossover does not become executed with a probability of 1-p_c and mutation is not executed probability of (1-p_c)(1-p_m).<br>
# We can omit the probabilities and apply crossover and mutation for all pairs and single chromosomes respectively.
# 

# <large> We have the new reproduction function based on the explanation in question4: 

# In[147]:


def produce_next_generation(given_parents):
    random.shuffle(given_parents)
    new_generation = []
    p_c = 0.5
    p_m = 0.7
    random_c = np.random.uniform(low = 0.0, high = 1.0, size = None)
    random_m = np.random.uniform(low = 0.0, high = 1.0, size = None)
    for parent1, parent2 in itertools.combinations(given_parents,2):
        new_child = crossover_child(parent1, parent2)
        new_child = mutated_child(parent)
        if new_child not in all_populations: #if it is in previous populations it's not an optimal solution
            new_generation.append(new_child)
    current_population = new_generation
    all_populations.extend(current_population)


# ## Final Model and Solution
# Using all the parts above, we produce the final solution which is the chromosome that gives all the correct output.

# In[ ]:


def model():
    solved = False
    chromosome = []
    while solved != True:
        parents = select_parents(current_population)
        produce_next_generation(parents)
        solved, chromosome = is_solution(current_population)
    return chromosome

sol = model()
print(sol)


# ## Summary
# We started with a primary population of chromosomes(circuits) and chose the fittest chromosomes as the parents of next generations.The next generation is produced by applying crossover and mutation to these parents.If a chromosome fits the population 100%, it becomes the solution and the process is stopped, otherwise the process continues.
#  <img src="GA_cycle.png"> 
