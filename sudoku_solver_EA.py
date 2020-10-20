#
#670058382
#Project by Josh Li
#FROM (ECM2423) Artificial Intelligence and Applications
#
import random
import numpy
import copy
import time 


#Given Grids, set as list of 9 rows

Grid1 = [[4,0,0,1,0,2,6,3,0],
      [5,0,0,6,4,3,8,0,0],
      [7,6,0,5,0,8,4,1,2],
      [6,0,0,0,0,9,3,4,8],
      [2,4,0,8,3,0,9,5,0],
      [8,0,9,4,1,5,0,7,0],
      [0,7,2,0,0,4,0,6,0],
      [0,5,4,2,0,0,0,8,9],
      [0,8,6,3,0,7,1,2,4]]

Grid2 = [[6,0,0,7,0,0,0,0,1],
      [7,0,0,0,0,9,2,0,0],
      [0,2,9,6,0,0,0,0,0],
      [0,5,7,0,3,6,1,0,4],
      [2,0,3,0,7,1,0,5,8],
      [1,8,0,2,9,0,0,6,3],
      [0,0,0,0,0,2,5,0,9],
      [4,9,6,3,0,0,8,0,0],
      [0,0,2,9,8,0,6,0,7]]

Grid3 = [[4,0,0,0,6,0,0,0,1],
      [9,0,0,0,0,3,0,5,0],
      [0,1,0,7,0,0,0,3,9],
      [8,0,6,0,0,0,0,0,0],
      [0,0,4,5,9,1,0,0,0],
      [0,0,0,3,0,6,0,0,0],
      [0,0,0,0,7,2,0,0,4],
      [2,5,1,6,0,0,9,0,0],
      [0,0,0,8,5,9,2,0,0]]

#To set your own sudoku problem, alter this grid
Grid0 = [[0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0]]


#Choose which grid you want to solve

q = numpy.array(Grid1)


#################### EVOLUTIONARY ALGORITHM #############################

### PARAMATERS VALUES ###
NUMBER_GENERATION = 1000
POPULATION_SIZE = 100
TRUNCATION_RATE = 0.10
MUTATION_RATE = 0.15
TERMINATION_GENS = 50



population = []
def evolve():

    #initialize and evaluate population
    population = create_pop()
    fitness_population = evaluate_pop(population)
    
    #keep track of a set amount of last generations for termination condition
    xvals = [(9**3)]*TERMINATION_GENS
    x = 0

    for gen in range(NUMBER_GENERATION):

        #select mating pool, and crossover with mating pool      
        mating_pool = select_pop(population, fitness_population)
        offspring_population = crossover_pop(mating_pool)

        #combine mating pool and offspring to form new population
        population = []
        for r in mating_pool:
            population.append(r)
        for s in offspring_population:
            population.append(s)

        #mutate and evaluate new population
        population = mutate_pop(population)
        fitness_population = evaluate_pop(population)
        
        #find best guess and print progress
        x = min(int(s) for s in fitness_population)
        print ("GEN: " + str(gen) + " Best Guess has: " + str(x) + " Conflicts") 

        #termination condition 1: if 0 conflicts puzzle solved, return individual
        if x == 0:
            y = fitness_population.index(x)
            print("Eureka! Solution found after " + str(gen) + " generations")
            return population[y] 
        
        xvals.append(x)
        x = xvals.pop(0)
        #take best guess from TERMINATION_GENS ago to compare with xvals

        #termination condition 2: no improvement over last TERMINATION_GENS generations
        if x <= min(int(t) for t in xvals):
            print ("Program terminated, best guess has " + str(min(int(s) for s in fitness_population)) + " Conflicts" )
            print ("No improvement in last " + str(TERMINATION_GENS) + " generations, Currently at Generation  " + str(gen))
            print ( "Please try again or alter PARAMETERS VALUES" )
            return False
        
        #termination condition 3, 1000 generations reached. 

### POPULATION-LEVEL OPERATORS ###

#calls appropriate individual level operators
def create_pop():   
    return[ create_ind() for _ in range(POPULATION_SIZE) ]

def evaluate_pop(population):
    return [ evaluate_ind(individual) for individual in population ]

def select_pop(population, fitness_population):
    #retruns best POPULATION_SIZE * TRUNCATION_RATE of population based on fitness
    sorted_population = sorted(zip(population, fitness_population), key = lambda ind_fit: ind_fit[1])
    sorted_population = [i[0] for i in sorted_population]  
    return sorted_population[:int(POPULATION_SIZE * TRUNCATION_RATE)] 

def crossover_pop(population):
    #cross over mating population POPULATION_SIZE*(1 - TRUNCATION_RATE) times
    #uses inverted geometric series to assign probability - 
    #best parent is 50% more likely to be picked than next best

    offspring = []
    probability = []
    n = len(population)
    r = 1.5
    lowp = (r-1.0)/(r**n-1) #probability of last(worst parent) to be picked

    #creates list of probability for numpy.random.choice
    for p in range(n):
        probability.append(lowp*(r**(n-p-1)))

    for _ in range(int(POPULATION_SIZE*(1 - TRUNCATION_RATE))):
        #randomly chooses without replacement two parents based on list of probability
        #calls crossover individual operator and builds offspring population list
        parents = numpy.random.choice(len(population),2, probability)
        child = crossover_ind(population[parents[0]], population[parents[1]])
        offspring.append(child)
    return offspring

def mutate_pop(population):
    return [ mutate_ind(individual) for individual in population ]
        


### INDIVIDUAL-LEVEL OPERATORS: REPRESENTATION & PROBLEM SPECIFIC ###



def create_ind():
    #each individual is a numpy matrix, each row consists of numbers 1-9 non repeating
    
    individual = []
    for i in range(0, 9): 
        #checks question grid to remove given values from list of 1-9
        #then randomly shuffles remaining values and reinserts them to create a row
        items = list(range(1, 10))
        row = q[i,:]
      
        for n in row:
            if n != 0:
                items.remove(n)
        random.shuffle(items)
        p = 0
        newrow = []
        for m in row:
            if m == 0:
                m = items[p]
                p += 1
            newrow.append(m)
        individual.append(newrow) 

    a = numpy.array(individual)
    return a 

def evaluate_ind(individual):
    #evaluates each individual for errors based on columns and boxes
    #err found by 9 - number of unique values per col or box
    err = 0
    toterr = 0   
    for j in range(0,9):
        col = individual[:,j]               
        seen = set()
        uniq = []
        for x in col:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        err = 9-len(uniq)
        toterr += err

    for x in range (0,3):
        for y in range (0,3):    
            box = individual[x*3:x*3+3,y*3:y*3+3]
            box2 = zip(*box)
            square = []
            for sublist in box2:
                for item in sublist:
                    square.append(item)
            seen = set()
            uniq = []
            for s in square:
                if s not in seen:
                    uniq.append(s)
                    seen.add(s)
            err = 9-len(uniq)
            toterr += err
    return toterr

def crossover_ind(individual1, individual2):
    #one point crossover, first parent copied until random crossover point
    #remaining values taken as order of second parent
    child = copy.copy(q) 
    for r in range(0,9):
        row1 = individual1[r,:]
        row2 = individual2[r,:]

        row1 = row1.tolist()
        row2 = row2.tolist()

        val = 0
        for k in range(0,9):
            if q[(r,k)] == 0:
                val += 1
      
            else:
                row1[k] = 0
                row2[k] = 0     
        cross = random.randint(0,val)
  
        mix = []

        for i in range(0,9):
            if row1[i] != 0:
                if len(mix) < cross:
                    mix.append(row1[i])
                    row2.remove(row1[i])
            
        for t in row2:
            if t != 0:
                mix.append(t)

     
                
        for p in range(0,9):
            if child[(r,p)] == 0:
                child[r,p]= mix.pop(0)
    return child


def mutate_ind(individual):
    #point mutation set by mutation rate
    #if mutate randomly shuffle non given values
    for r in range(0,9):
        nlist = []
        for k in range(0,9):
            if q[(r,k)] == 0:
                nlist.append(individual[(r,k)])
        
        if random.random() < (MUTATION_RATE):
            random.shuffle(nlist)
            
       
        for l in range(0,9):
            if q[(r,l)] == 0:
                individual[(r,l)] = nlist.pop(0)

    return individual


### EVOLVE! ###


start_time = time.time()
print (evolve())
print("Program took: %s seconds" % (time.time() - start_time))






