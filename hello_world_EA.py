#
# "HELLO WORLD!" EVOLUTIONARY ALGORITHM  
#  BY DR.ALBERTO Alberto Moraglio
# FROM (ECM2423) Artificial Intelligence and Applications 
#


print ("\n")

from random import choice, random 

### EVOLUTIONARY ALGORITHM ###

def evolve():
    population = create_pop()
    # print (population)
    fitness_population = evaluate_pop(population)
    # print (fitness_population)
    
    for gen in range(NUMBER_GENERATION):
        # print(population, fitness_population)
        mating_pool = select_pop(population, fitness_population)

        # print("MATING POOL")
        # print (mating_pool)
        
        
        offspring_population = crossover_pop(mating_pool)

        # print ("----------------------------------------")
        # print (offspring_population)
        
        population = mutate_pop(offspring_population)
        fitness_population = evaluate_pop(population)
        best_ind, best_fit = best_pop(population, fitness_population)
        
        print ("#%3d" % gen, "fit:%3d" % best_fit, "".join(best_ind))
        print ("/////////////////////////////////////////////////")

### POPULATION-LEVEL OPERATORS ###

def create_pop():
    return [ create_ind() for _ in range(POPULATION_SIZE) ]

def evaluate_pop(population):
    return [ evaluate_ind(individual) for individual in population ]

def select_pop(population, fitness_population):
    sorted_population = sorted(zip(population, fitness_population), key = lambda ind_fit: ind_fit[1])
    # print ("SORTING")
    # print (sorted_population)
    # print ("SORTED")
    return [ individual for individual in sorted_population[:int(POPULATION_SIZE * TRUNCATION_RATE)] ]

def crossover_pop(population):
    return [ crossover_ind(choice(population), choice(population)) for _ in range(POPULATION_SIZE) ]

def mutate_pop(population):
    return [ mutate_ind(individual) for individual, score in population ]

def best_pop(population, fitness_population):
    return sorted(zip(population, fitness_population), key = lambda ind_fit: ind_fit[1])[0]

### INDIVIDUAL-LEVEL OPERATORS: REPRESENTATION & PROBLEM SPECIFIC ###

target  = list("HELLO WORLD!")
alphabet = " !ABCDEFGHIJLKLMNOPQRSTUVWXYZ"
INDIVIDUAL_SIZE = len(target)

def create_ind():
    return [ choice(alphabet) for _ in range(INDIVIDUAL_SIZE) ]

def evaluate_ind(individual):
    return sum( i != t for i, t in zip(individual, target) )

def crossover_ind(individual1, individual2):
    return [ choice(ch_pair) for ch_pair in zip(individual1, individual2) ]

def mutate_ind(individual):
    return [ (choice(alphabet) if random() < MUTATION_RATE else ch) for ch in individual ]

### PARAMERS VALUES ###

NUMBER_GENERATION = 20
POPULATION_SIZE = 400
TRUNCATION_RATE = 0.5
MUTATION_RATE = 1.0 / INDIVIDUAL_SIZE

### EVOLVE! ###

evolve()




