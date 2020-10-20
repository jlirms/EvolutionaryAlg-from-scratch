# -*- coding: utf-8 -*-
"""
Evolutionary/Genetic algorithm from scratch for the Superstring Scrabble Challenge
# find a superstring with all scrabble tiles which score the most points
https://fivethirtyeight.com/features/whats-your-best-scrabble-string/

"""

import numpy as np
import pandas as pd
from random import randint
from random import choice, sample, uniform
import string
#%%

dfwords = pd.read_csv('./scrabble_scores/wordscore.csv', index_col = 'WORD', keep_default_na=False, na_values=[''])
di = dict.fromkeys(string.ascii_lowercase, 0)

for pt in ['e','a','i','o','u','n','r','t','l','s']:
    di[pt] = 1
for pt in ['d','g']:
    di[pt] = 2
for pt in ['b','c','m','p']:
    di[pt] = 3
for pt in ['f','h','v','w','y']:
    di[pt] = 4
di['k'] = 5
di['j'] = 8
di['x'] = 8
di['q'] = 10
di['z'] = 10

#%% MAIN EVOLVE FUNCTION
def evolve(gens, popsize, truncate = 0.5, mutrate = 0.1, feed = []):
    '''main evolve function given truncation and mutation rates
    feed: populaiton as returned from evolve to continue more generations
    '''
    if len(feed) == 0: ## generate new sample
        pop = [get_sample() for _ in range(popsize)]
        fitness = [getScore(indiv) for indiv in pop]
        gen_pop = sorted(zip(pop, fitness), key = lambda ind_fit: ind_fit[1])
    
    elif len(feed) != 0: ## use population given
        gen_pop = feed
        
    for gen in range(gens):
        print("Generation: ", gen)
        
        parents_index = chooseParents(gen_pop, popsize*truncate)
        
        children = children_crossover_call(gen_pop, parents_index)
        
        m_children = [mutate2(indiv,score, mutrate) for indiv,score in children]
        
        print('Total children: ' + str(len(m_children)))
        
        sz_pop = getFinalpop(sorted(gen_pop+m_children, key = lambda ind_fit: ind_fit[1]), 
                             popsize, numchildren = popsize*0.5, childrenall = [c for c, f in m_children])
      
        fitness = [s for _,s in sz_pop ]

        print (max(fitness), np.mean(fitness), fitness)
        print('\n')
        
        if np.var(fitness)<7 or len(set(fitness)) <=2 :
            print('TERMINATE')
            break

    return sz_pop



#%%FUNCTIONS USED IN INVOLVE FUNCTION


def chooseParents(sz_pop, num_offsprings):
    '''choose best parents to generate each offspring'''
    worst = sz_pop[0][1]
    #print(worst)
    probs = []
    arrs = []
    parents_index = []

    for a , p in sz_pop:
        probs.append(p - worst)
        arrs.append(a)

    sump = np.sum(probs)
    probs = [p/sump for p in probs]

    while len(parents_index) < num_offsprings:
        newchoice = list(np.random.choice(list(range(0,len(arrs))), 2, replace = False, p = probs))
        if newchoice not in parents_index:
            parents_index.append(newchoice)

    return parents_index

def children_crossover_call(sz_pop, parents_index, show = False):
    '''calls crossover function on population, given index of parents chosen'''
    children = []
    for pp in parents_index:
        if show:
            print(pp, (sz_pop[pp[0]][1] , sz_pop[pp[1]][1]), end = ' ')
        
        children.append(new_crossover22(sz_pop[pp[0]][0] , sz_pop[pp[1]][0]))
    return children

def new_crossover22(id1, id2, show = False):
    '''new and improved crossover function
    used in children_cross_over_call, uses find_RandomChild to separate strings
    uses smart_remove to replace letters
    returns best children possible'''

    
    other = id2[:]
    blank_pos = [pos for pos, char in enumerate(id1) if char.islower()]
    
    child = find_RandomChild(id1, blank_pos)
    
    
    for f in child:
        smart_remove(other, f)
    
    oc = getScore(other+child)
    co = getScore(child+other)
    #print(len(child), end = '!')
    if oc>co:
        #print (oc)
        return (other+child, oc)
    #print (co)
    return (child+other, co)


def find_RandomChild(id1, blank_pos):
    '''find position to break string for crossover
    used in new_crossover22()'''
    rgen = uniform(0,1)

    if rgen>1/3 and len(id1[0:blank_pos[0]]) > 2: 
        return id1[0:blank_pos[0]]
    
    elif rgen>2/3 and len(id1[blank_pos[1]+1: 100]) > 2:
        return id1[blank_pos[1]+1: 100]
    else:
        return id1[blank_pos[0]+1: blank_pos[1]]
   

def smart_remove(other, f):
    '''removes letter for new_crossover22()
    using letter which contributes lowest score to string'''
    c1 = len(other)
    secscore = findLetterScore(other)
    letter = secscore.loc[f]

    if isinstance(letter, (int, np.integer)):
        other.remove(f)
    else:
        letter = list(letter.values)
        minletter = letter.index(min(letter))
        removeNthletter(other, f, minletter)
        
    assert c1 > len(other)
    return other

def findLetterScore(this_word):
    '''find letter contribution to string, used in smart_remove()'''
    wordf = pd.Series([0]*len(this_word),index = this_word)
    tcaps = ''.join([x.upper() for x in this_word])

    for word in dfwords.index:
        if word in tcaps:
            indices = find_substring(word,tcaps)
            for i in indices:
                for w in range(len(word)):
                    wordf.iloc[i+w] += 1
    return wordf

def removeNthletter(word, letter, N): 
    '''removes Nth letter in a given word, used in smart_remove()'''
    count = 0
      
    for i in range(0, len(word)): 
        if (word[i] == letter): 
            if(count == N): 
                del(word[i]) 
                return True
            
            count = count + 1
                  
    raise Exception




def mutate2(indiv, score, rate, show = False):
    '''new mutation function for individual characters'''

    lowers = [pos for pos, char in enumerate(indiv) if char.islower()]
    if show: print (lowers, end = '')
    
    if abs(lowers[0] - lowers[1]) > 2:
        if show: print ('Skipped', end = '  ')
        return(indiv, score)
    
    pos1 = randint(0,99)
    pos2 = randint(0,99)
    
    copy = indiv[:]
    copy[lowers[0]], copy[pos1] = copy[pos1], copy[pos1].lower()
    copy[lowers[1]], copy[pos2] = copy[pos2], copy[pos2].lower()

    newscore = getScore(copy)
    
    if show: print((score, newscore))
    if newscore<score and score > 600:
        print ('Woops')
        print(''.join(indiv))
        print (''.join(copy))
    
    return (copy, newscore)


def getFinalpop(sz_pop, popsize, numchildren, childrenall):
    '''returns final population as well as survived stats'''
    survivedchildren = 0
    finalpop = []
    finalstrs = []
    
    for i in range(len(sz_pop)):
        
        indiv, score = sz_pop.pop(-1)
        if indiv in finalstrs:
            print('C', i, end = '  ')
            continue
        else:
            finalstrs.append(indiv)
            finalpop.append((indiv, score))
            
            if indiv in childrenall:
                survivedchildren += 1
            
            if len(finalpop) >= popsize:
                print('Survived: ' + str(survivedchildren))
                return finalpop[::-1]

    if len(finalpop) < popsize:
        print(finalpop)
        return -1


def get_sample():
    '''returns random sample of tiles'''
    tiles = ['E']*12 + ['I','A']*9 + ['O']*8 + ['N','R','T']*6 + ['L','S','U','D']*4 + ['G']*3 + ['B','C','M','P','F','H','V','W','Y']*2 + ['K','J','X','Q','Z'] + [choice(string.ascii_lowercase), choice(string.ascii_lowercase)]
    return sample(tiles, len(tiles))


def getScore(indiv , dfwords = dfwords, show = False):
    '''returns score given a word, and dfwords
    indiv is a list, with 98 upper case and 2 lower case characters'''
    indiv = ''.join(indiv)
    indivCAPS = ''.join([x.upper() for x in indiv])
    
    totscore = 0
    for word in dfwords.index:
        if word in indivCAPS:
            indices = find_substring(word,indivCAPS)
            totscore += len(indices)*dfwords.loc[word].SCORE
            lowers = []
            for i in indices: 
                    emptylets = [c for c in indiv[i:i+len(word)] if c.islower()]
                    if len(emptylets) == 0:
                        continue 
                    for e in emptylets:
                        lowers.append(e)
                        totscore -= di[e]
            
            if show:
                print(word, len(indices), dfwords.loc[word].SCORE, lowers, end = '   ')
                print([indiv[i:i+len(word)] for i in indices])        
    return totscore


def find_substring(substring, string):
    '''used in getScore
    Returns list of indices where substring begins in given string'''
    indices = []
    index = -1  # Begin at -1 so index + 1 is 0
    while True:
        # Find next index of substring, by starting search from index + 1
        index = string.find(substring, index + 1)
        if index == -1:  
            break 
        indices.append(index)
    return indices

#%% Helper functions
def wordpossible(word):
    wlist = [char for char in word]
    tiles = ['E']*12 + ['I','A']*9 + ['O']*8 + ['N','R','T']*6 + ['L','S','U','D']*4 + ['G']*3 + ['B','C','M','P','F','H','V','W','Y']*2 + ['K','J','X','Q','Z']
    blankchar = 0 
    for w in wlist:
        try:
            tiles.remove(w)
        except ValueError: 
            blankchar += 1
            if blankchar > 2:
                print(word)
                return False
    return True

def checkValid(final, tiles):
    assert len(final) == 100
    if sorted(final)[:-2] == sorted(tiles[:-2]):
        return True
    else:
        return False



#%%

ans = evolve(gens = 20, popsize = 50, truncate = 0.5)
