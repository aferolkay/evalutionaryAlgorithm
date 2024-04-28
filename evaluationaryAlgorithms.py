
# Each individual has one chromosome.
# Each gene in a chromosome represents one circle to be drawn.
# Each gene has at least 7 values:
#    1. x-coordinate of the center of the circle
#    2. y-coordinate of the center of the circle
#    3. radius of the circle
#    4. red value of the circle
#    5. green value of the circle
#    6. blue value of the circle
#    7. alpha value of the circle

import random
import numpy as np
import cv2 as cv2
import threading
import time

num_generations = 10000 # Number of generations
num_inds       = 20  # Number of individuals in the population
num_genes     = 50   # Number of genes in each individual
tm_size       = 5    # Tournament size
frac_elite    = 0.2  # Fraction of elite individuals
frac_parents  = 0.6  # Fraction of parents
mutation_prob = 0.2  # Probability of mutation
mutation_type = 1    # 0: Random mutation, 1: guided 
image         = cv2.imread('painting.png')
image_height  = image.shape[0]
image_width   = image.shape[1]


generated_image = np.zeros((image_height, image_width, 3), np.uint8)

class Gene:
    def __init__(self, RADIUS, X_COORD, Y_COORD, RED, GREEN, BLUE, ALPHA):
        self.RADIUS = RADIUS
        self.X_COORD = X_COORD
        self.Y_COORD = Y_COORD
        self.RED = RED
        self.GREEN = GREEN
        self.BLUE = BLUE
        self.ALPHA = ALPHA

class Individual:
    def __init__(self, num_genes):
        self.num_genes = num_genes
        self.chromosome = []
        for i in range(num_genes):
            self.chromosome.append(Gene(random.randint(1, 50), 
                                        random.randint(0, image_width), 
                                        random.randint(0, image_height), 
                                        random.randint(0, 255), 
                                        random.randint(0, 255), 
                                        random.randint(0, 255), 
                                        random.random()))
    def showImage(self):
        generated_image = np.zeros((image_height, image_width, 3), np.uint8)
        for gene in self.chromosome:
            generated_image_copy = generated_image.copy()
            cv2.circle(generated_image_copy, (gene.X_COORD, gene.Y_COORD), gene.RADIUS, (gene.RED, gene.GREEN, gene.BLUE), -1)
            cv2.addWeighted(generated_image, gene.ALPHA, generated_image_copy, 1 - gene.ALPHA, 0, generated_image)
        cv2.imshow('Generated Image', generated_image)
        cv2.imshow('Original Image', image)
        cv2.waitKey(1)
    
    def mutate(self):
        for geneIndex in range(len(self.chromosome)):
            if random.random() < mutation_prob:
                if mutation_type == 0:
                    self.chromosome[geneIndex] = Gene(random.randint(1, 50), 
                                                        random.randint(0, image_width), 
                                                        random.randint(0, image_height), 
                                                        random.randint(0, 255), 
                                                        random.randint(0, 255), 
                                                        random.randint(0, 255), 
                                                        random.random())
                elif mutation_type == 1:
                    # radius
                        if (self.chromosome[geneIndex].RADIUS - 10) > 0 :
                            lower_bound = self.chromosome[geneIndex].RADIUS - 10
                        else:
                            lower_bound = 1
                        self.chromosome[geneIndex].RADIUS = random.randint(lower_bound, 50)


                        # x-coordinate
                        if (self.chromosome[geneIndex].X_COORD - image_width/4) > 0 :
                            lower_bound = self.chromosome[geneIndex].X_COORD - image_width/4
                        else:
                            lower_bound = 0
                        if (self.chromosome[geneIndex].X_COORD + image_width/4) < image_width :
                            upper_bound = self.chromosome[geneIndex].X_COORD + image_width/4
                        else:
                            upper_bound = image_width
                        self.chromosome[geneIndex].X_COORD = random.randint(lower_bound, upper_bound)

                        # y-coordinate
                        if (self.chromosome[geneIndex].Y_COORD - image_height/4) > 0 :
                            lower_bound = self.chromosome[geneIndex].Y_COORD - image_height/4
                        else:    
                            lower_bound = 0
                        if (self.chromosome[geneIndex].Y_COORD + image_height/4) < image_height :
                            upper_bound = self.chromosome[geneIndex].Y_COORD + image_height/4
                        else:
                            upper_bound = image_height
                        self.chromosome[geneIndex].Y_COORD = random.randint(lower_bound, upper_bound)

                        # red value
                        if (self.chromosome[geneIndex].RED - 64) > 0 :
                            lower_bound = self.chromosome[geneIndex].RED - 64
                        else:
                            lower_bound = 0
                        if (self.chromosome[geneIndex].RED + 64) < 255 :
                            upper_bound = self.chromosome[geneIndex].RED + 64
                        else:
                            upper_bound = 255
                        self.chromosome[geneIndex].RED = random.randint(lower_bound, upper_bound)

                        # green value
                        if (self.chromosome[geneIndex].GREEN - 64) > 0 :
                            lower_bound = self.chromosome[geneIndex].GREEN - 64
                        else:
                            lower_bound = 0
                        if (self.chromosome[geneIndex].GREEN + 64) < 255 :
                            upper_bound = self.chromosome[geneIndex].GREEN + 64
                        else:
                            upper_bound = 255
                        self.chromosome[geneIndex].GREEN = random.randint(lower_bound, upper_bound)

                        # blue value
                        if (self.chromosome[geneIndex].BLUE - 64) > 0 :
                            lower_bound = self.chromosome[geneIndex].BLUE - 64
                        else:
                            lower_bound = 0
                        if (self.chromosome[geneIndex].BLUE + 64) < 255 :
                            upper_bound = self.chromosome[geneIndex].BLUE + 64
                        else:
                            upper_bound = 255
                        self.chromosome[geneIndex].BLUE = random.randint(lower_bound, upper_bound)

                        # alpha value 
                        if (self.chromosome[geneIndex].ALPHA - 0.25) > 0 :
                            lower_bound = self.chromosome[geneIndex].ALPHA - 0.25
                        else:
                            lower_bound = 0
                        if (self.chromosome[geneIndex].ALPHA + 0.25) < 1 :
                            upper_bound = self.chromosome[geneIndex].ALPHA + 0.25
                        else:
                            upper_bound = 1
                        self.chromosome[geneIndex].ALPHA = random.uniform(lower_bound, upper_bound)

            else:
                print("Invalid mutation type")
                return
        self.chromosome.sort(key = lambda x: x.RADIUS, reverse = True)

class Population:
    def __init__(self, num_inds, num_genes):
        self.population = []
        for i in range(num_inds):
            self.population.append(Individual(num_genes))
    
    def sort(self):
        self.population.sort(key=evaluate_individual, reverse=True)
    
    def getFittest(self):
        if len(self.population) == 0:
            return None
        else:
            return self.population[0]
    
    def getNumOfIndividuals(self):
        return len(self.population)
    
    def crossover(self):
        children = []
        numOfParents = int(frac_parents * num_inds)
        numOfElite = int(frac_elite * num_inds)
        parents = self.population[numOfElite:numOfElite + numOfParents]
        for i in range(numOfParents // 2):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child1, child2 = crossover(parent1, parent2)
            children.append(child1)
            children.append(child2)
        self.population = self.population + children

    def mutate(self):
        numOfElite = int(frac_elite * num_inds)
        for i in range(len(self.population) - numOfElite):
            self.population[numOfElite + i].mutate()



def evaluate_individual(individual):
    generated_image = np.zeros((image_height, image_width, 3), np.uint8)
    for gene in individual.chromosome:
        generated_image_copy = generated_image.copy()
        cv2.circle(generated_image_copy, (gene.X_COORD, gene.Y_COORD), gene.RADIUS, (gene.RED, gene.GREEN, gene.BLUE), -1)
        cv2.addWeighted(generated_image, gene.ALPHA, generated_image_copy, 1 - gene.ALPHA, 0, generated_image)
    
    
    error_of_individual = np.subtract(image, generated_image)
    np.square(error_of_individual, out=error_of_individual)
        
    return np.sum(error_of_individual) * -1
        


def tournament_selection(tournament):
    tournament.sort(key = lambda x: evaluate_individual(x), reverse = True)
    return tournament[0]

def select_individuals(population, frac_elite, tm_size):
    elites = population.population[:int(frac_elite * num_inds)]
    non_elites = population.population[int(frac_elite * num_inds):]
    population.population = elites
    
    while population.getNumOfIndividuals() < num_inds:
        tournament = []
        for i in range(tm_size): 
            tournament.append(random.choice(non_elites))
        population.population.append(tournament_selection(tournament))
    

def crossover(individual1, individual2):
    child1 = Individual(num_genes)
    child2 = Individual(num_genes)
    for i in range(num_genes):
        if random.random() < 0.5:
            child1.chromosome[i] = individual1.chromosome[i]
        else:
            child1.chromosome[i] = individual2.chromosome[i]
        if random.random() < 0.5:
            child2.chromosome[i] = individual1.chromosome[i]
        else :
            child2.chromosome[i] = individual2.chromosome[i]
    return child1, child2

def show_state(population):
    iter = 0
    for individual in population.population:
        
        print(iter, "nd fitness: ", evaluate_individual(individual))
        iter += 1
    print("             ***             ")

                  

# Initialize population with <num_inds> individuals each having <num_genes> genes
# While not all generations (<num_generations>) are computed:
#     Evaluate all the individuals
#     Select  individuals +++++
#     Do crossover on some individuals
#     Mutate some individuals

population = Population(num_inds, num_genes)

for i in range(num_generations):
    population.sort()
    select_individuals(population, frac_elite, tm_size)
    population.crossover()
    population.mutate()
    fittest_individual = population.getFittest()
    fittest_individual.showImage()
    print("Generation: ", i , " Fittest individual fitness: ", evaluate_individual(fittest_individual))


    

    

    





