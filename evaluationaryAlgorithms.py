
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
import copy 

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

radius_limit  = 100

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
            self.chromosome.append(Gene(random.randint(1, radius_limit), 
                                        random.randint(0, image_width), 
                                        random.randint(0, image_height), 
                                        random.randint(0, 255), 
                                        random.randint(0, 255), 
                                        random.randint(0, 255), 
                                        random.random()))
        assert len(self.chromosome) == num_genes

    def showImage(self):
        generated_image = np.zeros((image_height, image_width, 3), np.uint8)
        for gene in self.chromosome:
            generated_image_copy = generated_image.copy()
            cv2.circle(generated_image_copy, (gene.X_COORD, gene.Y_COORD), gene.RADIUS, (gene.RED, gene.GREEN, gene.BLUE), -1)
            cv2.addWeighted(generated_image, gene.ALPHA, generated_image_copy, 1 - gene.ALPHA, 0, generated_image)
        cv2.imshow('Generated Image', generated_image)
        cv2.imshow('Original Image', image)
        cv2.waitKey(1)
    
    def mutateInd(self):
        
        if random.random() < mutation_prob:
            geneIndex = random.randint(0, self.num_genes - 1)
            if mutation_type == 0:
                match geneIndex:
                    case 0:
                        self.chromosome[geneIndex].RADIUS = random.randint(1, radius_limit)
                    case 1:
                        self.chromosome[geneIndex].X_COORD = random.randint(0, image_width)
                    case 2:
                        self.chromosome[geneIndex].Y_COORD = random.randint(0, image_height)
                    case 3:
                        self.chromosome[geneIndex].RED = random.randint(0, 255)
                    case 4:
                        self.chromosome[geneIndex].GREEN = random.randint(0, 255)
                    case 5:
                        self.chromosome[geneIndex].BLUE = random.randint(0, 255)
                    case 6:
                        self.chromosome[geneIndex].ALPHA = random.random()

            elif mutation_type == 1:
                    
                match random.randint(0, 6):
                    case 0:
                        # radius
                        while True:
                            new_radius = random.randint(1, radius_limit)
                            if new_radius < self.chromosome[geneIndex].RADIUS + 10 and  new_radius > self.chromosome[geneIndex].RADIUS - 10:
                                self.chromosome[geneIndex].RADIUS = new_radius
                                break
                            else :
                                continue

                    case 1:            
                        # x-coordinate
                        while True:
                            new_x_coord = random.randint(0, image_width)
                            if new_x_coord < self.chromosome[geneIndex].X_COORD + image_width / 4 and new_x_coord > self.chromosome[geneIndex].X_COORD - image_width / 4:
                                self.chromosome[geneIndex].X_COORD = new_x_coord
                                break
                            else:
                                continue
                    case 2:                 
                        # y-coordinate
                        while True:
                            new_y_coord = random.randint(0, image_height)
                            if new_y_coord < self.chromosome[geneIndex].Y_COORD + image_height / 4 and new_y_coord > self.chromosome[geneIndex].Y_COORD - image_height / 4:
                                self.chromosome[geneIndex].Y_COORD = new_y_coord
                                break
                            else:
                                continue
                    case 3:
                        # red value
                        while True:
                            new_red = random.randint(0, 255)
                            if new_red < self.chromosome[geneIndex].RED + 64 and new_red > self.chromosome[geneIndex].RED - 64:
                                self.chromosome[geneIndex].RED = new_red
                                break
                            else:
                                continue
                    case 4:
                        # green value
                        while True:
                            new_green = random.randint(0, 255)
                            if new_green < self.chromosome[geneIndex].GREEN + 64 and new_green > self.chromosome[geneIndex].GREEN - 64:
                                self.chromosome[geneIndex].GREEN = new_green
                                break
                            else:
                                continue     
                    case 5:
                        # blue value
                        while True:
                            new_blue = random.randint(0, 255)
                            if new_blue < self.chromosome[geneIndex].BLUE + 64 and new_blue > self.chromosome[geneIndex].BLUE - 64:
                                self.chromosome[geneIndex].BLUE = new_blue
                                break
                            else:
                                continue
                    case 6: 
                        # alpha value 
                        while True:
                            new_alpha = random.random()
                            if new_alpha < self.chromosome[geneIndex].ALPHA + 0.25 and new_alpha > self.chromosome[geneIndex].ALPHA - 0.25:
                                self.chromosome[geneIndex].ALPHA = new_alpha
                                break
                            else:
                                continue            
            else:
                print("Invalid mutation type")
                return
        self.chromosome.sort(key = lambda x: x.RADIUS, reverse = True)
        assert len(self.chromosome) == num_genes

class Population:
    def __init__(self, num_inds, num_genes):
        self.population = []
        for i in range(num_inds):
            self.population.append(Individual(num_genes))
    
    def sort(self):
        size_at_start = len(self.population)
        fitnesses = evaluate_individuals(self.population)
        # pair each individual with its fitness
        individual_with_fitness = list(zip(self.population, fitnesses))
        # sort the list of individuals based on their fitness
        individual_with_fitness.sort(key = lambda x: x[1], reverse = True)
        # replace the population with the sorted individuals
        self.population = [x[0] for x in individual_with_fitness]
        size_at_end = len(self.population)
        # make sure the size of the population is the same after multiprocessing evaluations
        assert size_at_start == size_at_end

    def getFittest(self):
        if len(self.population) == 0:
            return None
        else:
            return self.population[0]
    
    def getNumOfIndividuals(self):
        return len(self.population)
    
    def crossover(self):
        numOfParents = int(frac_parents * num_inds)
        numOfElite = int(frac_elite * num_inds)

        for i in range(numOfParents // 2):
            parent1, parent2 = random.sample(self.population[numOfElite:numOfElite + numOfParents], 2)
            
            self.population.remove(parent1)
            self.population.remove(parent2)
            child1, child2 = crossover(parent1, parent2)
        
            self.population.append(child1)
            self.population.append(child2)

            assert len(self.population) == num_inds

    def mutate(self):
        numOfElite = int(frac_elite * num_inds)
        for i in range(len(self.population) - numOfElite):
            self.population[numOfElite + i].mutateInd()

    def select_individuals(self, frac_elite, tm_size):
        elites = self.population[:int(frac_elite * num_inds)]
        non_elites = self.population[int(frac_elite * num_inds):]

        assert len(elites) + len(non_elites) == len(self.population)

        self.population = elites

        while population.getNumOfIndividuals() < num_inds:
            tournament = []
            for i in range(tm_size): 
                tournament.append(random.choice(non_elites))
            self.population.append(tournament_selection(tournament))



def evaluate_individual(individual):
    generated_image = np.zeros((image_height, image_width, 3), np.uint8)
    for gene in individual.chromosome:
        generated_image_copy = generated_image.copy()
        cv2.circle(generated_image_copy, (gene.X_COORD, gene.Y_COORD), gene.RADIUS, (gene.RED, gene.GREEN, gene.BLUE), -1)
        cv2.addWeighted(generated_image, gene.ALPHA, generated_image_copy, 1 - gene.ALPHA, 0, generated_image)
    
    error_of_individual = np.zeros((image_height, image_width, 3), np.int16)
    error_of_individual = np.subtract(image, generated_image)
    np.abs(error_of_individual, out=error_of_individual)
        
    return np.sum(error_of_individual) * -1


from multiprocessing import Pool
def evaluate_individuals(population):
    with Pool() as pool:
        fitnesses = pool.map(evaluate_individual, population)
    return fitnesses



def tournament_selection(tournament):
    tournament.sort(key = lambda x: evaluate_individual(x), reverse = True)
    return tournament[0]

      

    

def crossover(individual1, individual2):
    parent1 = copy.deepcopy(individual1)
    parent2 = copy.deepcopy(individual2)

    child1 = Individual(num_genes)
    child2 = Individual(num_genes)
    for i in range(num_genes):
        if random.random() < 0.5:
            child1.chromosome[i] = parent1.chromosome[i]
        else:
            child1.chromosome[i] = parent2.chromosome[i]
        if random.random() < 0.5:
            child2.chromosome[i] = parent1.chromosome[i]
        else :
            child2.chromosome[i] = parent2.chromosome[i]
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
    
    population.select_individuals(frac_elite, tm_size)

    population.crossover()

    population.mutate() # suspected to be the problem


    fittest_individual = population.getFittest()
    fittest_individual.showImage()
    print("Generation: ", i , " Fittest individual fitness: ", evaluate_individual(population.getFittest()))
    # print("*")
    


    

    

    





