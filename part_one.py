import numpy as np
from random import uniform
from random import random
import copy
import math

class Particle:
    def __init__(self, length, min_velocity, max_velocity):
        self.values = [uniform(-30, 30) for x in range(length)]
        self.fitness = -1
        self.pbest_fitness = -1
        self.pbest_values = []
        self.velocities = [uniform(min_velocity, max_velocity) for x in range(length)]

def update_fitnesses(population, gbest, function):
    for particle in population:
        particle.fitness = rosenbrock_formula(particle.values) if function == "Rosenbrock" else griewank_formula(particle.values)
        if particle.fitness < particle.pbest_fitness:
            particle.pbest_fitness = particle.fitness
            particle.pbest_values = particle.values
        if particle.pbest_fitness < gbest.pbest_fitness:
            gbest = copy.deepcopy(particle)
    return population, gbest

def rosenbrock_formula(values):
    summation = 0
    for x in range(len(values) - 1):
        summation += 100 * ((values[x] ** 2) - (values[x + 1])) ** 2 + (values[x] - 1) ** 2
    return summation

# Used code for the expression from here:
# https://github.com/illyuha/Griewank/blob/master/lib.py
def griewank_formula(values):
    sum = 0
    for x in values:
        sum += x * x
    product = 1
    for i in range(len(values)):
        product *= math.cos(values[i] / math.sqrt(i + 1))
    return 1 + sum / 4000 - product

def pso(d, c1, c2, w, min_velocity, max_velocity, generations, population_size, function):
    # Repeat PSO 30 times and record values
    values = []
    fitnesses = []
    for repeat in range(30):
        # Create population
        population = []
        for x in range(population_size):
            population.append(Particle(d, min_velocity, max_velocity))
        # Check fitness values
        for particle in population:
            particle.fitness = rosenbrock_formula(particle.values) if function == "Rosenbrock" else griewank_formula(particle.values)
            particle.pbest_fitness = particle.fitness
            particle.pbest_values = particle.values
        # Set baseline gbest
        gbest = population[0]
        generation = 0
        while generation < generations:
            # Update fitness values and pbest
            population, gbest = update_fitnesses(population, gbest, function)
            # Update velocities and positions
            for particle in population:
                r1 = random()
                r2 = random()
                # Update velocity with clamping
                for x in range(len(particle.velocities)):
                    new_velocity = w * particle.velocities[x] + (c1 * r1 * (particle.pbest_values[x] - particle.values[x])) + (c2 * r2 * (gbest.pbest_values[x] - particle.values[x]))
                    particle.velocities[x] = min(new_velocity, max_velocity) if new_velocity >= 0 else max(new_velocity, min_velocity)
                # Update position, limiting to [-30, 30]
                for x in range(len(particle.values)):
                    new_value = particle.values[x] + particle.velocities[x]
                    particle.values[x] = min(new_value, 30) if new_value >= 0 else max(new_value, -30)
            #print("Gen:", generation, " | Global best:", gbest.pbest_fitness)
            generation += 1
        # Record this run's best values
        print("Completed repeat", repeat)
        population, gbest = update_fitnesses(population, gbest, function)
        values.append(gbest.pbest_values)
        fitnesses.append(gbest.pbest_fitness)
    # Fitness mean, std, and values
    print("\n", function, "\n")
    for count, fitness in enumerate(fitnesses):
        print("Repeat", count + 1, " | Result:", fitness)
    print("\nResults |", "Mean:", np.mean(fitnesses), "   STD:", np.std(fitnesses), "\n")
    # Calculate means and standard deviations for each position value
    for col in range(len(values[0])):
        xi_values = []
        for row in range(len(values)):
            xi_values.append(values[row][col])
        print("Value", col + 1, "| Mean:", np.mean(xi_values), "   STD:", np.std(xi_values))
    print("")

if __name__ == "__main__":
    #pso(50, 1.49618, 1.49618, 0.7298, -10, 10, 200, 10000, "Rosenbrock")
    pso(50, 1.49618, 1.49618, 0.7298, -10, 10, 200, 10000, "Griewank")