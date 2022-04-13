import operator
import math
from deap import gp, creator, base, tools, algorithms
from numpy import exp
import random

# Read Part 1 and Part 2 of the basic tutorials and also the Genetic Programming advanced tutorials from the resource provided in the handout
# https://deap.readthedocs.io/en/master/

# Used code from the genetic programming example from the same resource
# https://deap.readthedocs.io/en/master/examples/gp_symbreg.html

# Specially designed division to avoid 0 error
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def square(input):
    return input**2

def is_positive(input):
    return True if input > 0 else False

def if_then_else(input, output1, output2):
    return output1 if input else output2

# Creating the function set for tree nodes
pset = gp.PrimitiveSetTyped("MAIN", [float], float)
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protected_div, [float, float], float) # AVOIDS 0 ERROR
pset.addPrimitive(math.sin, [float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.tan, [float], float)
pset.addPrimitive(square, [float], float)

# Creating the terminal set for tree nodes
for x in range(20):
    pset.addTerminal(x + 1, float)
pset.addTerminal(1, bool)
pset.addTerminal(0, bool)

# Create the types for this GP problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Negative weights because this is a minimisation problem
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Create tools for this evolution process
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Find what the expected result using the target formula should be for a given input
def calculate_real_value(input):
    if input > 0:
        return (1 / input) + math.sin(input)
    else:
        return 2 * input + math.pow(input, 2) + 3.0

# Create the fitness function that individuals are evaluated against
def fitness_function(individuals, fitness_cases):
    # Transform the Individual's tree into a callable expression
    functionA = toolbox.compile(expr=individuals[0])
    functionB = toolbox.compile(expr=individuals[1])
    # Evaluate the MSE between the expression and real function value
    square_errors = []
    for case in fitness_cases:
        expression_value = functionA(case) if case > 0 else functionB(case)
        real_value = calculate_real_value(case)
        square_errors.append((expression_value - real_value)**2)
    return sum(square_errors)/len(square_errors),

# Used this forum thread to help create my selection method implementing tournament selection and also elitism
# https://groups.google.com/g/deap-users/c/iannnLI2ncE
def select_tournament_elitism(individuals, pop_size):
    return tools.selBest(individuals=individuals, k=int(0.1*pop_size)) + tools.selTournament(individuals=individuals, k=pop_size - int(0.1*pop_size), tournsize=7)

# Add more tools for the GP to use in the evolution process#
toolbox.register("evaluate", fitness_function, fitness_cases=[x for x in range(-100, 100)])
toolbox.register("select", select_tournament_elitism)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("species", tools.initRepeat, list, toolbox.individual, 3000)
toolbox.register("get_best", tools.selBest, k=1)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
species = [toolbox.species() for _ in range(2)]
representatives = [random.choice(species[i]) for i in range(2)]

# Begin the evolution process
# Used code from - https://deap.readthedocs.io/en/master/examples/coev_coop.html
generation = 0
generations = 100
best_ind = species[0]
while generation < generations:
    # Initialize a container for the next generation representatives
    next_repr = [None] * len(species)
    for i, s in enumerate(species):
        # Vary the species individuals
        s = algorithms.varAnd(s, toolbox, 0.9, 0.1)

        # Get the representatives excluding the current species
        r = representatives[:i] + representatives[i+1:]
        for ind in s:
            # Evaluate and set the individual fitness
            ind.fitness.values = toolbox.evaluate([ind] + r) if i == 0 else toolbox.evaluate(r + [ind])

        # Select the individuals
        species[i] = toolbox.select(s, len(s))  # Tournament selection
        next_repr[i] = toolbox.get_best(s)[0]   # Best selection
    representatives = next_repr
    print("Completed generation", generation)
    generation += 1

def print_results(function_a, function_b):
    for x in range(-50, 50):
        if x > 0:
            print(x * 2,") Expected:", calculate_real_value(x * 2), "     GP Result:", function_a(x * 2))
        else:
            print(x * 2,") Expected:", calculate_real_value(x * 2), "     GP Result:", function_b(x * 2))

best_function_A = toolbox.compile(expr=representatives[0])
best_function_B = toolbox.compile(expr=representatives[1])
treeA = gp.PrimitiveTree(representatives[0])
treeB = gp.PrimitiveTree(representatives[1])

print("Tree A - X > 0")
print(str(treeA))
print("Tree B - X < 0")
print(str(treeB))
print_results(best_function_A, best_function_B)
print("Done!")