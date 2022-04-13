import numpy as np
from numpy.random import rand
import pandas as pd
import matplotlib.pyplot as plt
import random
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.factory import get_performance_indicator
from sklearn.neighbors import KNeighborsClassifier

# I used the following guides from Pymoo's website to create my NSGA-II program.
# https://pymoo.org/algorithms/moo/nsga2.html (Information relating to this particular task)
# https://pymoo.org/getting_started/part_2.html# (This walked me through and explained the main parts of the library)
# https://pymoo.org/customization/custom.html (This explained how to use custom variables with the library)
# The individual representation is a bit string, which needs GA functionality to be specifically defined.
class MyProblem(ElementwiseProblem):

    def __init__(self, n_features, instances, labels):
        super().__init__(n_var=1, n_obj=2, n_constr=0)
        self.n_features = n_features
        self.instances = instances
        self.labels = labels
        self.VALUES = ['0', '1']
        
    def _evaluate(self, x, out, *args, **kwargs):
        # Determine fitness value from error rate and selected feature ratio
        f1 = calc_individual_error_rate(x, self.instances, self.labels)
        f2 = calc_num_selected_features(x) / self.n_features
        out["F"] = [f1, f2]

class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), None, dtype=object)

        for i in range(n_samples):
            X[i, 0] = "".join([np.random.choice(problem.VALUES) for _ in range(problem.n_features)])

        return X

class MyCrossover(Crossover):

    def __init__(self):
        # Define number of parents and offspring
        super().__init__(2, 2)
    
    def _do(self, problem, X, **kwargs):
        # Input:
        _, n_matings, n_var = X.shape
        # Output:
        Y = np.full_like(X, None, dtype=object)
        # Perform crossover
        for k in range(n_matings):
            a, b = X[0, k, 0], X[1, k, 0]
            # 1-point crossover
            index = random.randrange(1, len(a))
            offspring_a_string = a[:index] + b[index:]
            offspring_b_string = b[:index] + a[index:]
            # Ensure that offsprings have at least one feature selected
            if "1" not in offspring_a_string:
                offspring_a_string = "".join([np.random.choice(problem.VALUES) for _ in range(problem.n_features)])
            if "1" not in offspring_b_string:
                offspring_b_string = "".join([np.random.choice(problem.VALUES) for _ in range(problem.n_features)])
            # Output
            Y[0, k, 0], Y[1, k, 0] = offspring_a_string, offspring_b_string
        return Y

class MyMutation(Mutation):

    def __init__(self):
        super().__init__()
    
    def _do(self, problem, X, **kwargs):
        # Attempt mutation on every individual
        for x in range(len(X)):
            # Flip a bit with a probability of 20%
            if random.random() < 0.2:
                bit_string = list(X[x, 0])
                index = random.randrange(len(X[x, 0]))
                if bit_string[index] == "1":
                    bit_string[index] = "0"
                else:
                    bit_string[index] = "1"
                # Ensure that offsprings have at least one feature selected
                if "1" not in bit_string:
                    index = random.randrange(0, problem.n_features)
                    bit_string[index] = "1"
                X[x, 0] = "".join(bit_string)
        return X

class MyDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return a.X[0] == b.X[0]

def calc_individual_error_rate(individual, instances, labels):
    transformed_instances = []
    for instance_index in range(len(instances.index)):
        transformed_instance = []
        bit_string = list(individual[0])
        for bit_index in range(len(bit_string)):
            if bit_string[bit_index] == '1':
                transformed_instance.append(instances.iloc[instance_index][bit_index])
        transformed_instances.append(transformed_instance)
    # Score it - minimise CA error
    clf = KNeighborsClassifier()
    clf.fit(transformed_instances, labels)
    score = clf.score(transformed_instances, labels)
    f1 = 1 - score
    return f1

def calc_num_selected_features(individual):
    selected_features = 0
    for feature in individual[0]:
        if feature == "1":
            selected_features += 1
    return selected_features

def nsgaii(instances, labels):
    instances = instances.copy()
    labels = labels.copy()
    algorithm = NSGA2(
        pop_size=100,
        sampling=MySampling(),
        crossover=MyCrossover(),
        mutation=MyMutation(),
        eliminate_duplicates=MyDuplicateElimination()
    )
    termination = get_termination("n_gen", 30)
    res = minimize(MyProblem(len(instances.columns), instances, labels),
                algorithm,
                termination,
                verbose=False)
    X = res.X   
    F = res.F

    return X, F

def main(file):
    # Split data into instances and labels
    data = pd.read_csv(file, header=None)
    instances = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    # Find default error rate using all of the features
    clf = KNeighborsClassifier()
    clf.fit(instances, labels)
    error_rate = 1 - clf.score(instances, labels)
    print("Error rate with all features selected:", error_rate, "\n")
    for run in range(3):
        print("File:", file, "  Run:", run + 1)
        # Get solution set
        X, F = nsgaii(instances, labels)
        results = X[np.argsort(F[:, 0])]

        # Find error rate of solution set
        print("\nSolution set error rates and number of selected features:")
        for index in range(len(results)):
            print("Error rate:", calc_individual_error_rate(results[index], instances, labels), "    Features Selected:", calc_num_selected_features(results[index]))

        # Calculate hyper-volume area for each solution
        # https://pymoo.org/misc/indicators.html#nb-hv
        hv = get_performance_indicator("hv", ref_point=np.array([1.1, 1.1])) # Set reference point slightly outside of boundary
        print("\nSolution set hyper-volume area:", hv.do(F), "\n")
        
        # Plotting
        plt.figure(figsize=(7, 5))
        plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
        plt.suptitle("Objective Space")
        title = "File: " + file + "   Run: " + str(run + 1)
        plt.title(title)
        plt.xlabel("Error Rate")
        plt.ylabel("Selected Feature Ratio")
        plt.show()

if __name__ == "__main__":
    #main("data/musk/clean1-preprocessed.csv")
    main("data/vehicle/vehicle-preprocessed.csv")