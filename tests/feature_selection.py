import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from numpy import random
import argparse
import pickle
import seaborn as sns

seed = 100

def load(filename):
    f = open(filename, "rt", encoding="utf-8")
    df = pd.read_csv(f)
    return df

class Fitness:
    def __init__(self, classifier, X, y, prop_train, random_state):
        self._classifier = OneVsRestClassifier(classifier)
        self._X = X
        self._y = label_binarize(y, classes=["bot", "human", "news"])
        self._n_classes = self._y.shape[1]
        self._prop_train = prop_train
        self._random_state = random_state
        self._cache = dict()

    def __call__(self, population):
        n_individuals = population.shape[0]
        pop_fitness = np.zeros(n_individuals)

        for i in range(n_individuals):
            pop_fitness[i] = self._calc(population[i])

        return pop_fitness

    def _calc(self, individual):
        ind_str = "".join(map(str, individual))
        if ind_str in self._cache: return self._cache[ind_str]
        # select a subset of columns.
        X, y = self._select_columns(self._X, individual), self._y

        # Split dataset in train/test
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self._prop_train,
                             random_state=self._random_state)

        # Predict the test set
        y_score = self._classifier.fit(X_train, y_train).predict_proba(X_test)
        fpr, tpr, roc_auc = dict(), dict(), dict()

        for i in range(self._n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self._n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(self._n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= self._n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        self._cache[ind_str] = roc_auc["macro"]
        return roc_auc["macro"]

    def _select_columns(self, X, individual):
        # Select columns where X == 1.
        index = individual == 1
        return X.iloc[:, index]


class FS:
    def __init__(self,
                 classifier,
                 X,
                 y,
                 pop_sz,
                 crossover_prob,
                 mut_prob,
                 k,
                 prop_train,
                 random_state):

        self._classifier = classifier
        self._X = X
        self._y = y
        self._crossover_prob = crossover_prob
        self._mut_prob = mut_prob
        self._k = k
        self._prop_train = prop_train
        self._random_state = random_state

        self._pop_sz = (pop_sz, X.shape[1])
        self._fitness = Fitness(classifier, X, y, prop_train, random_state)
        self._rd = random.RandomState(random_state)
        self._fitness_history = list()

    def generate_initial_population(self):
        self._pop = self._rd.random_integers(0, 1, (self._pop_sz[0], self._pop_sz[1]))
        self._eval_fitness()
        self._fitness_history.append(self._pop_fitness)


    def next_generation(self):
        new_gen = np.zeros(self._pop.shape, dtype=np.int)
        n_ind, ind_sz = self._pop_sz

        for i in range(0, n_ind, 2):
            p1, p2 = self.crossover()
            new_gen[i], new_gen[i+1] = p1, p2

        if n_ind % 2:
            new_gen[-1] = self.crossover()[0]

        for i in range(n_ind):
            self.mutation(new_gen[i])

        self._pop = np.vstack((self._pop, new_gen))
        self._eval_fitness()
        self._pop = self._pop[:n_ind, :]
        self._pop_fitness = self._pop_fitness[:n_ind]
        self._fitness_history.append(self._pop_fitness)


    def crossover(self):
        i1, i2 = self._selection(), self._selection()
        split_point = self._rd.random_integers(0, self._pop_sz[1] + 1)
        temp = self._pop[i1,:]
        p1, p2 = self._pop[i1,:], self._pop[i2,:]
        p1[split_point:] = p2[split_point:]
        p2[:split_point] = temp[:split_point]
        return p1, p2


    def mutation(self, p1):
        for i in range(self._pop_sz[1]):
            if self._rd.random() < self._mut_prob:
                p1[i] = (p1[i] + 1) % 2

        return p1


    def _eval_fitness(self):
        pop_fitness = self._fitness(self._pop)
        sorted_fitness = np.argsort(-pop_fitness)

        self._pop = self._pop[sorted_fitness, :]
        self._pop_fitness = pop_fitness

    def _selection(self):
        n_individuals = self._pop_sz[0]
        tournament = self._rd.choice(np.arange(n_individuals), self._k)
        tournament = np.sort(tournament)
        return tournament[0] # already sorted. return index to population


    def save(self, filename):
        print(self._fitness_history)
        f = open(filename, "wb")
        pickle.dump(self._fitness_history, f)

    
def main():
    aparse = argparse.ArgumentParser()

    aparse.add_argument("output",
                        nargs=1,
                        type=str,
                        help="file to save fitness history.")
    
    aparse.add_argument("--n-estimators",
                        nargs=1,
                        type=int,
                        default=[100],
                        help="number of trees.")

    aparse.add_argument("--classifier",
                        nargs=1,
                        type=str,
                        choices=["RF", "SVM"],
                        default=["RF"],
                        help="classifier to be used.")

    aparse.add_argument("--prob-crossover",
                        nargs=1,
                        type=float,
                        default=[0.8],
                        help="crossover probability.")

    aparse.add_argument("--prob-mutation",
                         nargs=1,
                         type=float,
                         default=[0.2],
                         help="mutation probability.")

    aparse.add_argument("--population-size",
                         nargs=1,
                         type=int,
                         default=[10],
                         help="size of population.")

    aparse.add_argument("--tournament-size",
                         nargs=1,
                         type=int,
                         default=[3],
                         help="size of tournament.")

    aparse.add_argument("--sample-prop",
                         nargs=1,
                         type=float,
                         default=[0.5],
                         help="propotion to sample of the dataset.")

    aparse.add_argument("--random-state",
                         nargs=1,
                         type=int,
                         default=[0],
                         help="random state")

    aparse.add_argument("--max-generations",
                        nargs=1,
                        type=int,
                        default=[40],
                        help="maximum of generations.")
    
    args = aparse.parse_args()

    print("loading dataset", end="\r")
    df = load("dataset.csv")
    X = df.iloc[:, 2:-2]
    y = df.iloc[:, -1]
    print("dataset loaded.", end="\r")
    print(" "*30, end="\r")


    output = args.output[0]
    n_estimators = args.n_estimators[0]
    classifier_type = args.classifier[0]
    prob_crossover = args.prob_crossover[0]
    prob_mutation = args.prob_mutation[0]
    population_size = args.population_size[0]
    tournament_size = args.tournament_size[0]
    sample_prop = args.sample_prop[0]
    random_state = args.random_state[0]
    max_generations = args.max_generations[0]
    
    
    print(output, n_estimators, classifier_type, prob_crossover, prob_mutation,
          population_size, tournament_size, sample_prop, random_state)

    if classifier_type == "RF":
        classifier = RandomForestClassifier(n_estimators=n_estimators)
    elif classifier_type == "SVM":
        classifier = SVC(probability=True)

        
    fs = FS(classifier,
            X, y,
            population_size,
            prob_crossover,
            prob_mutation,
            tournament_size,
            sample_prop,
            random_state)
    
    fs.generate_initial_population()
    for i in range(max_generations):
        print(f"generation: {i+1}")
        fs.next_generation()

    fs.save(output)
    

if __name__ == "__main__":
    main()
