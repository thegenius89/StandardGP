from pickle import dump, load
from collections import defaultdict

from GP import GP


class FunctionSpaceLearner:

    def __init__(self):
        self.equivalent_matrix = defaultdict(list)
        self.load()

    def save(self):
        with open("../store/equivalent_store", "wb") as handle:
            dump(self.equivalent_matrix, handle)

    def load(self):
        with open("../store/equivalent_store", "rb") as handle:
            self.equivalent_matrix = load(handle)

    def build_equivalent_matrix(self):
        self.save()

    def build_reduction_matrix(self):
        self.save()

    def build_pam_matrix(self):
        self.save()

    def learn_from_problems(self):
        self.save()


def main():
    learner = FunctionSpaceMetaLearner()
    learner.build_equivalent_matrix()
    # learner.build_reduction_matrix()
    # learner.build_pam_matrix()
    # learner.learn_from_problems()


if __name__ == "__main__":
    main()
