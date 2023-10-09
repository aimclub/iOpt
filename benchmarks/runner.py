from experiment import Experiment
from argparser import parse_arguments
from frameworks import OptunaSearcher, HyperoptSearcher, iOptSearcher


def main():
    arguments = parse_arguments()

    seachers = [
        OptunaSearcher(arguments.max_iter, algorithm='random'),
        OptunaSearcher(arguments.max_iter, algorithm='tpe'),
        OptunaSearcher(arguments.max_iter, algorithm='cmaes'),
        OptunaSearcher(arguments.max_iter, algorithm='nsgaii'),
        HyperoptSearcher(arguments.max_iter),
        iOptSearcher(arguments.max_iter, r=3, refine_solution=True,
                     proportion_of_global_iterations=0.75)
    ]

    experiment = Experiment(arguments.estimator,
                            arguments.hyperparams,
                            seachers,
                            arguments.dataset)

    experiment.run(arguments.dir,
                   non_deterministic_trials=arguments.trials,
                   n_jobs=arguments.n_jobs)


if __name__ == '__main__':
    main()
