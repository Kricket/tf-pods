from multiprocessing import Pool
from typing import Callable, Iterable


def run_multiproc(n_proc: int, func: Callable, args: Iterable[Iterable]):
    """
    Execute the given function in multiple parallel processes
    :param n_proc: Number of processes to spawn
    :param func: Function to execute
    :param args: Arguments of the function for each process
    :return: A list of the return values of each call of the function
    """
    with Pool(n_proc) as p:
        results = p.starmap(func, args)
    p.close()
    p.join()
    return results
