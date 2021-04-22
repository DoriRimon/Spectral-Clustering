from invoke import task
from main import main

"""
Desc:   This is the tasks file as described in the assignment
"""


@task
def build(c):
	"""
	Builds the C module

	:param c: context

	"""
	c.run("python3.8.5 setup.py build_ext --inplace")
	print("Done building the kmeans module", flush=True)


@task
def run(c, k=None, n=None, Random=True):
	"""
	Run Forrest run

	:param c: context
	:param k: int
	:param n: int
	:param Random: boolean

	"""
	build(c)

	import cProfile, pstats, sys
	pr = cProfile.Profile()
	pr.enable()

	main(k, n, Random)

	pr.disable()
	sortby = pstats.SortKey.TIME
	ps = pstats.Stats(pr, stream=sys.stdout).sort_stats(sortby)
	ps.print_stats()
