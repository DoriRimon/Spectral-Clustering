from invoke import task
from main import main

# TODO - remove time import and code
import time

"""
Desc:   This is the tasks file as described in the assignment
"""


@task
def build(c):
	"""
	Builds the C module

	:param c: context

	"""
	c.run("python3.8.5 kmeans/setup.py build_ext --inplace")
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
	main(k, n, Random)


"""
This code was done for testing
"""
@task
def test_max_capacity(c, first_n='400', k='5', n_limit='600', time_limit='275', T='5', eps='10'):
	build(c)
	max_n = 0
	first_n, k, n_limit, time_limit, T, eps = list(
		map(int, first_n, k, n_limit, time_limit, T, eps))
	original_first = first_n
	original_limit = n_limit
	for t in range(T):
		first_n = original_first
		n_limit = original_limit
		n = (first_n + n_limit) // 2
		t = 0
		while abs(t - time_limit) > eps:
			n = (first_n + n_limit) // 2
			t = 0
			start_time = time.time()
			main(k, n, False)
			t = time.time() - start_time
			if t > time_limit:
				n_limit = n
			else:
				first_n = n

			if n_limit - first_n == 1:
				break
		max_n += n

	max_n /= T
	print(f'max n: {max_n}')
