from invoke import task
from main import main

# TODO - remove time import and code
import time

# TODO - document

"""
Desc:   This is the tasks file as described in the assignment
"""


@task
def build(c):
	c.run("python3.8.5 kmeans/setup.py build_ext --inplace")
	print("Done building", flush=True)


@task
def run(c, k, n, Random=True):
	build(c)
	main(k, n, Random)


@task
def test_max_capacity(c, first_n='200', first_k='5', n_step='10', k_step='1', n_time_limit='275', k_time_limit='300',
                      T='5'):
	build(c)
	max_k = 0
	max_n = 0
	first_n, first_k, n_step, k_step, n_time_limit, k_time_limit, T = list(
		map(int, first_n, first_k, n_step, k_step, n_time_limit, k_time_limit, T))
	for t in range(T):
		n = first_n - n_step
		k = first_k - k_step
		t = 0
		while t < n_time_limit:
			n += n_step
			t = 0
			start_time = time.time()
			main(k, n, False)
			t = time.time() - start_time
		while t < k_time_limit:
			k += k_step
			t = 0
			start_time = time.time()
			main(k, n, False)
			t = time.time() - start_time
		max_k += k
		max_n += n
	max_k /= T
	max_n /= T
	print(f'max k: {max_k}, max n: {max_n}')
