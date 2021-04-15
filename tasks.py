from invoke import task
from main import main

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
