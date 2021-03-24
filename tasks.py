from invoke import task
from main import main

"""
Desc:   This is the tasks file as described in the assignment
"""


@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")
    print("Done building", flush=True)


@task
def run(c, k, n, Random=True):
    # build(c) TODO - solve import problem
    main(k, n, Random)
