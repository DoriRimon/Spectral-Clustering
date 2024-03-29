#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void error_handling();
static double norm();
static void initialize();
static void clusters_assignments();
static void clustering();
static void update_centroids();
static void deallocate_imatrix();
static void deallocate_dmatrix();
static PyObject* kmeans();

static int k, N, d, MAX_ITER;

/* --------------- Error Handling --------------- */
static void error_handling(const char *error) {
    char res[200];
    strcpy(res, "---------- || Please notice! || ----------\nAn error was caught:\n");
    strcat(res, "Error:     ");
    strcat(res, error);
    strcat(res, "\nPath:      ./kmeans/kmeans.c\n");

    printf("%s", res);
    exit(1);
}

/* --------------- Memory Handling --------------- */

/* Allocate a matrix of ints */
static int** allocate_imatrix(int rows, int cols) {
    int i;
    int **matrix = calloc(rows, sizeof(int*));

    if (matrix == NULL) {
        error_handling("Memory allocation failed");
    }
    for (i = 0; i < rows; i++) {
        matrix[i] = calloc(cols, sizeof(int));
        if (matrix[i] == NULL) {
            error_handling("Memory allocation failed");
        }
    }
    return matrix;
}

/* Deallocate a matrix of ints */
static void deallocate_imatrix(int** matrix, int rows)
{
    int i;
    for(i = 0 ; i < rows ; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

/* Allocate a matrix of doubles */
static double** allocate_dmatrix(int rows, int cols) {
    int i;
    double **matrix = calloc(rows, sizeof(double*));
    if (matrix == NULL) {
        error_handling("Memory allocation failed");
    }
    for (i = 0; i < rows; i++) {
        matrix[i] = calloc(cols, sizeof(double));
        if (matrix[i] == NULL) {
            error_handling("Memory allocation failed");
        }
    }
    return matrix;
}

/* Deallocate a matrix of doubles */
static void deallocate_dmatrix(double** matrix, int rows)
{
    int i;
    for(i = 0 ; i < rows ; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

/* Copy the values from the vector v to the vector u */
static void vector_dassign(double *u, double *v, int size) {
    int i;
    for (i=0; i<size; i++) {
        u[i] = v[i];
    }
}

/* The main function of the code */
static PyObject* m(double** obs, double** cents, int** cltrs) {
    double **observations = obs;
    double **centroids = cents;
    int **clusters = cltrs;
    PyObject* res, *temp;
    Py_ssize_t i, j;

    /* cluster */
    clustering(observations, clusters, centroids);

    /* build the result python list from observations*/
    res = PyList_New(N);
    for (i = 0; i < N; i++) {
        temp = PyList_New(d+1);
        for (j = 0; j < d+1; j++) {
            PyList_SET_ITEM(temp, j, Py_BuildValue("d", observations[i][j]));
        }
        PyList_SET_ITEM(res, i, temp);
    }

    /* deallocating matrices */
    deallocate_dmatrix(observations,N);
    deallocate_dmatrix(centroids,k);
    deallocate_imatrix(clusters,k);

    return res;
}

/* Initialize the centroids from the given init_centroids indexes */
static void initialize(int* init_centroids, double** centroids, double** observations, int** clusters)
{
    int i;
    for (i = 0; i < k; i++) {
        int index = init_centroids[i];
        clusters[i][index] = 1;
        vector_dassign(centroids[i], observations[index], d);
        centroids[i][d] = i;
    }
}

/* Update centroids to be the mean value of the cluster, as described in the kmeans algorithm */
static void update_centroids(double **centroids, double** observations) {
    int i;
    double **sum = allocate_dmatrix(k, d);
    int r;
    int *counter = (int *)calloc(k, sizeof(int));
    if (sum == NULL) {
        error_handling("Memory allocation failed");
    }

    for (i = 0; i < N; i++)
    {
        int cluster = observations[i][d];
        int j;
        counter[cluster]++;
        for (j = 0; j < d; j++)
        {
            sum[cluster][j] += observations[i][j];
        }
    }
    for (i = 0; i < k; i++)
    {
        for (r = 0; r < d; r++) {
            if (counter[i] == 0) {
                error_handling("Division By Zero");
            }
            sum[i][r] /= counter[i];
        }
        vector_dassign(centroids[i], sum[i], d);
    }
    deallocate_dmatrix(sum, k);
    free(counter);
}

/* Assign observation i to cluster val */
/* Also removes observation from the previous cluster (prev) */
static void clusters_assignments(int i, int prev, int val, double **observations, int **clusters) {
    observations[i][d] = val;
    if(prev >= 0)
    {
        clusters[prev][i] = 0;
    }
    clusters[val][i] = 1;
}

/* The main clustering algorithm */
static void clustering(double **observations, int **clusters, double **centroids) {
    int iter = 0;
    int flag = 0;

    while (iter < MAX_ITER && flag != 1) {
        int i;
        flag = 1;

        for (i = 0; i < N; i++) {
            int j = 0;
            int r;
            double n = norm(observations[i], centroids[0], d);

            for (r = 1; r < k; r++) {
                double val = norm(observations[i], centroids[r], d);
                if (val < n) {
                    n = val;
                    j = r;
                }
            }

            if (j != observations[i][d]) {
                flag = 0;
                clusters_assignments(i, observations[i][d], j, observations, clusters);
            }
        }

        update_centroids(centroids, observations);
        iter++;
    }
}

/* Calculate the norm of two vectors */
static double norm(double *v1, double *v2, int length) {
    double sum = 0;
    int i;
    for (i = 0; i < length; ++i) {
        sum = sum + (v2[i] - v1[i]) * (v2[i] - v1[i]);
    }
    return sum;
}

/* --------------- Python Handling --------------- */

/*
    input: double** observations, int* centroids, int k, int N, int d, int MAX_ITER
    output: double**, shape(n, d+1) - Matrix of observations, where at the last index of each
    observation is the cluster index
*/
static PyObject* kmeans(PyObject *self, PyObject *args) {

    Py_ssize_t i, j;
    PyObject *_list;
    PyObject *item;
    PyObject *py_observations;
    PyObject *py_centroids;
    double** observations;
    int* init_centroids;
    double** centroids;
    int** clusters;

    if(!PyArg_ParseTuple(args, "O:kmeans", &_list)) {
        return NULL;
    }

    if (!PyList_Check(_list))
        return NULL;

    py_observations = PyList_GetItem(_list, 0);
    py_centroids = PyList_GetItem(_list, 1);
    k = PyLong_AsLong(PyList_GetItem(_list, 2));
    N = PyLong_AsLong(PyList_GetItem(_list, 3));
    d = PyLong_AsLong(PyList_GetItem(_list, 4));
    MAX_ITER = PyLong_AsLong(PyList_GetItem(_list, 5));

    observations = allocate_dmatrix(N, d+1);
    init_centroids = malloc(sizeof(int) * k);
    centroids = allocate_dmatrix(k, d+1);
    clusters = allocate_imatrix(k, N);

    for (i = 0; i < N; i++) {
        item = PyList_GetItem(py_observations, i);
        for (j = 0; j < d; j++) {
            observations[i][j] = PyFloat_AsDouble(PyList_GetItem(item, j));
        }
        observations[i][d] = -1;
    }

    for (i = 0; i < k; i++) {
        item = PyList_GetItem(py_centroids, i);
        init_centroids[i] = PyLong_AsLong(item);
    }

    initialize(init_centroids, centroids, observations, clusters);
    free(init_centroids);

    return m(observations, centroids, clusters);
}

/*
 * A macro to help us with defining the methods
 * Compare with: {"f1", (PyCFunction)f1, METH_NOARGS, PyDoc_STR("No input parameters")}
*/
#define FUNC(_flag, _name, _docstring) { #_name, (PyCFunction)_name, _flag, PyDoc_STR(_docstring) }

static PyMethodDef _methods[] = {
    {"kmeans", (PyCFunction)kmeans, METH_VARARGS, PyDoc_STR("6 input parameters")},
    {NULL, NULL, 0, NULL}   /* sentinel */
};

static struct PyModuleDef _moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    _methods
};

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    return PyModule_Create(&_moduledef);
}