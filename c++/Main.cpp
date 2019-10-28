#include <iostream>
#include "LoadCSV.h"
#include "Data.h"
#include "Preprocessing.h"
#include "KnnAlgorithm.h"
#include <stdio.h>
#include <ctime>
#include <mpi.h>

using namespace std;
/**
 * Run in terminal using following command:
 * mpirun -np 4 cmake-build-debug/lab2
 * where number after -np option specify number of processes
 * @return
 */
int main() {
    int rank, ROOT = 0;

//    freopen( "log.txt", "w", stdout );
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Data ourData = Data();
    Preprocessing preprocessing = Preprocessing();
    KnnAlgorithm knn = KnnAlgorithm();

    if (rank == ROOT) {
        LoadCSV().myLoad("../dataset/mnist_train.csv", ourData.data, ourData.rows, ourData.columns);
    }
	//preprocessing.Normalization(ourData.data, ourData.rows, ourData.columns);
    preprocessing.Standarization(ourData.data, ourData.rows, ourData.columns);
    knn.fit(ourData, 75);
    float accuracy = knn.predict();
    if (rank == ROOT) {
        printf("Accuracy knn: %f\n", accuracy);
    }

    MPI_Finalize();
	return 0;
}
