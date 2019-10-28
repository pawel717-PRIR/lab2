#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <mpi.h>
#include <limits>

class Preprocessing
{
public:
	Preprocessing();
	~Preprocessing();
    void Normalization(float *data, int rows, int columns);
	void Standarization(float *data, int rows, int columns);
};

