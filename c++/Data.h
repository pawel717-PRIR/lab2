#pragma once
#include <stdlib.h>
#include <iostream>

class Data {
public:
	Data();
	~Data();
    void print_data();

	float *data;
	int rows = 6000;
	int columns = 784;
};

