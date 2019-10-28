#pragma once
#include <stdlib.h>
#include <fstream>
#include <sstream>

using namespace std;

class LoadCSV {
public:
	LoadCSV();
	~LoadCSV();
	void myLoad(string file_path, float *data, int rows, int columns);
};

