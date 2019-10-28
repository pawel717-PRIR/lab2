#include "LoadCSV.h"

LoadCSV::LoadCSV() {
}

LoadCSV::~LoadCSV() {
}

void LoadCSV::myLoad(string file_path, float *data, int rows, int columns) {
	int i, j, k = 0;
    string val, line;

	ifstream file(file_path);

	for (i = 0; i < rows; i++) {
		getline(file, line);
		if (!file.good())
			break;
		stringstream iss(line);

		for (j = 0; j < columns; j++) {
			getline(iss, val, ',');
			if (!iss.good())
				break;

            *(data+k) = atof(val.c_str());
            k++;
		}
	}
}

