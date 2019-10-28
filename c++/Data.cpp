#include "Data.h"

Data::Data() {
	this->data = (float*) malloc (rows * columns * sizeof(float));
}


Data::~Data() {
    free(this->data );
}

void Data::print_data() {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<columns; j++) {
            std::cout << "i:" << i << " , j:" << j << " , data:" << *(data+i*columns+j) << std::endl;
        }
    }
}
