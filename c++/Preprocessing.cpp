#include "Preprocessing.h"

Preprocessing::Preprocessing() {
}


Preprocessing::~Preprocessing() {
}

void Preprocessing::Normalization(float *data, int rows, int columns) {
    int min, max, comm_size, rank, ROOT = 0, col_start_index = 0;
    float max_float = std::numeric_limits<float>::max();
    MPI_Datatype column_type, column_wise_array_type, column_type2, column_wise_array_type2;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    int part_columns = columns / comm_size;
    float * data_part = (float*)malloc(rows * part_columns * sizeof(float));

    if (rank == ROOT) {
        // need to skip first column in data (this is column containing labels)
        col_start_index = 1;
        MPI_Type_vector(rows, 1, columns, MPI_FLOAT, &column_type);
        MPI_Type_commit(&column_type);
        MPI_Type_create_resized(column_type, 0, 1 * sizeof(float), &column_wise_array_type);
        MPI_Type_commit(&column_wise_array_type);
    }

    MPI_Type_vector(rows, 1, part_columns, MPI_FLOAT, &column_type2);
    MPI_Type_commit(&column_type2);
    MPI_Type_create_resized(column_type2, 0, 1 * sizeof(float), &column_wise_array_type2);
    MPI_Type_commit(&column_wise_array_type2);

    MPI_Scatter(data, part_columns, column_wise_array_type, data_part,
            part_columns, column_wise_array_type2, ROOT, MPI_COMM_WORLD);

    for (int i = col_start_index; i < part_columns; ++i) {
        min = max_float; max = 0;
		for (int j = 0; j < rows; ++j) {
			if (*(data_part + (j*part_columns)+i) < min) {
				min = *(data_part + (j*part_columns)+i);
			} else if (*(data_part + (j*part_columns)+i) > max) {
				max = *(data_part + (j*part_columns)+i);
			}
		}

		float max_min_reciprocal = max - min;
		if (max_min_reciprocal == 0) {
			continue;
		}
		max_min_reciprocal = 1. / max_min_reciprocal;

		for (int j = 0; j < rows; ++j) {
            *(data_part + (j*part_columns)+i) = (*(data_part + (j*part_columns)+i) - min) * max_min_reciprocal;
		}
	}

    MPI_Gather(data_part, part_columns, column_wise_array_type2, data,
            part_columns, column_wise_array_type, ROOT, MPI_COMM_WORLD);
    free(data_part);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == ROOT) {
        printf("Czas obliczen normalizacja: %f\n", MPI_Wtime() - startTime);
    }
}


void Preprocessing::Standarization(float *data, int rows, int columns) {
    int comm_size, rank, ROOT = 0, col_start_index = 0;
    float var, ave, amo;
    MPI_Datatype column_type, column_wise_array_type, column_type2, column_wise_array_type2;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    int part_columns = columns / comm_size;
    float * data_part = (float*)malloc(rows * part_columns * sizeof(float));

    if (rank == ROOT) {
        // need to skip first column in data (this is column containing labels)
        col_start_index = 1;
        MPI_Type_vector(rows, 1, columns, MPI_FLOAT, &column_type);
        MPI_Type_commit(&column_type);
        MPI_Type_create_resized(column_type, 0, 1 * sizeof(float), &column_wise_array_type);
        MPI_Type_commit(&column_wise_array_type);
    }

    MPI_Type_vector(rows, 1, part_columns, MPI_FLOAT, &column_type2);
    MPI_Type_commit(&column_type2);
    MPI_Type_create_resized(column_type2, 0, 1 * sizeof(float), &column_wise_array_type2);
    MPI_Type_commit(&column_wise_array_type2);

    MPI_Scatter(data, part_columns, column_wise_array_type, data_part,
            part_columns, column_wise_array_type2, ROOT, MPI_COMM_WORLD);

    for (int i = col_start_index; i < part_columns - 1; ++i) {
        amo = 0, var = 0;
        for (int j = 0; j < rows; ++j) {
            amo = amo + *(data_part + (j * part_columns) + i);
        }
        ave  = amo / float(rows);

        for (int j = 0; j < rows; ++j) {
            float factor = *(data_part + (j * part_columns) + i) - ave;
            var = var + (factor * factor);
        }

        if (var == 0) {
            for (int j = 0; j < rows; j++) {
                *(data_part + (j * part_columns) + i) = *(data_part + (j * part_columns) + i) / 255.;
            }
            continue;
        }

        float sd_reciprocal = 1./sqrt(var);

        for (int j = 0; j < rows; j++) {
            *(data_part + (j * part_columns) + i) = (*(data_part + (j * part_columns) + i) - ave) * sd_reciprocal;
        }

    }

    MPI_Gather(data_part, part_columns, column_wise_array_type2, data, part_columns, column_wise_array_type,
               ROOT, MPI_COMM_WORLD);
    free(data_part);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == ROOT) {
        printf("Czas obliczen standaryzacja: %f\n", MPI_Wtime() - startTime);
    }
}