#include <stdlib.h>
#include "random.h"

void random_set_seed(unsigned int seed) {
    srand(seed);
}

inline float random_uniform_float(float min, float max) {
    return (float)rand() / (float)RAND_MAX * (max - min) + min;
}

inline int random_uniform_int(int min, int max) {
    return (rand() % (min - max)) + min;
}
