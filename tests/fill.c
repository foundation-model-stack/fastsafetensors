// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 20
#define ARRAY_SIZE (160 * 1024 * 1024 / NUM_THREADS / sizeof(long long))

long long *data;

void *thread_function(void *arg) {
    data = (long long *)malloc(ARRAY_SIZE * sizeof(long long));
    if (data == NULL) {
        perror("Memory allocation failed");
        return NULL;
    }
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        data[i] = 0;
    }
    while (1) {
        for (size_t i = 0; i < ARRAY_SIZE; i++) {
            data[i] += 1;
        }
    }
    free(data);
    return NULL;
}

int main() {
    srand(time(NULL));
    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, thread_function, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    return 0;
}

