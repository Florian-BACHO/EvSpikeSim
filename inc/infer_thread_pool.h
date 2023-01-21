#ifndef INFER_THREAD_POOL
#define INFER_THREAD_POOL

#include <pthread.h>
#include "spike_list.h"

/*
** Thread pool based on this blog implementation:
** https://nachtimwald.com/2019/04/12/thread-pool-in-c/
*/

typedef void (*infer_fct_t)(void *, const spike_list_t *, unsigned int, unsigned int);

// Internal struct for infer works
typedef struct infer_work {
    struct infer_work *next;
    infer_fct_t fct;
    void *layer;
    const spike_list_t *pre_spikes;
    unsigned int neuron_start;
    unsigned int neuron_end;
} infer_work_t;

typedef struct {
    infer_work_t *first_work;
    infer_work_t *last_work;
    pthread_mutex_t work_mutex;
    pthread_cond_t work_cond;
    pthread_cond_t working_cond;
    unsigned int n_working;
    unsigned int n_threads;
    bool stop;
} infer_thread_pool_t;

infer_thread_pool_t *infer_thread_pool_new(unsigned int n_threads);
void infer_thread_pool_destroy(infer_thread_pool_t *pool);
bool infer_thread_pool_add_work(infer_thread_pool_t *pool, infer_fct_t infer_fct, void *layer,
                                const spike_list_t *pre_spikes, unsigned int neuron_start,
				unsigned int neuron_end);
void infer_thread_pool_wait(infer_thread_pool_t *pool);

#endif
