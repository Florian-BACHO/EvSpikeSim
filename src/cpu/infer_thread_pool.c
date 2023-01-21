#include <stdlib.h>
#include "infer_thread_pool.h"

static infer_work_t *infer_work_new(infer_fct_t infer_fct, void *layer,
				    const spike_list_t *pre_spikes, unsigned int neuron_start,
				    unsigned int neuron_end) {
    infer_work_t *work = malloc(sizeof(infer_work_t));

    if (work == 0)
	return 0;
    work->next = 0;
    work->fct = infer_fct;
    work->layer = layer;
    work->pre_spikes = pre_spikes;
    work->neuron_start = neuron_start;
    work->neuron_end = neuron_end;
    return work;
}

static void infer_work_destroy(infer_work_t *work) {
    free(work);
}

static infer_work_t *get_next_work(infer_thread_pool_t *pool) {
    infer_work_t *work;

    work = pool->first_work;
    if (work == 0)
        return 0;
    if (work->next == 0) {
        pool->first_work = 0;
        pool->last_work  = 0;
    } else
        pool->first_work = work->next;
    return work;
}

static void *worker_thread(void *arg) {
    infer_thread_pool_t *pool = (infer_thread_pool_t *)arg;
    infer_work_t *work;

    while (true) {
        pthread_mutex_lock(&(pool->work_mutex));
	// If no work available: awaits for work or stop
        while (pool->first_work == 0 && !pool->stop)
            pthread_cond_wait(&(pool->work_cond), &(pool->work_mutex));
	if (pool->stop)
            break;
	// Get next work
        work = get_next_work(pool);
        pool->n_working++;
        pthread_mutex_unlock(&(pool->work_mutex));
	// Execute inference work
        if (work != 0) {
            work->fct(work->layer, work->pre_spikes, work->neuron_start, work->neuron_end);
            infer_work_destroy(work);
        }
        pthread_mutex_lock(&(pool->work_mutex));
        pool->n_working--;
	// Signal end of inference
        if (!pool->stop && pool->n_working == 0 && pool->first_work == 0)
	    pthread_cond_signal(&(pool->working_cond));
        pthread_mutex_unlock(&(pool->work_mutex));
    }
    // Terminate thread
    pool->n_threads--;
    pthread_cond_signal(&(pool->working_cond));
    pthread_mutex_unlock(&(pool->work_mutex));
    return 0;
}

infer_thread_pool_t *infer_thread_pool_new(unsigned int n_threads) {
    infer_thread_pool_t *pool = calloc(1, sizeof(infer_thread_pool_t));
    pthread_t thread;

    if (pool == 0)
	return 0;
    pool->n_threads = n_threads;
    pthread_mutex_init(&(pool->work_mutex), 0);
    pthread_cond_init(&(pool->work_cond), 0);
    pthread_cond_init(&(pool->working_cond), 0);
    pool->first_work = 0;
    pool->last_work  = 0;
    pool->stop = false;
    for (unsigned int i = 0; i < n_threads; i++) {
        pthread_create(&thread, 0, worker_thread, (void *)pool);
        pthread_detach(thread);
    }
    return pool;
}

void infer_thread_pool_destroy(infer_thread_pool_t *pool) {
    infer_work_t *work;
    infer_work_t *tmp;

    pthread_mutex_lock(&(pool->work_mutex));
    work = pool->first_work;
    while (work != 0) {
        tmp = work->next;
        infer_work_destroy(work);
        work = tmp;
    }
    pool->stop = true;
    pthread_cond_broadcast(&(pool->work_cond));
    pthread_mutex_unlock(&(pool->work_mutex));

    infer_thread_pool_wait(pool);

    pthread_mutex_destroy(&(pool->work_mutex));
    pthread_cond_destroy(&(pool->work_cond));
    pthread_cond_destroy(&(pool->working_cond));

    free(pool);
}

bool infer_thread_pool_add_work(infer_thread_pool_t *pool, infer_fct_t infer_fct, void *layer,
				const spike_list_t *pre_spikes, unsigned int neuron_start,
				unsigned int neuron_end) {
    infer_work_t *work;

    if (pool == 0)
        return false;
    // Create new work for inference
    work = infer_work_new(infer_fct, layer, pre_spikes, neuron_start, neuron_end);
    if (work == 0)
        return false;
    // Add work to queue
    pthread_mutex_lock(&(pool->work_mutex));
    if (pool->first_work == 0) {
        pool->first_work = work;
        pool->last_work = work;
    } else {
        pool->last_work->next = work;
        pool->last_work = work;
    }
    // Signal threads that new work is available
    pthread_cond_broadcast(&(pool->work_cond));
    pthread_mutex_unlock(&(pool->work_mutex));
    return true;
}

void infer_thread_pool_wait(infer_thread_pool_t *pool) {
    if (pool == 0)
        return;
    pthread_mutex_lock(&(pool->work_mutex));
    while (true) {
        if ((!pool->stop && pool->n_working != 0) ||
	    (!pool->stop && pool->first_work != 0) ||
	    (pool->stop && pool->n_threads != 0)) {
            pthread_cond_wait(&(pool->working_cond), &(pool->work_mutex));
        } else {
            break;
        }
    }
    pthread_mutex_unlock(&(pool->work_mutex));
}
