#ifndef SPIKE_LIST
#define SPIKE_LIST

#include <stdbool.h>

// Doubly Circular Linked List of spike events
typedef struct spike_list {
    struct spike_list *prev;
    struct spike_list *next;
    unsigned int index;
    float time;
} spike_list_t;

spike_list_t *spike_list_add(spike_list_t *start, unsigned int new_spike_index,
			     float new_spike_time);
bool spike_list_empty(const spike_list_t *start);
void spike_list_destroy(spike_list_t *start);
void spike_list_print(const spike_list_t *start);
unsigned int spike_list_count(const spike_list_t *start);

#endif
