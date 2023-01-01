#include <stdlib.h>
#include <stdio.h>
#include "spike_list.h"

// TODO: search from the end of the linked list as new spikes are likely to be inserted at the end
static spike_list_t* find_insertion_node(spike_list_t *list_start, float time) {
    spike_list_t *node = list_start->prev;

    while (node->time > time)
	node = node->prev;
    return node;
}

static inline spike_list_t *add_first_node(spike_list_t *new_node) {
    new_node->prev = new_node;
    new_node->next = new_node;
    return new_node;
}

static inline spike_list_t *insert_first(spike_list_t *start, spike_list_t *new_node) {
    start->prev->next = new_node;
    new_node->prev = start->prev;
    start->prev = new_node;
    new_node->next = start;
    return new_node;
}

static inline void insert_at(spike_list_t *insert_node, spike_list_t *new_node) {
    insert_node->next->prev = new_node;
    new_node->next = insert_node->next;
    new_node->prev = insert_node;
    insert_node->next = new_node;
}

spike_list_t *spike_list_add(spike_list_t *list, unsigned int new_spike_index,
			     float new_spike_time) {
    spike_list_t *new_node = malloc(sizeof(spike_list_t));
    spike_list_t *insert_node;

    if (new_node == 0)
	return 0;
    new_node->index = new_spike_index;
    new_node->time = new_spike_time;
    if (spike_list_empty(list)) // First element in list
	return add_first_node(new_node);
    else if (new_spike_time <= list->time) // New first element
	return insert_first(list, new_node);
    insert_node = find_insertion_node(list, new_spike_time); // Find insertion node
    insert_at(insert_node, new_node); // Insert after insertion node
    return list;
}

inline bool spike_list_empty(const spike_list_t *list) {
    return list == 0;
}

void spike_list_destroy(spike_list_t *start) {
    spike_list_t *current, *tmp;

    if (spike_list_empty(start))
	return;
    current = start;
    do {
	tmp = current->next;
	free(current);
	current = tmp;
    } while (current != start);
}

void spike_list_print(const spike_list_t *start) {
    const spike_list_t *current;

    if (spike_list_empty(start))
	return;
    current = start;
    do {
	printf("Index: %d, Time: %f\n", current->index, current->time);
	current = current->next;
    } while (current != start);
}

unsigned int spike_list_count(const spike_list_t *start) {
    const spike_list_t *current;
    unsigned int out = 0;

    if (spike_list_empty(start))
	return 0;
    current = start;
    do {
	out++;
	current = current->next;
    } while (current != start);
    return out;
}
