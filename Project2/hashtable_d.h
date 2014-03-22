#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef __hash_d__
#define __hash_d__

typedef struct key_value_pair_d{
	const void *key;
	int val;
} kvp_t_d;

typedef struct ht_node_d{
	struct ht_node_d *next;
	kvp_t_d kvp;
} ht_node_t_d;

typedef struct{
	ht_node_t_d **arr;
	size_t cap;
	size_t size;
	int (*cmp)(const void*, const void*);
	uint32_t (*hash_code)(const void*);
} hashtable_t_d;

__device__ void ht_init_d(hashtable_t_d*);
__device__ void ht_init_s_d(hashtable_t_d*, size_t);

#define HT_INIT_VOID_D(t, cmp, hash_code) ht_init_void_d(t, (int (*)(const void*, const void*))(cmp), (uint32_t (*)(const void*))(hash_code))
__device__ void ht_init_void_d(hashtable_t_d*, int (*)(const void*, const void*), uint32_t (*)(const void*));

#define HT_INIT_VOID_S_D(t, capacity, cmp, hash_code) ht_init_void_s_d(t, capacity, (int (*)(const void*, const void*))(cmp), (uint32_t (*)(const void*))(hash_code))
__device__ void ht_init_void_s_d(hashtable_t_d*, size_t, int (*)(const void*, const void*), uint32_t (*)(const void*));

#define HT_ADD_D(t, key, val) ht_add_d(t, (const void*)(uint64_t)(key), (void*)(uint64_t)(val))
__device__ int ht_add_d(hashtable_t_d*, const void*, int);
#define HT_GET_D(t, key) ht_get_d(t, (const void*)(uint64_t)(key))
__device__ int ht_get_d(hashtable_t_d*, const void*);
#define HT_DELETE_D(t, key) ht_delete_d(t, (const void*)(uint64_t)(key))
__device__ int ht_delete_d(hashtable_t_d*, const void*);
__device__ void ht_dispose_d(hashtable_t_d*);
__device__ uint32_t string_hash_d(const char*);
__device__ int string_eq_d(const char*, const char*);

__device__ void get_all_kvps_d(hashtable_t_d*, kvp_t_d*);

#endif
