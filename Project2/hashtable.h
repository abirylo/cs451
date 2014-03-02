#include <string.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct key_value_pair{
	const void *key;
	int val;
} kvp_t;

typedef struct ht_node{
	struct ht_node *next;
	kvp_t kvp;
} ht_node_t;

typedef struct{
	ht_node_t **arr;
	size_t cap;
	size_t size;
	int (*cmp)(const void*, const void*);
	uint32_t (*hash_code)(const void*);
} hashtable_t;

void ht_init(hashtable_t*);
void ht_init_s(hashtable_t*, size_t);

#define HT_INIT_VOID(t, cmp, hash_code) ht_init_void(t, (int (*)(const void*, const void*))(cmp), (uint32_t (*)(const void*))(hash_code))
void ht_init_void(hashtable_t*, int (*)(const void*, const void*), uint32_t (*)(const void*));

#define HT_INIT_VOID_S(t, capacity, cmp, hash_code) ht_init_void_s(t, capacity, (int (*)(const void*, const void*))(cmp), (uint32_t (*)(const void*))(hash_code))
void ht_init_void_s(hashtable_t*, size_t, int (*)(const void*, const void*), uint32_t (*)(const void*));

#define HT_ADD(t, key, val) ht_add(t, (const void*)(uint64_t)(key), (void*)(uint64_t)(val))
void *ht_add(hashtable_t*, const void*, int);
#define HT_GET(t, key) ht_get(t, (const void*)(uint64_t)(key))
void *ht_get(hashtable_t*, const void*);
#define HT_DELETE(t, key) ht_delete(t, (const void*)(uint64_t)(key))
void *ht_delete(hashtable_t*, const void*);
void ht_dispose(hashtable_t*);
uint32_t string_hash(const char*);
int string_eq(const char*, const char*);

void get_all_kvps(hashtable_t*, kvp_t*);
