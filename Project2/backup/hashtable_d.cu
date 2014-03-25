#include "hashtable_d.h"

__device__ static ht_node_t_d *ht_get_ht_node_d(hashtable_t_d*,const void*);
__device__ static ht_node_t_d *ht_get_ht_node_i_d(hashtable_t_d*,const void*,uint32_t);
__device__ static void ht_rehash_d(hashtable_t_d*, size_t);

__device__ void ht_init_void_d(hashtable_t_d *t, int (*cmp)(const void*, const void*), uint32_t (*hash_code)(const void*)){
	ht_init_void_s_d(t, 16, cmp, hash_code);
}

__device__ void ht_init_void_s_d(hashtable_t_d *t, size_t capacity, int (*cmp)(const void*, const void*), uint32_t (*hash_code)(const void*)){
	uint32_t cap = 1;
	while(capacity > cap) cap <<= 1;

	t->cap = cap;
	t->arr = (ht_node_t_d**)malloc(t->cap*sizeof(ht_node_t_d*));
  memset(t->arr, 0, t->cap*sizeof(ht_node_t_d*));
	t->size = 0;
	t->cmp = cmp;
	t->hash_code = hash_code;
}

__device__ void ht_init_s_d(hashtable_t_d *t, size_t capacity){
	HT_INIT_VOID_S_D(t, capacity, string_eq_d, string_hash_d);
}

__device__ void ht_init_d(hashtable_t_d *t){
	ht_init_s_d(t, 16);
}

__device__ int ht_add_d(hashtable_t_d *t, const void* key, int val){
	if(t->size >= (t->cap >> 1) + (t->cap >> 2)){
		ht_rehash_d(t, t->cap << 1);
	}

	uint32_t index = t->hash_code(key) & (t->cap - 1);
	int ret = -1;
	ht_node_t_d *n = ht_get_ht_node_i_d(t, key, index);
	if(!n){
		n = (ht_node_t_d*)malloc(sizeof(ht_node_t_d));
		n->next = t->arr[index];
		n->kvp.key = key;

		t->arr[index] = n;
		t->size++;
	}
	else{
		ret = n->kvp.val;
	}
	n->kvp.val = val;

	return ret;
}

__device__ int ht_get_d(hashtable_t_d *t, const void* key){
	ht_node_t_d *n = ht_get_ht_node_d(t, key);
	return n?n->kvp.val:-1;
}

__device__ ht_node_t_d *ht_get_ht_node_d(hashtable_t_d *t, const void* key){
	return ht_get_ht_node_i_d(t, key, t->hash_code(key) & (t->cap - 1));
}

__device__ ht_node_t_d *ht_get_ht_node_i_d(hashtable_t_d *t, const void* key, uint32_t index){
	ht_node_t_d *n;
	for(n = t->arr[index]; n && !t->cmp(key, n->kvp.key); n = n->next);
	return n;
}

__device__ int ht_delete_d(hashtable_t_d *t, const void*key){
	uint32_t index = t->hash_code(key) & (t->cap-1);
	int ret = NULL;
	ht_node_t_d *n = t->arr[index];
	if(!n) return ret;
	if(t->cmp(key,n->kvp.key)){
		t->arr[index] = n->next;
		ret = n->kvp.val;
		free(n);
		t->size--;
	}
	else{
		for(; n->next && !t->cmp(key, n->next->kvp.key); n = n->next);
		if(n->next){
			ht_node_t_d *target = n->next;
			n->next = target->next;
			ret = target->kvp.val;
			free(target);
			t->size--;
		}
	}
	if(t->size <= t->cap >> 2 && t->cap > 16){
		ht_rehash_d(t, t->cap >> 1);
	}
	return ret;
}

__device__ void subrehash_d(hashtable_t_d *t, ht_node_t_d **newarr, size_t newcap, ht_node_t_d *n){
	if(n->next){
		subrehash_d(t, newarr, newcap, n->next);
	}
	uint32_t index = t->hash_code(n->kvp.key) & (newcap-1);
	n->next = newarr[index];
	newarr[index] = n;
}

__device__ void ht_rehash_d(hashtable_t_d *t, size_t newcap){
	ht_node_t_d **newarr = (ht_node_t_d**)malloc(newcap*sizeof(ht_node_t_d*));
	memset(newarr, 0, newcap*sizeof(ht_node_t_d*));
  uint32_t i;
	for(i = 0; i < t->cap; i++){
		if(t->arr[i]){
			subrehash_d(t, newarr, newcap, t->arr[i]);		
		}
	}
	free(t->arr);
	t->arr = newarr;
	t->cap = newcap;
}

__device__ void subdispose_d(ht_node_t_d *n){
	if(n->next){
		subdispose_d(n->next);
	}
	free(n);
}

__device__ void ht_dispose_d(hashtable_t_d *t){
	ht_node_t_d *n;
	uint32_t i;
	for(i = 0; i < t->cap; i++){
		n = t->arr[i];
		if(n){
			subdispose_d(n);
		}
	}
	free(t->arr);
}

#define FNV_PRIME 0x1000193
#define FNV_BASE 0x811C9DC5 

__device__ uint32_t string_hash_d(const char* key){
	uint32_t i, hc = FNV_BASE;
	for(i = 0; key[i]; i++){
		hc ^= key[i];
		hc *= FNV_PRIME;
	}
	return hc;
}

__device__ int string_eq_d(const char *a, const char *b){
	while(*a != NULL && *b != NULL){
    if(*a != *b) return 0;
    a++;
    b++;
  }
  if(*a==*b) return 1;
  else return 0;
}

__device__ void get_all_kvps_d(hashtable_t_d *t, kvp_t_d *kvps){
	ht_node_t_d *n;
	uint32_t i, j = 0;
	for(i = 0; i < t->cap; i++){
		n = t->arr[i];
		while(n){
			kvps[j++] = n->kvp;
			n = n->next;
		}
	}
}

/*
void ht_print(hashtable_t *t){
	ht_node_t *n;
	uint32_t i;
	for(i = 0; i < t->cap; i++){
		printf("%03u: ",i);
		n = t->arr[i];
		while(1){
			if(n){
				printf("k:%p v:%p -> ",n->kvp.key,n->kvp.val);
				n = n->next;
			}
			else{
				printf("null\n");
				break;
			}
		}
	}
	printf("\n");
}
*/
