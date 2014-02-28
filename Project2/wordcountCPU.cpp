//wordcountCPU.c

#include <map>
#include <iostream>
#include <cstring>
#include <string>
#include <cstdio>
#include "c_tokenizer.h"

using namespace std;

const char* testString = "you?\"\" yoyo?for|me you; you’re \"you you--I\" young <wow></wow> http://www/index.html";
const char* DELIM = "?\";<>~`!@#^&*()_+=/\\:;{}[]|. ";

typedef std::map<std::string, int> TStrIntMap;
typedef std::pair<std::string, int> TStrIntPair;
TStrIntMap wcMap;

void count(const char* s, TStrIntMap* map){
	tokenizer_t tok = tokenizer(s, DELIM, TOKENIZER_NO_EMPTIES);
	const char* t;
	std::string token;
	int value;
	TStrIntMap::iterator iter;

	while(t = tokenize(&tok)){
		token = std::string(t);
		iter = map->find(token);
		if(iter == map->end()){
			map->insert(TStrIntPair(token, 1));
			// std::cout << "Inserting token: " << token << " for the first time." << std::endl;
		}
		else{
			value = (*map)[token];
			value++;
			map->erase(iter);
			map->insert(TStrIntPair(token, value));
			// std::cout << "Updating token: " << token << " to have a value of: " << value << std::endl;
		}
	}
}

int main(int argc, char *argv[])
{	
	int numElements;
	
	wcMap.insert(TStrIntPair("hello",100));
	numElements = count(testString, &wcMap);
	for (TStrIntMap::iterator it=wcMap.begin(); it!=wcMap.end(); ++it){
		std::cout << it->first << " => " << it->second << std::endl;
	}

	return 0;
}
