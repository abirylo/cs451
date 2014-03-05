//wordcountCPU.c

#include <map>
#include <iostream>
#include <cstring>
#include <string>
#include <cctype>
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <vector>
#include <thread>
#include "c_tokenizer.h"
#include <fstream>

using namespace std;

typedef std::map<std::string, int> TStrIntMap;
typedef std::pair<std::string, int> TStrIntPair;

//Hooray Global Variables!
TStrIntMap wcMap;
const char* DELIM = "?\";<>,~`!@#^&*()_+-=/\\:;{}[]|. ";
ifstream myFile;
const int BUFF_SIZE = 4096; //max number of lines for each thread to take from the file.
std::mutex mapMtx;
std::mutex fileMtx;
int loopCounter = 0;

void count(std::string s, TStrIntMap* map){
	tokenizer_t tok = tokenizer(s.c_str(), DELIM, TOKENIZER_NO_EMPTIES);
	const char* t;
	std::string token;
	int value;
	TStrIntMap::iterator iter;

	while(t = tokenize(&tok)){
		token = std::string(t);
		std::transform(token.begin(), token.end(), token.begin(), ::tolower); //make sure the word is lowercase
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

void mergeMaps(TStrIntMap* addMap){
	TStrIntMap::iterator iter;
	int value;

	for (TStrIntMap::iterator it=addMap->begin(); it!=addMap->end(); ++it){
		iter = wcMap.find(it->first);
		if(iter == wcMap.end()){
			wcMap.insert(TStrIntPair(it->first, it->second));
		} else {
			value = wcMap[it->first];
			value += it->second;
			wcMap.erase(iter);
			wcMap.insert(TStrIntPair(it->first, value));
		}
	}
}

int getBuff(std::string* s){
	int n=0;
	std::string tempStr;
	if(myFile.is_open()){
		if(myFile.eof()){
			myFile.close();
			return 1;
		} else {
			while((getline(myFile,tempStr)) && (n<BUFF_SIZE)){
				s->append(tempStr);
				s->append(" ");
				n++;
			}
		}
	}else{ return 1; }
}

void t_process(){
	std::string buff;
	TStrIntMap smallMap;
	int fileClosed;
	
	do{
		fileMtx.lock();
		fileClosed = getBuff(&buff);
		//std::cout << loopCounter << std::endl;
		loopCounter++;
		fileMtx.unlock();

		if(fileClosed == 1) return;

		count(buff, &smallMap);
	
		mapMtx.lock();
		mergeMaps(&smallMap);
		mapMtx.unlock();
		
		buff.clear();
		smallMap.clear();
	}while(fileClosed != 1);
}

bool compByValue(TStrIntPair i, TStrIntPair j){
	return ((i.second)>(j.second));
}

int main(int argc, char *argv[])
{	
	int numThreads = 1;
	std::vector<std::thread> threads;
	unsigned totalWords = 0;
	std::vector<TStrIntPair> mapValues;
	int file_arg = 2;

	numThreads = atoi(argv[1]);

	myFile.open(argv[file_arg]);
	if(!(myFile.good())){
		std::cout << "Error opening file." << std::endl;
		return 1;
	}
	while(myFile.is_open()){
		threads.clear();
		for(int i=0; i<numThreads; i++){
			threads.push_back(std::thread(t_process));
		}
		for(std::vector<std::thread>::iterator it = threads.begin(); it != threads.end(); it++){
			it->join();
		}
	}

	ofstream outFile("tempCount.out", std::ofstream::out);
	for(TStrIntMap::iterator it=wcMap.begin(); it!=wcMap.end(); it++){
		mapValues.push_back(*it);
		//outFile << it->first << " => " << it->second << std::endl;
		totalWords += it->second;
	}
	std::sort(mapValues.begin(), mapValues.end(), compByValue);
	for(std::vector<TStrIntPair>::iterator it=mapValues.begin(); it!=mapValues.end(); it++){
		outFile << it->first << "\t" << it->second << std::endl;
	}
	outFile.close();

	std::cout << "Number of file access/mapmerges: " << loopCounter << std::endl;
	std::cout << "Total words counted: " << totalWords << std::endl;

	return 0;
}
