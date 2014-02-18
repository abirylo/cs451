//wordcountCPU.c

#include <map>
#include <string>
#include <cstdio>

using namespace std;

typedef std::map<std::string, int> TStrIntMap;
typedef std::pair<std::string, int> TStrIntPair;
TStrIntMap wcMap;

int main(int argc, char *argv[])
{

  wcMap.insert(TStrIntPair("hello",100));
   
  int intValue = wcMap["hello"];
        
  printf("hello: %i\n", intValue);
            
}
