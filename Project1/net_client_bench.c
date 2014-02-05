//net_server_bench.c

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <string.h>
#include <netinet/in.h>
#include <wchar.h>

unsigned long MAX_OPS = 2000000;
char mode = 'U';
char test = 'B';
int bufferSize = 1;
int threads = 1;

int createClientUDP(struct sockaddr_in* sock, int port, char* ip)
{
  int sd = 0;
  struct sockaddr_in temp;
  if((sd = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
  { 
    return -1;
  }
  memset(sock, 0, sizeof(temp));
  sock->sin_family = AF_INET;
  sock->sin_addr.s_addr = inet_addr(ip);
  sock->sin_port = htons(port);

  return sd;  

}

int createClientTCP(struct sockaddr_in* sock, int port)
{
  int sd = 0;
  struct sockaddr_in temp;

  if((sd = socket(AF_INET, SOCK_DGRAM, IPPROTO_TCP)) < 0)
  { 
    return -1;
  }
  memset(&sock, 0, sizeof(sock));
  sock->sin_family = AF_INET;
  sock->sin_addr.s_addr = htonl(INADDR_ANY);
  sock->sin_port = htonl(port);

  if(bind(sd, (struct sockaddr *)sock, sizeof(temp)) < 0)
  {
    return -1;
  }

  return sd;  

}
void* networkClient(void* arg)
{   
  int sd = 0;
  struct sockaddr_in server, client;
  char buf[bufferSize];  
  int recvSize = 0;
  unsigned int socklen = sizeof(client);
  char ip[] = "127.0.0.1";
  struct timeval tv;
  long long start, stop;
  double secs;

//  memset(buf, 0x55, bufferSize);

  if(mode == 'U')
  {
    sd = createClientUDP(&client,((int*)arg)[0], ip);
  }
  else
  {
    sd = createClientTCP(&client,((int*)arg)[0]);
  }
  if( sd <= 0)
  {
    return NULL;
  }
  
  if(mode == 'U')
  {
    gettimeofday(&tv, NULL);
    start = tv.tv_sec*1000000LL + tv.tv_usec;
    for(int i=0; i<MAX_OPS; i++)
    {
      sendto(sd, buf, bufferSize, 0, (struct sockaddr*)&client,sizeof(client)); 
      if((recvSize = recvfrom(sd, buf, bufferSize, 0, (struct sockaddr*)&server,&socklen)) != bufferSize)
      {
        printf("Return value is different then sent. ERROR\n");
      } 
    
    }
    gettimeofday(&tv, NULL);
    stop = tv.tv_sec*1000000LL + tv.tv_usec;
    secs = (stop-start)/1000000.0;
    printf("%i\t%i\t%c\t%lf\t%lf\n", threads, bufferSize, mode, (bufferSize*MAX_OPS)/(secs*1048576),(secs*1000000)/(MAX_OPS*2));
 
  }
  else if(mode == 'T')
  {
    //if((cd = accept()))
      return NULL;
  }
  
  
  return NULL;
}


int main(int argc, char** argv)
{
    char c;

    while ( (c = getopt(argc, argv, "n:l:m:t:") ) != -1) 
    {
        switch (c) 
        {
            case 'n':
                threads = atoi(optarg);
                break;
            case 'l':
                MAX_OPS = atol(optarg);
                break;
            case 'm':
                mode = optarg[0];
                break;
            case 't':
                test = optarg[0];
                if(test == 'B')
                  bufferSize = 1;
                else if(test == 'K')
                  bufferSize = 1024;
                else if(test == '6')
                  bufferSize = 64000;
                break;                
            default:
                printf("Usage: ./net_server_bench -n [number of threads] -l [number of loops] -m [mode U(UDP) T(TCP)] -t [test size B(byte) K(kilobyte) 6(64KB))\n");
                return -1;
                break;
        }
    }
    
    pthread_t net_threads[threads]; 
    int port[2] = {8000, 8001};
    for(int i=0; i<threads; i++)
    {
        pthread_create(&net_threads[i], NULL, networkClient, &(port[i]));
    }
    
    for(int i=0; i<threads; i++)
    {
        pthread_join(net_threads[i], NULL);
    }
	      
    return 0;
}
