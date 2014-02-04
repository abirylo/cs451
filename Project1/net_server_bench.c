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

char mode = 'U';

int createServerUDP(struct sockaddr_in* sock, int port)
{
  int sd = 0;
  if((sd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
  { 
    return -1;
  }
  memset(&sock, 0, sizeof(sock));
  sock->sin_family = AF_INET;
  sock->sin_addr.s_addr = htonl(INADDR_ANY);
  sock->sin_port = htonl(port);

  if(bind(sd, (struct sockaddr *)sock, sizeof(sock)) < 0)
  {
    return -1;
  }

  return sd;  

}

int createServerTCP(struct sockaddr_in* sock, int port)
{
  int sd = 0;
  if((sd = socket(AF_INET, SOCK_DGRAM, IPPROTO_TCP)) < 0)
  { 
    return -1;
  }
  memset(&sock, 0, sizeof(sock));
  sock->sin_family = AF_INET;
  sock->sin_addr.s_addr = htonl(INADDR_ANY);
  sock->sin_port = htonl(port);

  if(bind(sd, (struct sockaddr *)sock, sizeof(sock)) < 0)
  {
    return -1;
  }

  return sd;  

}
void* networkServer(void* arg)
{   
  int sd = 0;
  int cd = 0;
  struct sockaddr_in server, client;
  char buf[100000];  
  int recvSize = 0;

  if(mode == 'U')
  {
    sd = createServerUDP(&server,((int*)arg)[0]);
  }
  else
  {
    sd = createServerTCP(&server,((int*)arg)[0]);
    listen(sd, 5);
  }
  if( sd <= 0)
  {
    return NULL;
  }

  for(;;)
  {
    //wait for client 
    if(mode == 'U')
    {
      if(recvSize = recvfrom(sd,buf,100000,0,(struct sockaddr*)&client, sizeof(client)) < 0)
      {
        return NULL;
      }
      sendto(sd, buf, recvSize, 0, (struct sockaddr*)&client,sizeof(client)); 
    }
    else if(mode == 'T')
    {
      //if((cd = accept()))
        return NULL;
    }
  }
  
  return NULL;
}


int main(int argc, char** argv)
{
    char c;
    int threads = 1;

    while ( (c = getopt(argc, argv, "n:m:") ) != -1) 
    {
        switch (c) 
        {
            case 'n':
                threads = atoi(optarg);
                break;
            case 'm':
                mode = optarg[0];                
            default:
                printf("Usage: ./benchCPU -n [number of threads]\n");
                return -1;
                break;
        }
    }
    
    pthread_t net_threads[threads]; 
    int port = 8000;
    for(int i=0; i<threads; i++)
    {
        pthread_create(&net_threads[i], NULL, networkServer, &port);
        port++;
    }
    
    for(int i=0; i<threads; i++)
    {
        pthread_join(net_threads[i], NULL);
    }
	      
    return 0;
}
