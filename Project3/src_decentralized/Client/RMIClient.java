package Client;

import java.rmi.*;
import java.rmi.server.*;
import java.rmi.Naming;
import java.util.*;
import java.io.*;

public class RMIClient extends UnicastRemoteObject implements Client{

    static Integer peerId = 0;
    static String instanceName = null;
    static List<Client> clientNeighbors = new ArrayList<Client>();
    static List<String> messageList = new LinkedList<String>();
    static List<String> originatingMessages = new LinkedList<String>();
    static Map<String,List<String>> messageResponse = new HashMap<String,List<String>>();
	static Map<String,List<String>> fileLocation = new HashMap<String,List<String>>();
    static final int TTL = 15;
    
    //make as a hashmap???
    //static List<Message> messageList = new LinkList<Message>();
	
    public RMIClient() throws RemoteException
    {
        super();
    }

    @Override
    public int query(String messageID, int ttl, String fileName) throws RemoteException
    {
        synchronized (this) {
            //add message to list if it's the first time seeing the new message
            if(!messageList.contains(messageID) && !originatingMessages.contains(messageID))
            {
                messageList.add(messageID);
            }
            else
            {
                return 0;
            }
        }            
        
        int queryAmount = 0;        
        //pass message to neighbors
        ttl--;
        if(ttl != 0)
        {
            for(Client client: clientNeighbors)
            {
                try {
                    queryAmount += client.query(messageID, ttl, fileName);
                } catch(Exception e){
                    System.out.println("Can not pass query to neighbor.");
                }
            }
        }
        
        //find if file exists locally
        File listDir = new File(instanceName);
        File[] sharedFiles = listDir.listFiles();
        for(File file:sharedFiles)
        {
            if(file.getName().equals(fileName))
            {
                queryAmount++;
                for(Client client: clientNeighbors)
                {
                    client.hitQuery(messageID, TTL, fileName, "Client"+peerId);
                }
                break;
            }
        }
        return queryAmount;
    }
	
    @Override
    public void hitQuery(String messageID, int ttl, String fileName, String havePeerID) throws RemoteException
    {
            //record message response 
            if(messageResponse.containsKey(messageID))
            {
                if(messageResponse.containsKey(messageID))
                {
                    List<String> peerList = messageResponse.get(messageID);
                    if(peerList != null)
                    {
                        peerList.add(havePeerID);
                    }
                    else
                    {
                        peerList = new LinkedList<String>();
                        peerList.add(havePeerID);
                    }
                }
                else
                {
                    List<String> tempList = new LinkedList<String>();
                    tempList.add(havePeerID);
                    messageResponse.put(messageID, tempList);
                }
            }
            else
            {
                List<String> tempList = new LinkedList<String>();
                tempList.add(havePeerID);
                messageResponse.put(messageID, tempList);
            }
            
            //check if messageID belongs to this client
            if(originatingMessages.contains(messageID))
            {
                if(fileLocation.containsKey(fileName))
                {
                    List<String> peerList = fileLocation.get(fileName);
                    if(peerList != null)
                    {
                        if(!peerList.contains(havePeerID))
                        {
                            peerList.add(havePeerID);
                        }
                    }
                    else
                    {
                        peerList = new LinkedList<String>();
                        peerList.add(havePeerID);
                    }
                }
                else
                {
                    List<String> tempList = new LinkedList<String>();
                    tempList.add(havePeerID);
                    fileLocation.put(fileName, tempList);
                }
            }
    }
    
    @Override
    public byte[] obtain(String file) throws RemoteException
    {
        RandomAccessFile randomAccessFile = null;
        byte[] b = null;
        try {
            randomAccessFile = new RandomAccessFile(instanceName + "/" + file, "r");
            b = new byte[(int)randomAccessFile.length()];
            randomAccessFile.read(b);
            return b;
        } catch (Exception e) {
            System.out.println("Error - Error with File: " + file);
            System.out.println(e.toString());
        }
        return b;
    }
    
    public static void main(String[] args)
    {
        if (args.length == 0) {
            System.out.println("Please enter a peer id as a command line argument\n");
            System.exit(1);
        }

        try {
            peerId = Integer.parseInt(args[0]);
        } catch (NumberFormatException e) {
            System.err.println("Argument must be an integer");
            System.exit(1);
        }

        instanceName = "Client" + peerId;
        RMIClient client;

        try {
            client = new RMIClient();
            Naming.bind(instanceName, client);
        } catch (Exception e) {
            System.out.println("\nError - Peer " + peerId + " is already bound.  Please choose a different peer id or restart rmiregistry");
            System.exit(2);
        }

        System.out.println("Client running... PeerID = " + peerId);

        //get file names
        //use instance name as the folder containing the files to share
        File dir = new File(instanceName);
        if (!dir.exists()) {
            System.out.println("Creating new shared directory");
            dir.mkdir();
        }
        
        //connect to neighbors 
        //Find config file in in ./Config/"instanceName"
        File configFile = new File("./Config/" + instanceName + ".config");
        Scanner scan = null;
        try{
             scan = new Scanner(configFile);
        } catch (Exception e){
            System.out.println("Config file in ./Config/" + instanceName + ".config" + " Not Found.");
            System.exit(2);
        }
        while(scan.hasNextLine())
        {
            boolean neighborConnected = false;
            String neighborName = scan.nextLine();
            while(!neighborConnected)
            {
                try {
                    Client tempClient = (Client) Naming.lookup("rmi://localhost/" + neighborName);
                    clientNeighbors.add(tempClient);
                    neighborConnected = true;
                } catch (Exception e) {
                    try {
                        Thread.sleep(500); //sleep 500ms and try again
                    } catch (Exception f) {
                        System.out.println("Error with sleeping. Process will exit.");
                        System.exit(2);
                    }  
                }
            }
        }

        //UI
        boolean exit = false;
        Scanner scanner = new Scanner(System.in);
		int squenceNumber = 0;
        while (!exit)
        {
            System.out.println("Options:");
            System.out.println("1 - Search for filename");
            System.out.println("2 - Obtain filename from peer");
            System.out.println("3 - List files in shared directory");
            System.out.println("4 - Run Performance Test");
            System.out.println("5 - Exit");
            System.out.print(">");

            String line = scanner.nextLine();
            String filename = null;
            int option;
            try{
                option = Integer.parseInt(line);
            } catch (Exception e){
                System.out.println("Error - The command entered was not a number. Please enter a number.");
                continue;
            }
            List<Integer> peers = new ArrayList<Integer>();

            switch (option)
            {
                case 1:
                    System.out.print("Enter filename: ");
                    filename = scanner.nextLine();
                    System.out.println();
                    
                    
					//send query to nieghbors
					int queryAmount = 0;
					String messageId = "Client"+peerId+","+"Sequence"+squenceNumber;
                    originatingMessages.add(messageId); //add this since this is the root node
                    fileLocation.remove(filename);  //clear the location of files since the old ones may have been expired
					for(Client neighbor : clientNeighbors)
					{
                        try {
                            queryAmount += neighbor.query(messageId, TTL, filename);
                        } catch(Exception e){
                            System.out.println("Can not pass query to neighbor.");
                        }
					}
					squenceNumber++;
					
					boolean done = false;
					int retryCount = 0;
					while(!done)
					{
						List<String> peerList = messageResponse.get(messageId);
						if(peerList != null && peerList.size() >= queryAmount)
						{
							//print out peers
                            for(String peer: peerList)
                            {
                                System.out.println(peer);
                            }
							done = true;
						}
						else
                        {
							retryCount++;
							if(retryCount > 10)
                            {
                                //not all peers return print what is left if anything
                                if(peerList != null)
                                {
                                    for(String peer: peerList)
                                    {
                                        System.out.println(peer);
                                    }
                                }    
                                done = true;
							}
                            else
                            {
                                try {
                                    Thread.sleep(500);
                                } catch (Exception f) {
                                    System.out.println("Error with sleeping. Process will exit.");
                                }
                            }                                
						}
					}
                    break;
                case 2:
                    System.out.print("Enter filename: ");
                    filename = scanner.nextLine();
                    System.out.println("Enter peer. If no preference enter 0.");
                    int peer = Integer.parseInt(scanner.nextLine());
                    String peerString = null;
                    if(peer == 0)
                    {
                        List<String> peerList = fileLocation.get(filename);
                        if(peerList != null && peerList.size() > 0)
                        {
                            peerString = peerList.get(0);
                        }
                        else
                        {
                            System.out.println("There is no know peer with the file: "+filename);
                            break;
                        }
                    }
                    else
                    {
                        peerString = "Client"+peer;
                    }  
                    Client clientPeer;
                    byte[] data = null;
                    try {
                        clientPeer = (Client) Naming.lookup("rmi://localhost/"+peerString);
                        //check if peer has file before doing download?
                        //**********************************************
                        //**********************************************
                        data = clientPeer.obtain(filename);
                    } catch (Exception e) {
                        System.out.println("Error - Client not bound. The client is not running anymore.");
                        //try different host?
                        break;
                    }
                    if(data != null && data.length >= 0)
                    {
                        try {
                            FileOutputStream fos = new FileOutputStream(instanceName + "/" + filename);
                            fos.write(data);
                            fos.close();
                        } catch (Exception e) {
                            System.out.println("Error - File can not be created.");
                            System.out.println(e.toString());
                            break;
                        }
                    }
                    else
                    {
                        System.out.println("The peer selected did not have the file or the file was empty.");
                    }
                    break;
                case 3:
                    File listDir = new File(instanceName);
                    File[] sharedFiles = listDir.listFiles();
                    System.out.println("Files in the shared directory:");
                    for(File file:sharedFiles)
                    {
                        System.out.println(file.getName());
                    }
                    break;
                case 4:
                    //need to write test cases
					//**********************************************
					//**********************************************
                    break;
                case 5:
                    exit = true;
                    break;
                default:
                    System.out.println("Unknown Command");
                    break;
            }
        }

        try {
            Naming.unbind(instanceName);
        } catch (Exception e) {
            System.out.println("Error - Server cannot unbound.");
            System.exit(1);
        }

        System.exit(0);
    }
}