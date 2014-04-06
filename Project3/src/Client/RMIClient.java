package Client;

import java.rmi.*;
import java.rmi.server.*;
import java.rmi.Naming;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.io.*;

import Server.Server;

public class RMIClient extends UnicastRemoteObject implements Client{

    static Integer peerId = 0;
    static String instanceName = null;
    
    public RMIClient() throws RemoteException
    {
        super();
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

        //if (System.getSecurityManager() == null) {
        //    System.setSecurityManager(new RMISecurityManager());
        //}

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

        Server server = null;
        try {
            server = (Server) Naming.lookup("rmi://localhost/Server");
            File[] files = dir.listFiles();
            List<String> listFiles = new ArrayList<String>();
            for(File file:files)
            {
                listFiles.add(file.getName());
            }
            server.registry(peerId, listFiles);
        } catch (Exception e) {
            System.out.println("\nError - Index Server not bound. Please start index server before launching client");
            try {
                Naming.unbind(instanceName);
            } catch (Exception e1) {
                System.out.print("\nError - Server cannot unbound.");
            }
            System.exit(0);
        }

        //UI
        boolean exit = false;
        Scanner scanner = new Scanner(System.in);
        while (!exit)
        {
            System.out.println("Options:");
            System.out.println("1 - Search for filename");
            System.out.println("2 - Obtain filename from peer");
            System.out.println("3 - List files in shared directory");
            System.out.println("4 - Update registry");
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

                    try {
                        peers = server.search(filename);
                    } catch (RemoteException e) {
                        System.out.println("Error - Index Server not bound. Please start index server before launching client");
                    }
                    if(peers == null || peers.size() == 0)
                    {
                        System.out.println("There are no peers sharing " + filename);
                    }
                    else
                    {
                        System.out.println("The clients that have the file " + filename + " are:");
                        for(Integer integer:peers)
                        {
                            System.out.println("Client " + integer);
                        }
                    }
                    break;
                case 2:
                    System.out.print("Enter filename: ");
                    filename = scanner.nextLine();
                    System.out.println("Enter peer. If no preference enter 0.");
                    int peer = Integer.parseInt(scanner.nextLine());
                    try {
                        peers = server.search(filename);
                    } catch (RemoteException e) {
                        e.printStackTrace();
                    }
                    if((peers == null) || (peers.size() <= 0))
                    {
                        System.out.println("The file " + filename + " does not exist.");
                        break;
                    }
                    if(peer == 0)
                    {
                        peer = peers.get(0);
                    }
                    //check if client has file
                    if(!peers.contains(peer))
                    {
                        System.out.println("The peer specified does not have the file " + filename + ".");
                        break;
                    }
                    Client clientPeer;
                    byte[] data = null;
                    try {
                        clientPeer = (Client) Naming.lookup("rmi://localhost/Client"+peer);
                        data = clientPeer.obtain(filename);
                    } catch (Exception e) {
                        System.out.println("Error - Client not bound. The client is not running anymore.");
                        break;
                    }
                    try {
                        FileOutputStream fos = new FileOutputStream(instanceName + "/" + filename);
                        fos.write(data);
                        fos.close();
                    } catch (Exception e) {
                        System.out.println("Error - File can not be created.");
                        System.out.println(e.toString());
                        break;
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
                    try {
                        File[] files = dir.listFiles();
                        List<String> listFiles = new ArrayList<String>();
                        for(File file:files)
                        {
                            listFiles.add(file.getName());
                        }
                        server.registry(peerId, listFiles);
                    } catch (Exception e) {
                        System.out.println("\nError - Cannot update registry");
                    }
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