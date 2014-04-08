package Server;

import java.rmi.*;
import java.rmi.server.*;
import java.util.*;

public class RMIServer extends UnicastRemoteObject implements Server {

    HashMap<String, List<Integer>> registry = new HashMap<String, List<Integer>>();

    public RMIServer() throws RemoteException
    {
        super();
    }

    @Override
    public boolean registry(Integer peerId, List<String> fileNames) throws RemoteException
    {
        //remove all listing for peerId
        Iterator it = registry.entrySet().iterator();
        while(it.hasNext())
        {
            Map.Entry pairs = (Map.Entry)it.next();
            List<Integer> l = (List<Integer>)pairs.getValue();
            if(l.contains(peerId))
            {
                    l.remove(peerId);
            }        
        }
        
        for(String file:fileNames)
        {
            //check if file is in map
            if(registry.containsKey(file))
            {
                //check if pereId is in list
                if(!registry.get(file).contains(peerId))
                {
                    List<Integer> list = registry.remove(file);
                    list.add(peerId);
                    registry.put(file,list);
                }
            }
            else
            {
                List<Integer> list = new ArrayList<Integer>();
                list.add(peerId);
                registry.put(file,list);
            }
        }

        return true;
    }

	@Override
	public HashMap<String, List<Integer>> getRegistry() throws RemoteException{
		return registry;
	}

    @Override
    public List search(String filename) throws RemoteException
    {
        return registry.get(filename);
    }

    public static void main(String[] args)
    {
        //if(System.getSecurityManager() == null)
        //{
        //    System.setSecurityManager(new RMISecurityManager());
        //}

        RMIServer server;

        try {
            server = new RMIServer();
            Naming.bind("Server", server);
        } catch (Exception e) {
			System.out.println(e);
            System.out.println("Error - Server is already bound.");
            System.exit(0);
        }

        System.out.println("Server is running");
        System.out.println("To exit type 'exit'.");

        boolean exit=false;
        Scanner scanner = new Scanner(System.in);
        while(!exit)
        {
            String line = scanner.nextLine();
            if(line.equals("exit"))
            {
                exit = true;
            }
        }

        try {
            Naming.unbind("Server");
        } catch (Exception e) {
            System.out.println("Error - Server cannot unbound.");
        }

        System.exit(0);
    }
}
