package Server;

import java.rmi.*;
import java.util.List;
import java.util.HashMap;

public interface Server extends java.rmi.Remote {

    public boolean registry(Integer peerId, List<String> fileNames)throws RemoteException;

	public HashMap getRegistry() throws RemoteException;

    public List search(String filename) throws RemoteException;

}
