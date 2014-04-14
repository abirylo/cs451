package Client;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.rmi.*;

public interface Client extends java.rmi.Remote {

    public int query(String messageID, int ttl, String fileName) throws RemoteException;
    
    public void hitQuery(String messageID, int ttl, String fileName, String havePeerID) throws RemoteException;
    
    public byte[] obtain(String fileName) throws RemoteException;
}