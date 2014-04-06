package Client;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.rmi.*;

public interface Client extends java.rmi.Remote {

    public byte[] obtain(String file) throws RemoteException;

}