import java.rmi.*;
import java.util.*;
/**
** List the names and objects bound in RMI registries.
** Invoke with zero or more registry URLs as command-line arguments,
** e.g. "rmi://localhost", "rmi://localhost:1099".
*/
public class ListRegistry
{
	public static void main(String[] args)
	{
		//System.setSecurityManager(new RMISecurityManager());
		for (int i = 0; i < args.length; i++)
		{
			try
			{
				String[]list = Naming.list(args[i]);
				System.out.println("Contents of registry at "+args[i]);
				for (int j = 0; j < list.length; j++)
				{
					Remote remote = Naming.lookup(list[j]);
					System.out.println((j+1) + ".\tname=" + list[j] + "\n\tremote=" + remote);
				}
			}
			catch (java.net.MalformedURLException e)
			{
				System.err.println(e); // bad argument
			}
			catch (NotBoundException e)
			{
				// name vanished between list and lookup - ignore
			}
			catch (RemoteException e)
			{
				System.err.println(e); // General RMI exception
			}
		}
	}
}
