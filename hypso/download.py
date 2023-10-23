def download_directory(usereu,passeu,hostname="129.241.2.147",download_list=[],local_path=None):
    from stat import S_ISDIR, S_ISREG
    import os
    from urllib.parse import urlparse
    import pysftp
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None

    with pysftp.Connection(host = hostname, username = usereu, password = passeu, cnopts = cnopts) as sftp:

        def get_r_portable(sftp, remotedir, localdir, preserve_mtime=False):
            for entry in sftp.listdir(remotedir):
                if entry != "database-folder":
                    remotepath = remotedir + "/" + entry
                    localpath = os.path.join(localdir, entry)
                    mode = sftp.stat(remotepath).st_mode
                    if S_ISDIR(mode):
                        try:
                            os.mkdir(localpath)
                        except OSError:     
                            pass
                        get_r_portable(sftp, remotepath, localpath, preserve_mtime)
                    elif S_ISREG(mode):
                        sftp.get(remotepath, localpath, preserve_mtime=preserve_mtime)


        for f in download_list:

            rem_path=os.path.join("/home/hypso/hypso-1_data/processed",urlparse(f).path[1:-1])
            print(rem_path)
            
            local_sup_path=os.path.join(local_path,rem_path.split("/")[-1])
            os.mkdir(local_sup_path)

        
            get_r_portable(sftp, rem_path, local_sup_path, preserve_mtime=False)
