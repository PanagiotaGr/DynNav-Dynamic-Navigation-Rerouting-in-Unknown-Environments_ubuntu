import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/panagiotagrosd/DynNav-Dynamic-Navigation-Rerouting-in-Unknown-Environments_ubuntu/install/nav_research'
