class Plp():
    def __init__(self, path):
        import subprocess
        run = "gnome-terminal -e " + path
        subprocess.Popen(run, shell=True)
        print run