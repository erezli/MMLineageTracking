import os
from glob import glob


class LineageTrack:
    def __init__(self, filepath, *args):
        if os.path.isdir(filepath):
            self.directory = filepath
            self.files = glob(self.directory + "*")
        elif os.path.exists(filepath):
            self.files = [filepath]
        else:
            print("error")
        for path in args:
            if not os.path.exists(path):
                print("error")
            else:
                self.files.append(path)
        print(self.files)
