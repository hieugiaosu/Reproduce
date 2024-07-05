from typing import Any


class CacheTensor:
    def __init__(self,cacheSize:int,miss_handler) -> None:
        self.cacheSize = cacheSize
        self.miss_handler = miss_handler

        self.keys = []
        self.cache = {}
    
    def __readFile(self,fileName):
        data = None
        try: 
            data = self.cache[fileName]
        except:
            data = self.miss_handler(fileName)
            if len(self.cache.keys()) < self.cacheSize:
                self.cache[fileName] = data
            else: 
                deleted_key = self.keys[0]
                self.cache.pop(deleted_key)
                self.keys.pop(0)
                self.cache[fileName] = data
            self.keys.append(fileName)
        return data
    
    def __call__(self, fileName):
        return self.__readFile(fileName)