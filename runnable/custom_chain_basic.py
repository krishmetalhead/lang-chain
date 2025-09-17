from abc import ABC, abstractmethod
class Runnable:
    def __init__(self):
        pass

    @abstractmethod
    def invoke(item):
        pass
    
class CustomA(Runnable):
    def __init__(self,dictionary):
        self.dictionary=dictionary

    def invoke(self,item):
        print('Inside Custom A')
        return self.dictionary[item]

    

class CustomB(Runnable):
    def __init__(self,item):
        self.item = item

    def invoke(self,item):
        print('Inside Custom B' , item)
        return item

class CustomC(Runnable):
    def __init__(self,item):
        self.item = item

    def invoke(self,item):
        print('Inside Custom C  --> final output is  --> Hello' , item)
        return item
    
class Chain(Runnable):
    def __init__(self,listCustoms, item):
        #print('Here')
        self.listCustoms = listCustoms 
        self.item = item       
    
    def invoke(self):
        print('Here')
        for item in self.listCustoms:
            result = item.invoke(self.item)
            self.item = result
        return result
    
class Test():
  def __init__(self):
      info = {"1": "Alice", "2": "KB"} 
      customA = CustomA(info)
      customB = CustomB("2")
      customC = CustomC(info)
      classList = [customB,customA,customC]
      chain = Chain(classList, "2")
      result = chain.invoke()
  
a = Test()

  
       
       

    
      
 
       





