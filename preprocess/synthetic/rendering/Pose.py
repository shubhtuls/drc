import copy

class Pose:
    def __init__(self, poseVars):
        self.varList = poseVars.keys()        
        for x in self.varList:
            setattr(self,x,poseVars[x])
    
    def setVar(self,var,val):
        
        if(var not in self.varList):
            self.varList.append(var)
        setattr(self, var, val)
        return
    
    def getVar(self,varName):
        
        if(varName not in self.varList):
            return []
        else:
            return getattr(self,varName)
    
    def __deepcopy__(self, memo):
        poseVars = {}
        for x in self.varList:
            poseVars[x] = copy.deepcopy(self.getVar(x))
        return Pose(poseVars)
    
    def __str__(self):
        return "{" + ",".join( [ (var + " = " + str(getattr(self,var) )) for var in self.varList ] ) + "}"