import pandas as pd
class TreeNode:
    def __init__(self, examples):
        self.examples = examples
        self.left = None
        self.right = None
        self.split_point = None
        self.split_feature = None
    def split(self):
        # len(self.examples)==1 is a non ending case where the left node is always going to contain
        # the remaining element -> infinite loop if we don't handle it here
        if len(self.examples)==1 :
            return None
        df = pd.DataFrame.from_dict(self.examples)
        min_MSE = float("inf")
        for feature in df.columns.values[:-1] : 
            # we sort the examples so as to calculate the average of every two consecutive elements
            # and use it as a feature
            self.examples.sort(key=lambda example : example[feature])
            for i in range(len(self.examples)-1) : 
                split_point = (self.examples[i][feature]+self.examples[i+1][feature])/2
                left_side = df[df[feature]<=split_point]
                right_side = df[df[feature]>split_point]
                # don't make the error of calculating the MSE according to the value of the feature, 
                # compute it using the label values
                MSE = ((((left_side["bpd"]-left_side["bpd"].mean())**2).sum()+((right_side["bpd"]-right_side["bpd"].mean())**2).sum())/(len(left_side["bpd"])+len(right_side["bpd"])))
                if MSE < min_MSE : 
                    self.split_feature = feature
                    min_MSE = MSE
                    self.split_point = split_point
        self.left = TreeNode(df[df[self.split_feature]<=self.split_point].to_dict(orient="records"))
        self.right = TreeNode(df[df[self.split_feature]>self.split_point].to_dict(orient="records"))
        self.left.split()
        self.right.split()
class RegressionTree:
    def __init__(self, examples):
        # Don't change the following two lines of code.
        self.root = TreeNode(examples)
        self.train()

    def train(self):
        # Don't edit this line.
        self.root.split()

    def predict(self, example):
        node = self.root 
        while node.left and node.right : 
            if example[node.split_feature]<=node.split_point : 
                node = node.left
            else : 
                node = node.right
        bpds = list(map(lambda x : x["bpd"], node.examples))
        return sum(bpds)/len(bpds)
        #return self.dfs(self.root,example)
        
    def dfs(self, node,example) : 
        if (node.left==None and node.right==None) :
            print(node.examples)
            bpds = list(map(lambda x : x["bpd"], node.examples))
            return sum(bpds)/len(bpds)
        else : 
            if example[node.split_feature]<node.split_point : 
                return self.dfs(node.left,example)
            else : 
                return self.dfs(node.right,example)