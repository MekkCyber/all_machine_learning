import math
class MultinomialNB:
    def __init__(self, articles_per_tag):
        # Don't change the following two lines of code.
        self.likelihoods = {}
        self.priors = {}
        self.words = set()
        self.alpha = 1
        self.articles_per_tag = articles_per_tag 
        self.train()
        
    def train(self):
        num_articles = 0
        flattened = {}
        for tag in self.articles_per_tag : 
            self.priors[tag] = len(self.articles_per_tag[tag])
            flattened[tag] = [item for sublist in self.articles_per_tag[tag] for item in sublist]
            num_articles += len(self.articles_per_tag[tag])
            for document in self.articles_per_tag[tag] : 
                self.words = self.words.union(set(document))
        self.priors = {k:v/num_articles for k,v in self.priors.items()}
        for word in self.words : 
            for tag in self.articles_per_tag :
                self.likelihoods[(word,tag)] = (flattened[tag].count(word)+self.alpha)/(len(flattened[tag])+2*self.alpha)
            
    def predict(self, article):
        posteriors = {}
        for word in article : 
            likelihood = 0
            for tag in self.articles_per_tag : 
                if word not in self.words : 
                    likelihood = 1/2
                else :
                    likelihood = self.likelihoods[(word,tag)]
                if tag in posteriors :
                    posteriors[tag] += math.log(likelihood)
                else : 
                    posteriors[tag] = math.log(likelihood*self.priors[tag])
        return posteriors