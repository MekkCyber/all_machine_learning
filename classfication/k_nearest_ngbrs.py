def predict_label(examples, features, k, label_key="is_intrusive"):
    ngbrs = find_k_nearest_neighbors(examples, features, k)
    counter = sum([examples[example][label_key] for example in ngbrs])
    if counter*2<k : 
        return 0
    else : 
        return 1

def find_k_nearest_neighbors(examples, features, k):
    distances = []
    for example in examples : 
        ngbr_features = examples[example]["features"]
        distance = 0
        for i in range(len(features)) : 
            distance += (features[i]-ngbr_features[i])**2
        distances.append((distance,example))
    distances.sort()
    distances = list(map(lambda pair : pair[1],distances))
    return distances[:k]