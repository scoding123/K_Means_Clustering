import numpy as np
import random
'''
def eucl_dist(a, b):
    sub = np.subtract(a,b)
    return np.inner(sub,sub)
'''
def eucl_dist(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.

    
    :return: pi.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    centers=[]
    idx = generator.randint(n)
    print('n',n)
    print('size',x.size)
    print('index',idx)
    print('idx',x[idx])
    print('shape',x.shape)
    c1 = x[idx]
    #c1  = generator.choice(x, size=1)
    print('c1',c1)
    d_square = 0
    for i in x:
        d_square += (np.linalg.norm(c1-i)**2)
        
    c = c1
    cen = []
    #cen.append(c1[0])
    cen.append(c1)
    
    max_c = -99999999
    for i in range(n_cluster-1):
        
        
       # d2 = np.array([min([np.square(eucl_dist(i,c, None)) for c in cen]) for i in x])
       # prob = d2/d2.sum()
        
        #print('c:',c)
        
        for j in x:
            dist = (np.linalg.norm(c - j)**2)
            prob = float(dist)/float(d_square)
            
            if prob > max_c:
                max_c = prob
                temp = j
        
        c = temp
        cen.append(c)
    
    cen_temp = []
    for i in cen:
       cen_temp.append(i.tolist()) 
    x = x.tolist()   
    
    #print('cen',cen_temp)
    
    for i in cen_temp:
       
        centers.append(x.index(i))
 
    # DO NOT CHANGE CODE BELOW THIS LINE
    print('centers:',centers)
    #print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

        
        
        
    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape
        

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        
        #initial_centroid_idx = set()
        
        
        '''
        mu_k = x[np.random.choice(N, self.n_cluster, replace=False),:]
        iteration = 1
        J = np.inf
        converged = False
        number_of_updates = np.int(0)
        
        #- repeat
        while iteration < self.max_iter:
            R = np.zeros((N,self.n_cluster))
            cluster_assignment = []
            distance_matrix = []
            for i in range(N):
                distance_k = []
                for j in range(self.n_cluster):
                    distance_k.append(eucl_dist(x[i],mu_k[j])) 
                distance_matrix.append(distance_k)
                cluster = np.argmin(distance_k)
                cluster_assignment.append(cluster)
                R[i,cluster] = 1
            
            distance_ik = np.array(distance_matrix)
            cluster_assignment = np.array(cluster_assignment)
            J_new = np.sum(np.multiply(R,distance_ik))/N
            
            if np.abs(J - J_new) < self.e:
                break
                
            
            J = J_new 
            
            mu_k = np.zeros((self.n_cluster,D))
            for l in range(self.n_cluster):
                mu_k[l,:] = np.sum(x[cluster_assignment == l,],axis = 0)/np.sum(cluster_assignment == l)
            
            iteration = iteration +1 
            number_of_updates = number_of_updates + 1
        
        centroids = mu_k
        y = cluster_assignment
        self.max_iter = number_of_updates
        '''
        initial_centroids = x[list(self.centers), :].copy()
        
        centroids = initial_centroids
        clusters = np.zeros(N)

        update = 0
        J_previous = 0
        itr = 0
        max_itr = self.max_iter
        while itr < max_itr:
        #for itr in range(self.max_iter):
            # clusters = [[] for _ in range(self.n_cluster)]
            # Get clusters first
            dists = []
            for c in centroids:
                squared_with_c = np.square(c - x)
                dist = np.sum(squared_with_c, axis=1)
                #dist = np.sum(np.square(c - x), axis=1)
                dists.append(dist)
            dists = np.asarray(dists)
            #assert dists.shape == (self.n_cluster, N)
            clusters = np.argmin(dists, axis=0)

            # Calculate J
            dists = []
            k = 0
            num_cluster = self.n_cluster
            while k < num_cluster:
            #for k in range(self.n_cluster):
                # TODO change it to norm
                var = np.where(clusters == k)
                membership = x[var]
                squared_dist = np.square(centroids[k, :] - membership)
                dist = np.sum(squared_dist)
                #dist = np.linalg.norm(centroids[k, :] - membership)
                #dist = np.sum(np.square(centroids[k, :] - membership))
                dists.append(dist)
                k += 1
            
            delta_J = np.abs((np.sum(dists) / N) - J_previous) 
            J_previous = (np.sum(dists) / N)

            if delta_J < self.e and itr >= 1:
                break

            # Calculate new centroids
            
            i = 0
            num_clust = self.n_cluster
            while i < num_clust:
                
                cond_var = np.where(clusters == i)
                cluster = []
                for item in x[cond_var]: cluster.append(item)
                # TODO: replace with np.mean
                cluster = np.asarray(cluster)
                if cluster.shape[0] != 0:
                    sum_var  = np.sum(cluster, axis=0)
                    mean = sum_var / cluster.shape[0]
                    centroids[i, :] = mean
                i = i+ 1    
            update += 1
            itr +=1
        #return mu_k, cluster_assignment, number_of_updates 
        #raise Exception(
             #'Implement fit function in KMeans class')
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        y = clusters
        self.max_iter = update
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
             #'Implement fit function in KMeansClassifier class')
        '''
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, i = k_means.fit(x)
        centroid_labels = []
        
        for cluster in range(self.n_cluster):
            # its member
            sub_y = y[membership == cluster]
            (_, idx, counts) = np.unique(sub_y, return_index=True, return_counts=True)
            index = idx[np.argmax(counts)]
            mode = sub_y[index]
            centroid_labels.append(mode)
        
        centroid_labels = np.asarray(centroid_labels, dtype=np.float32)
        centroids = np.asarray(centroids, dtype=np.float32)
        '''
        
        
        classifier_list=[]
        
        centroid_labels=[]
        k = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, z, rp = k.fit(x)
        
        counter1 = len(centroids)
        i = 0
        while i < counter1:
        #for counter in range(len(centroids)):
            var_array = {}
            #var1 = np.unique(y)
            var1 = list(set(y))
            var_array = {t:0 for t in var1}
            #for t in var1:
            #    var_array[t]=0
            counter2 = len(x)
            j = 0 
            while j < counter2:
                
            #for u in range(len(x)):
                if i == z[j]: 
                    var_array[y[j]] = var_array[y[j]]+1
                j = j+1
            #ab=sorted(var_array,key=var_array.get,reverse=True)[0]
         
        
            classifier_list.append(sorted(var_array.items(),key=lambda x:x[1],reverse=True)[0][0])
            i += 1
        centroid_labels = np.asarray(classifier_list)
        
        '''
        kmeans_cluster = KMeans(self.n_cluster, self.max_iter, self.e)
        centroids, cluster, update = kmeans_cluster.fit(x)
        centroid_labels = []
        for k in range(self.n_cluster):
            members = y[np.where(cluster == k)]
            votes = dict()
            for c in members:
                votes[c] = votes.get(c, 0) + 1
            centroid_labels.append(sorted(votes.items(), key=lambda x:(-x[1], x[0]))[0][0])
        centroid_labels = np.array(centroid_labels)
        '''
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #distance = self.distance
        '''
        pred = []
        for m in range(N):
            distance_obs = []
            for n in range(self.n_cluster):
                distance_obs.append(eucl_dist(x[m,:],self.centroids[n,:]))
            pred.append(self.centroid_labels[np.argmin(distance_obs)])
        
        labels = pred
        '''
        
       
        r=[]
        counter1 = len(x)
        i = 0
        while i < counter1:
        
     
            pred1=[]
            counter2 = len(self.centroids)
            j = 0
            while j < counter2:
                
               
                norm_var = x[i]-self.centroids[j]
                distance_var = np.linalg.norm(norm_var)
                
                pred1.append(distance_var)
                j = j+1
           
            r.append(self.centroid_labels[np.argmin(np.array(pred1))])
            i = i+1
       
        labels = np.array(r) 
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function
    print('image shape',image.shape)
    print('code_vectors',code_vectors.shape)
    
    '''
    N, M, C = image.shape
    data = image.reshape(N * M, C)
    l2 = np.sum(((data - np.expand_dims(code_vectors, axis=1)) ** 2), axis=2)
    r = np.argmin(l2, axis=0)
    new_im = code_vectors[r].reshape(N, M, C)
    '''
    
    new_im = np.zeros_like(image)
    counter1 = image.shape[0]
    counter2 = image.shape[1]
    i = 0
    j = 0
   
    while i < counter1:
        while j < counter2:
        
            d = image[i, j, :]
            var1 = np.square(d - code_vectors)
            dists = np.sum(var1, axis=1)
            min_dist = np.argmin(dists)
            centroid_vec = code_vectors[min_dist, :]
            new_im[i, j, :] = centroid_vec
            j = j + 1
        i = i+1
    
    # DONOT CHANGE CODE ABOVE THIS LINE
    
    #raise Exception(
             #'Implement transform_image function')
    

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im


