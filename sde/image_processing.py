import os
import cv2
import numpy as np 
from glob import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(CURRENT_PATH, 'images/train/')
TEST_PATH = os.path.join(CURRENT_PATH, 'images/test/')


def get_files(path):
    img_dict = dict()
    total_count = 0

    for item in glob(path + '*'):
        obj = item.split('/')[-1]
        img_dict[obj] = list()
        for img in glob(path + obj + '/*'):
            img = cv2.imread(img, 0)
            img_dict[obj].append(img)
            total_count += 1

    return img_dict, total_count


class Classifier:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.kmeans_obj = KMeans(n_clusters=n_clusters)
        self.images_after_clustering_features = None
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                 hidden_layer_sizes=(5, 2), random_state=1)

    def stack_up_features(self, l):
        vStack = np.array(l[0])
        for remaining in l:
            vStack = np.vstack((vStack, remaining))
        return vStack

    def cluster(self, descriptor_stack):
        return self.kmeans_obj.fit_predict(descriptor_stack)

    def assign_clusters_to_img(self, n_images, descriptor_list, clusters=None):
        self.images_after_clustering_features = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
        old_count = 0
        for i in range(n_images):
            l = len(descriptor_list[i])
            for j in range(l):
                idx = clusters[old_count + j]
                self.images_after_clustering_features[i][idx] += 1
            old_count += l

    def standardize(self):
        self.scale = StandardScaler().fit(self.images_after_clustering_features)
        self.images_after_clustering_features = self.scale.transform(self.images_after_clustering_features)

    def train(self, train_labels):
        self.clf.fit(self.images_after_clustering_features, train_labels)

    def predict(self, iplist):
        return self.clf.predict(iplist)


class SIFTFeatures:
    def __init__(self):
        self.sift_object = cv2.xfeatures2d.SURF_create()

    def get_features(self, image):
        return self.sift_object.detectAndCompute(image, None)


class ImgProcessor:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.sift_object = SIFTFeatures()
        self.classifier = Classifier(no_clusters)
        self.train_labels = np.array([])
        self.label_dict = {}
        self.descriptor_list = []

    def train_model(self):
        images, images_total_count = get_files(TRAIN_PATH)
        label_count = 0

        for word, imlist in images.iteritems():
            self.label_dict[str(label_count)] = word
            for im in imlist:
                # Assign train labels per image
                self.train_labels = np.append(self.train_labels, label_count)
                # Get feature vector for each image
                _, des = self.sift_object.get_features(im)
                self.descriptor_list.append(des)
            label_count += 1

        # Stack up all features and cluster them.
        descriptor_stack = self.classifier.stack_up_features(self.descriptor_list)
        cluster_assignment_per_feature_row = self.classifier.cluster(descriptor_stack)

        # From clustered features, get clustered image vectors
        self.classifier.assign_clusters_to_img(n_images=images_total_count,
                                           descriptor_list=self.descriptor_list,
                                           clusters=cluster_assignment_per_feature_row)

        # Normalize the clustered image vectors
        self.classifier.standardize()

        # Assigns labels to clustered image vectors
        self.classifier.train(self.train_labels)

        print("Training done")

    def recognize(self, test_img):
        _, des = self.sift_object.get_features(test_img)
        vocab = np.array([0 for i in range(self.no_clusters)])
        test_ret = self.classifier.kmeans_obj.predict(des)
        for each in test_ret:
            vocab[each] += 1

        vocab = self.classifier.scale.transform(vocab)
        lb = self.classifier.clf.predict(vocab)

        return lb

    def test_model(self):
        test_images, _ = get_files(TEST_PATH)
        predicted_result = dict()
        for word, im in test_images.iteritems():
            cl = self.recognize(im[0])
            predicted_result[word] = self.label_dict[str(int(cl[0]))]

        return predicted_result

if __name__ == '__main__':
    processor = ImgProcessor(20)
    processor.train_model()
    processor.test_model()