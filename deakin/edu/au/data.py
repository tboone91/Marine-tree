# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021.  Mohamed Reda Bouadjenek, Deakin University                       +
#               Email:  reda.bouadjenek@deakin.edu.au                                    +
#                                                                                        +
#       Licensed under the Apache License, Version 2.0 (the "License");                  +
#       you may not use this file except in compliance with the License.                 +
#       You may obtain a copy of the License at:                                         +
#                                                                                        +
#       http://www.apache.org/licenses/LICENSE-2.0                                       +
#                                                                                        +
#       Unless required by applicable law or agreed to in writing, software              +
#       distributed under the License is distributed on an "AS IS" BASIS,                +
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         +
#       See the License for the specific language governing permissions and              +
#       limitations under the License.                                                   +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from keras.datasets import cifar100
from graphviz import Digraph
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from treelib import Tree
from sklearn.model_selection import train_test_split
from  skimage import transform



class Marine_Dataset:

    def __init__(self, name, dataset_path, train_labels_path, test_labels_path,output_level, image_size=(64, 64), batch_size=32):
        
        self.name = name
        self.image_size_ = image_size
        self.image_size = (image_size[0], image_size[1], 3)
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.output_level = output_level
        # Training set
        train_labels_df = pd.read_csv(train_labels_path, sep=",", header=0)
        train_labels_df = train_labels_df.sample(frac=1).reset_index(drop=True)
        self.train_labels_df = train_labels_df
        # Splitting into val and test sets
        test_labels_df = pd.read_csv(test_labels_path, sep=",", header=0)
        test_labels_df = test_labels_df.sample(frac=1, random_state=1).reset_index(drop=True)
        self.test_labels_df = test_labels_df
        
        
        train_labels_df,val_labels_df = train_test_split(train_labels_df, test_size=0.12,random_state=42, stratify=train_labels_df['class_level_4'])
        
        self.train_labels_df = train_labels_df
        
        self.val_labels_df = val_labels_df
        
        # Validation set
        self.val_dataset = self.get_pipeline(val_labels_df,output_level)
        #Train set
        self.train_dataset = self.get_pipeline(train_labels_df,output_level)
        # Test set
        self.test_dataset = self.get_pipeline(test_labels_df,output_level)
                
        
        # Number of classes
        self.num_classes_l0 = len(set(train_labels_df['class_level_0']))
        self.num_classes_l1 = len(set(train_labels_df['class_level_1']))
        self.num_classes_l2 = len(set(train_labels_df['class_level_2']))
        self.num_classes_l3 = len(set(train_labels_df['class_level_3']))
        self.num_classes_l4 = len(set(train_labels_df['class_level_4']))
        self.num_classes = [self.num_classes_l0, self.num_classes_l1, self.num_classes_l2,
                            self.num_classes_l3,self.num_classes_l4]
        
        
        # Encoding the taxonomy
        m0 = [[0 for x in range(self.num_classes_l1)] for y in range(self.num_classes_l0)]
        
        for (t, c) in zip(list(train_labels_df['class_level_0']), list(train_labels_df['class_level_1'])):
            m0[t][c] = 1
        
        m1 = [[0 for x in range(self.num_classes_l2)] for y in range(self.num_classes_l1)]
        for (t, c) in zip(list(train_labels_df['class_level_1']), list(train_labels_df['class_level_2'])):
            m1[t][c] = 1
            
        m2 = [[0 for x in range(self.num_classes_l3)] for y in range(self.num_classes_l2)]
        for (t, c) in zip(list(train_labels_df['class_level_2']), list(train_labels_df['class_level_3'])):
            m2[t][c] = 1
            
        m3 = [[0 for x in range(self.num_classes_l4)] for y in range(self.num_classes_l3)]
        for (t, c) in zip(list(train_labels_df['class_level_3']), list(train_labels_df['class_level_4'])):
            m3[t][c] = 1
       
       
        self.taxonomy = [m0, m1,m2,m3]
        
        
        # Build the labels
        self.labels = []
        labels = ['' for x in range(self.num_classes_l0)]
        for (l, c) in zip(list(train_labels_df['label_level_0']), list(train_labels_df['class_level_0'])):
            labels[c] = l
        self.labels.append(labels)

        labels = ['' for x in range(self.num_classes_l1)]
        for (l, c) in zip(list(train_labels_df['label_level_1']), list(train_labels_df['class_level_1'])):
            labels[c] = l
        self.labels.append(labels)

        labels = ['' for x in range(self.num_classes_l2)]
        for (l, c) in zip(list(train_labels_df['label_level_2']), list(train_labels_df['class_level_2'])):
            labels[c] = l
        self.labels.append(labels)
        
        labels = ['' for x in range(self.num_classes_l3)]
        for (l, c) in zip(list(train_labels_df['label_level_3']), list(train_labels_df['class_level_3'])):
            labels[c] = l
        self.labels.append(labels)
        
        labels = ['' for x in range(self.num_classes_l4)]
        for (l, c) in zip(list(train_labels_df['label_level_4']), list(train_labels_df['class_level_4'])):
            labels[c] = l
        self.labels.append(labels)
        

    def encode_single_sample(self, img_path, class_level_0, class_level_1, class_level_2, class_level_3, class_level_4,fname,output_level):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_image(img, expand_animations=False)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, self.image_size_)
        
        if self.output_level == 'last_level':
            return img, class_level_4
        
        if self.output_level == 'all':
            return img, (class_level_0, class_level_1, class_level_2, class_level_3, class_level_4)


    def get_pipeline(self, dataframe, output_level):
        
        self.output_level = output_level
        dataset = tf.data.Dataset.from_tensor_slices(([self.dataset_path + '/' + x for x in dataframe['fname']],
                                                      list(dataframe['class_level_0']),
                                                      list(dataframe['class_level_1']),
                                                      list(dataframe['class_level_2']),
                                                      list(dataframe['class_level_3']),
                                                      list(dataframe['class_level_4']),
                                                      list(dataframe['fname']),
                                                      [output_level for x in dataframe['fname']]))
        
    
        dataset = (
        dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
                .padded_batch(self.batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE))
            
    
        
        
        
        
        return dataset

    def get_tree(self):
        return get_tree(self.taxonomy, self.labels)


def get_tree(taxonomy, labels):
    """
    This method draws the taxonomy using the graphviz library.
    :return:
    :rtype: Digraph
     """
    tree = Tree()
    tree.create_node("Root", "root")  # root node

    for i in range(len(taxonomy[0])):
        tree.create_node(labels[0][i] + ' -> (L0_' + str(i) + ')', 'L0_' + str(i), parent="root")

    for l in range(len(taxonomy)):
        for i in range(len(taxonomy[l])):
            for j in range(len(taxonomy[l][i])):
                if taxonomy[l][i][j] == 1:
                    tree.create_node(labels[l + 1][j] + ' -> (L' + str(l + 1) + '_' + str(j) + ')',
                                     'L' + str(l + 1) + '_' + str(j),
                                     parent='L' + str(l) + '_' + str(i))

    return tree



def get_Marine_dataset(output_level,image_size=(64, 64), batch_size=32,subtype='Tropical'):
    # Get images
    
    dataset_path = 'D:\\Thesis\\Marine_dataset\\marine_images\\'
    
    if subtype == 'Tropical':
        
        train_labels_path ='D:\\Thesis\\MARINE_fix\\Tropical\\Train\\train_labels_trop.csv'
        test_labels_path = 'D:\\Thesis\\MARINE_fix\\Tropical\\Test\\test_labels_trop.csv'
        
    elif subtype == 'Temperate':
        
        train_labels_path ='D:\\Thesis\\MARINE_fix\\Temperate\\Train\\train_labels_temp.csv'
        test_labels_path = 'D:\\Thesis\\MARINE_fix\\Temperate\\Test\\test_labels_temp.csv'
        
    else:
        
        train_labels_path ='D:\\Thesis\\MARINE_fix\\Combined\\Train\\train_labels_comb.csv'
        test_labels_path = 'D:\\Thesis\\MARINE_fix\\Combined\\Test\\test_labels_comb.csv'
        
    
    dataset_name = 'Marine_dataset_'+subtype


    return Marine_Dataset(dataset_name, dataset_path, train_labels_path, test_labels_path,output_level, image_size, batch_size)


if __name__ == '__main__':
    dataset = get_Marine_dataset(output_level='all',image_size=(64,64),batch_size=32)

