import os
import open3d as o3d
import numpy as np
import pyarrow.dataset as ds
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import random

class HydroNet(Dataset):
    def __init__(self, num_points, survey_list, resolution, partition ='train', simple_load = True, col_list = None):
        
        #num_points is the number of points to construct the graph 
        self.BASE_DIR = "."
        
        #survey_list contains a list of all the survey names stored in data
        self.survey_list = survey_list
        
        self.num_points = num_points
        self.partition = partition 
        self.resolution = resolution 
        
        self.pc_tree = []
        self.pc = []
        print(len(survey_list))
        self.data = [[]]*len(survey_list)
        print(len(self.data))
        self.fragment_length = [[]]*len(survey_list)
        self.survey_length = []

        for i in range(len(survey_list)):
            
            if (self.partition == "test"):
                dataset = ds.dataset(os.path.join(self.BASE_DIR, 'data', survey_list[i], "parquet", "rej_" + survey_list[i] + ".parquet"), format="parquet", partitioning="hive")
                iterable = list(dataset.get_fragments())
            
            else:
                dataset = ds.dataset(os.path.join(self.BASE_DIR, 'data', survey_list[i], "parquet", "acc_" + survey_list[i] + ".parquet"), format="parquet", partitioning="hive")
                iterable = list(dataset.get_fragments())

            for fragment in iterable:
                    tb = fragment.to_table(columns = ["Flag"])
                    length = tb.num_rows
                    self.data[i].append(iterable)
                    self.fragment_length[i].append(length)
           
               
            self.load_pointcloud(survey_list[i],self.resolution)
            self.survey_length.append(len(self.fragment_length[i]))
        
        self.dataset_length = sum(self.survey_length)

    def __getitem__(self, item):
        
         # Randomly get an item from selected fragment
        survey = sum(item > self.survey_length - 1)
        fragment = item - np.insert(self.survey_length,0,0)[survey]
        fragment_data, fragment_label = self.load_data(self.data[survey][fragment])
        
        
        choice = random.randint(0, fragment_data.shape[0])
        
        point = fragment_data[choice,:]
        label = fragment_label[choice,:]
        
        
        #right now I still havent implemented the multiple resolutions
        tree = self.pc_tree[survey][self.resolution[0]][0]
        mean = self.pc_tree[survey][self.resolution[0]][1]
        cov = self.pc_tree[survey][self.resolution[0]][2]
        
        [_, idx, _] = tree.search_hybrid_vector_3d(point,self.num_points,100)
        
        nn = np.asarray(self.pc[survey][self.resolution[0]].points)[idx]
              
        return (np.concatenate((nn,point.reshape(1,-1))) - mean)/np.diagonal(cov),label
        
         

    def __len__(self):
         return self.dataset_length
     
    
    def load_pointcloud(self,survey, res_list):
    
        pc_tree_dict = {}
        pc_dict = {}
        
        for i in range(len(res_list)):
        
            pc = o3d.io.read_point_cloud(os.path.join(self.BASE_DIR, "data/Point_Clouds", survey + "_" + str(res_list[i]) + ".ply"))
            
            mean,var = pc.compute_mean_and_covariance()
            
            tree = o3d.geometry.KDTreeFlann(pc)
            
            pc_tree_dict[res_list[i]] = [tree,mean,var]
            pc_dict[res_list[i]] = pc
            
        
        
        self.pc_tree.append(pc_tree_dict)
        self.pc.append(pc_dict)
        
    def load_data(self,fragment, col_list = ['Footprint X', 'Footprint Y', 'Footprint Z']):

        data = fragment.to_table(columns = col_list).to_pandas().to_numpy()
        label = fragment.to_table(columns = ["Flag"]).to_pandas().to_numpy()
    
        return data, label