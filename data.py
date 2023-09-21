import os
import open3d as o3d
import numpy as np
import pyarrow.dataset as ds
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class HydroNet(Dataset):
    def __init__(self, num_points, survey_list, resolution, partition ='train', mode = 'autoencode', simple_load = True, col_list = None):
        
        #num_points is the number of points to construct the graph 
        self.BASE_DIR = "."
        
        #survey_list contains a list of all the survey names stored in data
        self.survey_list = survey_list
        
        self.num_points = num_points
        self.partition = partition 
        self.resolution = resolution 
        self.mode = mode

        self.data = []
        self.pc_tree = []
        self.pc = []
        self.fragment_length = []
        self.fragment_sum = []

        for i in range(len(survey_list)):
            
            
            dataset_acc = ds.dataset(os.path.join(self.BASE_DIR, 'data', survey_list[i], "parquet", "acc_" + survey_list[i] + ".parquet"), format="parquet", partitioning="hive")
            
            iterable_acc = list(dataset_acc.get_fragments())
            
            
            num_acc , num_rej = 0,0
            lengths_acc = []
            lengths_rej = []
            
            
            for fragment in iterable_acc:
                tb = fragment.to_table(columns = ["Flag"])
                lengths_acc.append(tb.num_rows)
                num_acc += tb.num_rows
                
            if (partition == "test"):
                dataset_rej = ds.dataset(os.path.join(self.BASE_DIR, 'data', survey_list[i], "parquet", "rej_" + survey_list[i] + ".parquet"), format="parquet", partitioning="hive")
                iterable_rej = list(dataset_rej.get_fragments())

                for fragment in iterable_rej:
                    tb = fragment.to_table(columns = ["Flag"])
                    lengths_rej.append(tb.num_rows)
                    num_rej += tb.num_rows
                
            
            
            if (self.mode == "balance"):
            #balance data_sets  (repeating least represented class) 
                self.data.append(np.hstack((np.repeat(iterable_acc,max(1,int(num_rej/num_acc))), np.repeat(iterable_rej,max(1,int(num_acc/num_rej))))))
                self.fragment_length.append(np.hstack((np.repeat(lengths_acc,max(1,int(num_rej/num_acc))), np.repeat(lengths_rej,max(1,int(num_acc/num_rej))))))
            else:   
                if (self.partition == 'test'):
                    self.data.append(iterable_rej)
                    self.fragment_length.append(lengths_rej)
                else:
                    self.data.append(iterable_acc)
                    self.fragment_length.append(lengths_acc)
           
           
            self.fragment_sum.append(np.cumsum(self.fragment_length[i]))
                
            self.load_pointcloud(survey_list[i],self.resolution)
            
        
        self.len_survey = np.array(list(map(lambda x: sum(x), self.fragment_length)))
        self.survey_sum = np.cumsum(self.len_survey)


    def __getitem__(self, item):
        
         # Continously get new item from different surveys if
         # ex if survey 1 has 100 points, 101 will be the first point of survey 2
        
        survey = sum(item > self.survey_sum - 1)
        item =  item - np.insert(self.survey_sum,0,0)[survey]
        
        fragment = sum(item > self.fragment_sum[survey] - 1)
        item = item - np.insert(self.fragment_sum[survey],0,0)[fragment]

        fragment_data, fragment_label = self.load_data(self.data[survey][fragment])
        
        
        point = fragment_data[item,:]
        label = fragment_label[item,:]
        
        
        #right now I still havent implemented the multiple resolutions
        tree = self.pc_tree[survey][self.resolution[0]][0]
        mean = self.pc_tree[survey][self.resolution[0]][1]
        var = self.pc_tree[survey][self.resolution[0]][2]
        
        [_, idx, _] = tree.search_hybrid_vector_3d(point,self.num_points,100)
        
        nn = np.asarray(self.pc[survey][self.resolution[0]].points)[idx]
              
        return (np.concatenate((nn,point.reshape(1,-1))) - mean)/np.diagonal(var),label
        
         

    def __len__(self):
         return sum(self.len_survey)
     
    
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