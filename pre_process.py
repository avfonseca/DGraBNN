import open3d as o3d
import open3d.core as o3c
# or import open3d.ml.tf as ml3d
import numpy as np
import pandas as pd
import os
import glob
import pyarrow.parquet as pq
import pyarrow as pa
import os
from itertools import islice
import warnings 

warnings.filterwarnings('ignore')  #deprecation warnings code works as intended

class PreProcess:
    
    def __init__(self,dir,survey):
        
        print("Creating Dataset for " + survey + " Survey\n")
        
        self.dir = dir
        self.survey = survey
        
        self.indir_csv_acc = os.path.join(dir, str(survey), "raw", "accepted")
        self.indir_csv_rej = os.path.join(dir, str(survey), "raw", "rejected/")
        
        
        self.outdir_csv_acc = self.ifnot_make(os.path.join(dir, survey, "csv", "accepted"))
        self.outdir_csv_rej = self.ifnot_make(os.path.join(dir, survey, "csv", "rejected"))
        
        print("Rewriting Survey Files to equal sized files\n")
        self.rewrite_file_size(self.indir_csv_acc,self.outdir_csv_acc)
        self.rewrite_file_size(self.indir_csv_rej,self.outdir_csv_rej)
        print("Done rewriting Survey Files to equal sized files\n")

        
        self.out_parquet = self.ifnot_make(os.path.join(dir, survey, "parquet"))
        
        self.out_parquet_acc = os.path.join(self.out_parquet, "acc_" + str(survey) + ".parquet") 
        self.out_parquet_rej = os.path.join(self.out_parquet, "rej_" + str(survey) + ".parquet")
        
        self.out_pc = self.ifnot_make(os.path.join(dir, survey, "point_clouds"))
        self.out_multiscale = self.ifnot_make(os.path.join(dir, survey, "multi_scale"))
        
        print("Rewriting Survey Files to parquet format\n")
        self.csv2parquet()
        print("Done rewriting Survey Files to parquet format\n")
        print("Creating Point Cloud\n")
        self.csv2pc()
        print("Done creating Point Cloud\n")
        print("Creating Multiscale decimated Point Clouds\n")
        self.create_multiscale_pc()
        print("Done creating Multiscale decimated Point Clouds\n")
        
        
        print("Done creating Dataset for " + survey + " Survey\n")
        
    def ifnot_make(self,dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
        
    def rewrite_file_size(self,input_folder,output_folder,desired_row_count = 300001):

        # Read the header from one of the input files
        header = None
        for root, _, files in os.walk(input_folder):
            for filename in files:
                if filename.endswith('.txt'):
                    with open(os.path.join(root, filename), 'r') as file:
                        header = file.readline()
                    break

        # Iterate through each input file, split into chunks, and create output files
        file_counter = 1
        current_rows = []
        for root, _, files in os.walk(input_folder):
            for filename in files:
                if filename.endswith('.txt'):
                    with open(os.path.join(root, filename), 'r') as input_file:
                        for line in islice(input_file, 1, None):
                            current_rows.append(line)
                            if len(current_rows) >= desired_row_count:
                                # Create a new output file with the same header
                                output_filename = os.path.join(output_folder, f'output_{file_counter}.txt')
                                with open(output_filename, 'w') as output_file:
                                    output_file.write(header)
                                    output_file.writelines(current_rows)
                                # Reset the row counter and current rows list
                                current_rows = []
                                file_counter += 1

        # Create a final output file for any remaining rows
        if current_rows:
            output_filename = os.path.join(output_folder, f'output_{file_counter}.txt')
            with open(output_filename, 'w') as output_file:
                output_file.write(header)
                output_file.writelines(current_rows)

    def csv2parquet(self):
        
        csv_acc = glob.glob(os.path.join(self.outdir_csv_acc, "*.txt"))
        csv_rej = glob.glob(os.path.join(self.outdir_csv_rej, "*.txt"))
        
        for i in range(len(csv_acc)):
            df =  pd.read_csv(csv_acc[i]).assign(Flag = 1)
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(table , root_path=self.out_parquet_acc)
    

        for i in range(len(csv_rej)):
            df =  pd.read_csv(csv_rej[i]).assign(Flag = 0)
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(table , root_path=self.out_parquet_rej)
    
    
    def csv2pc(self):
        
        csv_acc = glob.glob(os.path.join(self.outdir_csv_acc, "*.txt"))
        csv_rej = glob.glob(os.path.join(self.outdir_csv_rej, "*.txt"))

        for i,j in enumerate(csv_acc):
            df = pd.read_csv(j,delimiter= ',')
            points = df[["Footprint X","Footprint Y","Footprint Z"]].values

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(os.path.join(self.out_pc, "pc_" + str(i) + "_acc.ply"), pcd)
            
        for i,j in enumerate(csv_rej):
            df = pd.read_csv(j,delimiter= ',')
            points = df[["Footprint X","Footprint Y","Footprint Z"]].values

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(os.path.join(self.out_pc, "pc_" + str(i) + "_rej.ply"), pcd)
    
    def create_multiscale_pc(self,resolution = [0.5,1,10,100,500]):
        
        files = glob.glob(os.path.join(self.out_pc, "*.ply"))
        
        for voxel_size in resolution:

            pcds = o3d.io.read_point_cloud(files[0])
            pcds = pcds.voxel_down_sample(voxel_size=voxel_size)
            for i in range(1,len(files)):
                pcd = o3d.io.read_point_cloud(files[i])
                pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
                pcds.points.extend(np.asarray(pcd_down.points))
        
            o3d.io.write_point_cloud(os.path.join(self.out_multiscale,"pc_" + str(voxel_size).replace(".","")+ ".ply"), pcds)
    
