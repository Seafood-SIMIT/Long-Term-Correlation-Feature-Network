
from utils.readData import normaLization,readDataFilelists,readDataInFile
from tqdm import tqdm
data_dir = '/Volumes/T7/DataBase/aco_seis_Dataset/trainset'
frame_length = 1024
aco_dir, seis_dir,aco_filelist, seis_filelist = readDataFilelists(data_dir)

num_small,num_small_frame = 0,0
num_large,num_large_frame = 0,0
num_tracked,num_tracked_frame = 0,0
for index in tqdm(range(len(aco_filelist))):
    if aco_filelist[index].startswith('.'):
        continue
    flag,label,origin_signal_aco,origin_signal_seis = readDataInFile(
                                                    aco_dir,
                                                    seis_dir,
                                                    aco_filelist,
                                                    seis_filelist,
                                                    index)
    if flag == False:
        continue
    if label == 0:
        num_small+=1
        num_small_frame+=len(origin_signal_aco)//frame_length

    elif label == 1:
        num_large+=1
        num_large_frame+= len(origin_signal_aco)//frame_length
    elif label == 2:
        num_tracked+=1
        num_tracked_frame+= len(origin_signal_aco)//frame_length
    else:
        print("wronmg data")
print("small wheel file number: ",num_small,
        "small wheel frame number: ", num_small_frame )
print("large wheel file number: ",num_large,
        "large wheel frame number: ", num_large_frame )
print("track wheel file number: ",num_tracked,
        "track wheel frame number: ", num_tracked_frame )

