
# Assigment 15 Predict DepthMap and Mask
This assignment is to generate DepthMap and Mask

# 1.  Problem Statement:

We need to  generate DepthMap and Mask.The DataSet available for this Assignment  is as below:

100 background, 100x2 (including flip), and you randomly place the foreground on the background 20 times, you have in total 100x200x20 images. 

In total we MUST have:

    400k fg_bg images
    400k depth images
    400k mask images
    generated from:
     100 backgrounds
     100 foregrounds, plus their flips
     20 random placement on each background


# 2. DataSet Generation:

### Link to GoogleDrive for the complete dataset:

https://drive.google.com/open?id=1o5hPttBP_x5GD37AFYJQ41rvJ9Xdb8n0

### The dataset generated for this assigment is created as part of 15A_Generate_Mask_Depth_Dataset
Check the README.md for mode details on 15A for generate of DataSet for Foreground,BackGround,ForeGround-BackGroud,Mask,Mask_Fg_Bg,
Depth Image.

Also check the CodeBase available at 

https://github.com/sudhakarmlal/EVA4/tree/master/Session14-15

Details:

1.Image creation steps through GIMP tool:
https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/ImageCreationSteps.pdf

2.Generate Fg_Bg Images:
https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/generate_fg_bg_images_jpg.py

3.Generating Dense Depth Images:

https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/DenseDepth.ipynb

4.Findig the mean and standard deviation of the images

https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/imagestats_std_mean.py

5.Generating Dense Depth Images:

https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/DenseDepth.ipynb


5. GalleryUtil to generate the Gallery of images for Mask,ForeGroup,Backgroupd,Fg_Bg,Mask_Fg_Bg,DepthImages
https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/GalleryUtil.ipynb

Finally the following are generated:

     The directory contains 10 zip files (batch1_images.zip,batch2_images.zip, batch3_images.zip, batch4_images.zip, batch5_images.zip, batch6_images.zip batch7_images.zip, batch8_images.zip, batch9_images.zip, batch10_images.zip)
     
     Each zip file has the following folders:
    
     
     fg_jpg : Foreground jpg images of size (80x80) with each image is uniquely numbered across batches e.g. fg_img_91.jpg
     bg_jpg : Background jpeg images of size (160x160) with each image is uniquely numbered  but repeated across batches e.g. bg_img_5.jpg
     mask_jpg: mask jpeg images of size (80x80) with each image is uniquely numbered  but repeated across batches e.g. mask_img_5.jpg
     fg_bg_jpg : Foreground images overlayed on background images of size (160x160) with each image is uniquely numbered as
     fg_bg_1_4_0_15 where the first digit 1 represents the foreground image id from which it is generated
      second digit 4 represents the background image id from which it is generated, 0 represents, this image is not flipped (if flipped it will have value 1), last digit 15 represents the sequence number (1-20)
      mask_black_jpg: Mask image overlayed on black background with each image is uniquely numbered as
     bg_mask_1_4_0_15 where the first digit 1 represents the foreground image id from which it is generated
      second digit 4 represents the background image id from which it is generated, 0 represents, this image is not flipped (if flipped it will have value 1), last digit 15 represents the sequence number (1-20)
      
      depth_fg_bg_jpg: JPG images created using the depth model prediction on fg_bg images. The convention for depth_fg_bg_jp is same as corresponding fg_bg_jpg image from which it is generated, only depth_ is prepended to image name.
      

Also find the mean/standard deviation of the Different set of images(Generated from):

https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/imagestats_std_mean.py
(This would be later used by the Model)

	Category : bg_jpg  Mean : 0.739088,  Std :  0.265235	
	Category : depth_fg_bg_jpg  Mean : 0.777681,  Std :  0.311899
	Category : fg_bg_jpg  Mean : 0.729304,  Std :  0.271675
	Category : fg_jpg  Mean : 0.851283,  Std :  0.270902
	Category : mask_black_jpg  Mean : 0.068877,  Std :  0.249513
	Category : mask_jpg  Mean : 0.276214,  Std :  0.436924
	
      
      
 # 3 DataExtraction,Data Mapping,Data preparation for TrainDataLoader
 
 ### DataLayout:
 
 The below is the datalayout for the data created as part of Step2(DataSet generation)
 
 ![](https://github.com/sudhakarmlal/EVA4/blob/master/Session15/Images/DataLayout.gif) 
  
  As explained in the diagram the DataSet contains  10 Batches each batch has  40,000 Images(corresponding to 10 Foregrounds)
  
  ### Data Extraction:
  As a part of data extraction .All the 10 Zip files are loaded into the GoogleDrive .The Data would be extracted in Google Drive
  by running the following(e.g for batch1)
  
  ####  !unzip /content/drive/'My Drive'/MASK1/batch1_images.zip  -d /content/drive/'My Drive'/MASK1/batch_images
  
  Once the data is extracted it's available to read.The following code is used to read  IMAGES
  #### At a time we read data per batch.Also we  feed 1 batch data only  to train-data-loader at a time due to resource constraints.
  
  
  The following code expains how the data is read int various python variables:
  
          BG_DIR = "/content/gdrive/My Drive/MASK1/batch_images1/bg_jpg"
          FG_DIR = "/content/gdrive/My Drive/MASK1/batch_images1/fg_jpg"
          MASK_DIR = "/content/gdrive/My Drive/MASK1/batch_images1/mask_black_jpg"
          DP_DIR = "/content/gdrive/My Drive/MASK1/batch_images1/depth_fg_bg_jpg"
          FG_BG_DIR = "/content/gdrive/My Drive/MASK1/batch_images1/fg_bg_jpg"
	  
	  
	  def get_img_file_names(path):
  	     img_file_names =[]
  	     for root, dirs, files in os.walk(path):
                for filename in files:
                    #if img_file is not None:
                    #print(filename)
                    img_file_names.append(path + '/' + filename)  
                    return img_file_names
      
        bg_file_names = get_img_file_names(BG_DIR)
	fg_file_names =get_img_file_names(FG_DIR)
	mask_file_names =get_img_file_names(MASK_DIR)
	dp_file_names =get_img_file_names(DP_DIR)
	fg_bg_file_names =get_img_file_names(FG_BG_DIR)
		    
Since,there are 10 Foreground Images ,100 Foreground Images,40000 mask_fg_bg,40000 depth,40000 fg_bg Images.
It's going to generate the following(per batch):

          bg_file_names  : 100
	  fg_file_names  : 10
          mask_file_names : 40,000
	  dp_file_names : 40,000
	  fg_bg_file_names : 40,000
	  
### Data Mapping:
       
The below diagram explains how the data has to be mapped in order to feed it to the traindata loader.

![](https://github.com/sudhakarmlal/EVA4/blob/master/Session15/Images/TrainingDataLoader.gif) 


#### This Mapping is required in order to generate 40,000 data objects(batch1) for each of the  bg,fg,mask(fg-bg),depth,fg-bg in order to feed to the train_dataloader.

Below code explains whatever demonstrated in the above diagram:

        forground_image_names = []
	bg_mask_img_names = []
	bg_dp_img_names = []
	bg_fg_bg_img_names = []

	mask_img_names  = []
	dp_img_names = []
	fg_images_names = []
	fg_bg_img_names = []

	for i in range(len(fg_file_names)):
  		fg_str = fg_file_names[i].split('/')[-1]
  		fg_name = fg_str[0:fg_str.rfind('.jpg')]
  		print(fg_name)
  		forground_image_names.append(fg_name)

	for i in range(len(forground_image_names)):
  		for j in range(len(bg_file_names)):
    			print(bg_file_names[j])
    			bg_str = bg_file_names[j].split('_')[-1]
    			print(bg_str[0:bg_str.rfind('.jpg')])
    			bg_num  = bg_str[0:bg_str.rfind('.jpg')]
    			search_str_mask ="bg_mask"+forground_image_names[i][6:]
    			search_str_mask = search_str_mask + "_" +bg_num+"_"
    			print(search_str_mask)
    			search_str_depth_fg_bg ="depth_fg_bg"+forground_image_names[i][6:]
    			search_str_depth_fg_bg =search_str_depth_fg_bg + "_" +bg_num +"_"
    			print(search_str_depth_fg_bg)

    			search_str_fg_bg ="fg_bg"+forground_image_names[i][6:]
    			search_str_fg_bg =search_str_fg_bg + "_" +bg_num +"_"
    			print(search_str_fg_bg)
  
    				for k in range(len(mask_file_names)):
      					mask_str = mask_file_names[k].split('/')[-1]
      					fg_depth_bg_str =dp_file_names[k].split('/')[-1]
      					fg_bg_str = fg_bg_file_names[k].split('/')[-1]
      					if mask_str.startswith(search_str_mask):
        					mask_img_names.append(mask_file_names[k])
        					bg_mask_img_names.append(bg_file_names[j])
      					if fg_bg_str.startswith(search_str_fg_bg):
        					fg_bg_img_names.append(fg_bg_file_names[k])
        					bg_fg_bg_img_names.append(bg_file_names[j])
        					#fg_images_names.append(fg_file_names[i])  
      					if fg_depth_bg_str.startswith(search_str_depth_fg_bg):
        					dp_img_names.append(dp_file_names[k])
        					bg_dp_img_names.append(bg_file_names[j])
        					fg_images_names.append(fg_file_names[i]) 


  

# 4 Preparing Data for training and Training Strategy
     
  The DataExtraction,DataMapping generate only the filenames.The images objects has to be created to feed it to the Model.
  
  The below code is used to create the DataSet of Images:
  
  
  	class MasterDataset(Dataset):
  		def __init__(self,  transform= None, bg_files= None, fg_bg_files= None, ms_bg_files= None, dp_files= None):
    			self.bg_files= bg_files
    			self.fg_bg_files= fg_bg_files
    			self.ms_bg_files= ms_bg_files
    			self.dp_files= dp_files
    			#self.ms_bg_files= list([y for x in os.walk(MASK_DIR) for y in glob(os.path.join(x[0], '*.jpg'))]) 
    			#self.bg_files= list(BG_DIR.glob('*.jpg'))   
    			self.transform = transform
  

  		def __len__(self):
    			return len(self.bg_files)

  		def __getitem__(self,index):
    			bg_image = Image.open(self.bg_files[index])
    			fg_bg_image = Image.open(self.fg_bg_files[index])
    			ms_bg_image = Image.open(self.ms_bg_files[index])
    			dp_image = Image.open(self.dp_files[index])
    			if self.transform:
      				bg_image = self.transform(bg_image)
      				fg_bg_image = self.transform(fg_bg_image)
      				ms_bg_image = self.transform(ms_bg_image)
      				dp_image = self.transform(dp_image)
    		return {'bg_image' : bg_image,'fg_bg_image' : fg_bg_image,'ms_bg_image' : ms_bg_image, 'dp_image' : dp_image }

To call the above MasterDataSet The following transform is created:

The below takes some standard values for mean and standard deviation.The following code should have been used to generate mean and standard deviation https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/imagestats_std_mean.py

    	 mean, std = torch.tensor([0.485,0.456,0.406])*255, torch.tensor([0.229,0.224,0.225])*255
	 train_transform = transforms.Compose([
                  transforms.Resize((64,64)),
                  transforms.Grayscale(num_output_channels=3),
                  transforms.ToTensor()
	])
We can now feed to create the MasterDataSet:
#### Note: We are feeding the filenames only to the MasterDataSet class and created ImageObjects out of it by applying the transform in the above code.

     train_ds = MasterDataset(train_transform, bg_dp_img_names, fg_bg_img_names, mask_img_names,dp_img_names)
     
### Training Strategy due to  resource & memory constraints:

We need to apply the following training strategy as we hava a memory constraints.We can't feed all the 40K(Batch1) images as it takes huge time to run on google colab and some times it hangs.

## Tried  training 40K images in one shot it takes almost 12-14 hours in Google colab.Hence has to come up with a training strategy.

The below diagram explains the dataset to be fed to the traindataloader which is a subset of 40K(1st Batch)

#### Training Strategy 1:

The below digram explains the training strategy1.In this we would be feeing only  4000 Images to the TrainDataLoader
![](https://github.com/sudhakarmlal/EVA4/blob/master/Session15/Images/TrainingLayout1.gif)

  The below is the code snippet of the Mapping code(explained in Step3.DataMapping section) that has to be changed in order to generate     4000 images:	
	   
	    for i in range(len(forground_image_names)):
  		for j in range(10):
    			print(bg_file_names[j])
    			bg_str = bg_file_names[j].split('_')[-1]
    			print(bg_str[0:bg_str.rfind('.jpg')])
    			bg_num  = bg_str[0:bg_str.rfind('.jpg')]


#### Training Strategy 2:

The below digram explains the training strategy2.In this we would be feeing only  16,000 Images to the TrainDataLoader

![](https://github.com/sudhakarmlal/EVA4/blob/master/Session15/Images/TrainingLayout2.gif)

	
	

The below is the code snippet of the Mapping code(explained in Step3.DataMapping section) that has to be changed in order to generate 16000 images:

	
	for i in range(len(forground_image_names)):
  		for j in range(40):
    			print(bg_file_names[j])
    			bg_str = bg_file_names[j].split('_')[-1]
    			print(bg_str[0:bg_str.rfind('.jpg')])
    			bg_num  = bg_str[0:bg_str.rfind('.jpg')]

## I decided to go for training strategy2.Running model for training strategy2 took 3-4 hours for  16000 imaages.

Once you have the MasterDataSet available it can be used to feed to the train dataloader.

	train_dl = DataLoader(train_ds, batch_size=16, shuffle = True, pin_memory=True)
	sample = next( iter(train_dl))
	type(sample)
	

# 5 Defining the Model

The following Models are tried for generating DepthModel and MaskImages:

## Note:Have tried multiple models but The following two different models are what been considered.The output for these two models only shown in the later steps.


### Mode11(Generates both Depth and MaskOutput):



    class Net(nn.Module):
      def __init__(self):
        super(Net, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            # nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.resblock1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            # nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            # nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.resblock2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 3, 3, stride=1, padding=1, bias=False)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False,
                      groups=32),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(256, 3, 3, padding=(1, 1), stride=(1, 1), bias=False)
        )

    def forward(self, x):
        bg_image = x["bg_image"]
        fg_bg_image = x["fg_bg_image"]
        x = torch.cat([bg_image, fg_bg_image], dim=1)
        x = self.input_layer(x)
        x = self.layer1(x)
        r1 = self.resblock1(x)
        x = x + r1
        x = self.layer2(x)
        x = self.layer3(x)
        r2 = self.resblock2(x)
        x = x + r2
        x = self.layer4(x)
        # print(x.shape)

        y1 = self.layer5(bg_image)
        y1 = self.layer6(y1)
        y2 = self.layer5(fg_bg_image)
        y2 = self.layer6(y2)
        y = torch.cat([y1, y2], dim=1)
        y = self.layer7(y)
        y = self.layer8(y)
        y = self.layer9(y)
        y = self.layer10(y)
        return y, x


### Mode12(Generates both Mask and Depth Output but only MaskOutput considered as it showed good results for MaskOUtput):
    
    class ConvGen(nn.Module):
    def __init__(self):
        super(ConvGen, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(3,32,3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(32,32,3, stride=1, padding=1, bias=False, groups=32),
            nn.Conv2d(32,64,1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(128,256,3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(256,3,3, stride=1, padding=1, bias=False)            
        )
        
    def forward(self, x):
        bg_image = x["bg_image"]
        fg_bg_image = x["fg_bg_image"]        
        bg_image = self.convblock2(self.convblock1(bg_image))
        fg_bg_image = self.convblock2(self.convblock1(fg_bg_image))
        f= torch.cat([bg_image,fg_bg_image], dim=1)
        f = self.convblock4(self.convblock3(f))
        return f
        

# 6 Defining the Loss Criterion for the Model:

## Model1 Loss Criterion:

Two different  Loss criterions are defined for the Outputs of Model1.
###  The BCEWithLogitsLoss() criterion is for the MaskOutput
###  The SSIM criterion is for the  Depth Output from the Model
 	from torch.optim.lr_scheduler import StepLR
	from kornia.losses import SSIM
	criterion1 = nn.BCEWithLogitsLoss()
	criterion2 = SSIM(3, reduction="mean" )
### The following derivation is used in order to combine both the losses i.e total loss is  2*loss1 + loss2
###  Where loss1 is MaskOutput Loss and Loss2 is DepthOutput Loss 
       
           loss1 = criterion1(output[0],data["ms_bg_image"])
           loss2 = criterion2(output[1],data["dp_image"])
           loss = 2*loss1 + loss2
	   

# 7 Show Output Utility for the Model:

The Output Mask and Depth Images to be showed or stored in Google drive.For the same the following utility is written:

            IMG_DIR = "./gdrive/My Drive/MASK1/S15/out_images3"
	   def saveimage(tensors, name, figsize=(50,50), *args, **kwargs):
  	   	filename= IMG_DIR+ name
  	   	try:
    			tensors = tensors.detach().cpu()
  			except: 
    				pass
  		grid_tensor1= torchvision.utils.make_grid(tensors,*args, **kwargs)
  		grid_image1= grid_tensor1.permute(1,2,0)
  		plt.figure(figsize=figsize)
  		plt.imshow(grid_image1)
  		plt.xticks([])
  		plt.yticks([])
  		plt.savefig(filename, bbox_inches = 'tight')
  		plt.show()
		
# 8 Train  Model Function:
The below code snippnet shows the function to train the Model.Note the train function accepts the following arguments.
#### Note the following:
##### 1.The train method accepts the writer which is the tensorflow writer.This will set the tensorflow profiler to the model.
##### 2.Two of the loss  criterions used for loss calcuations.
##### 3.Show image utility invoked to show the output of Mask and Depth Images generated by Model
##### 4.The model is saved so that it can be reloaded.
##### 5.The timings for Model Execution,DataLoading,Other execution is recorded. 


	def train( batch, model,scheduler, criterion1,criterion2, device, train_loader, optimizer, epoch,iteration,writer):
  		other_time = 0
  		correct = 0
  		start = time()
  		other_s = time()
  		model.train()
  		other_e = time()
  		other_time += other_e - other_s
  		data_load_time = 0
  		model_time = 0
  		meow_time = 0
  		pbar = tqdm(train_loader)
  		for batch_idx, data in enumerate(pbar):
    			other_s = time()
    			optimizer.zero_grad()
    			other_e = time()
    			other_time += other_e - other_s
    			load_s = time()
    			data["bg_image"] = data["bg_image"].to(device)
    			data["fg_bg_image"] = data["fg_bg_image"].to(device)
    			data["ms_bg_image"] = data["ms_bg_image"].to(device)
    			data["dp_image"] = data["dp_image"].to(device)
    			load_e = time()
    			data_load_time += load_e - load_s
    			model_s = time() 
    			optimizer.zero_grad()
    			output=model(data)    
    			#loss= criterion(output,data["ms_bg_image"])
    			#loss1 = criterion(output[0],data["ms_bg_image"])
    			#loss2 = criterion(output[1],data["dp_image"])
    			loss1 = criterion1(output[0],data["ms_bg_image"])
    			loss2 = criterion2(output[1],data["dp_image"])
    			loss = 2*loss1 + loss2
    			pbar.set_description(desc= f'l1={round(loss1.item(),4)} l2={round(loss2.item(),4)}')
    			loss.backward()
    			optimizer.step()
    			model_e = time()
    			model_time += model_e - model_s
    			other_s = time()
    			if batch_idx % 250 == 0:
      				writer.add_scalar('training loss', loss.item() / 1000, epoch * iteration + batch_idx)
    			if batch_idx % 10 == 0:
      				torch.cuda.empty_cache()
    			if batch_idx % 50 == 0:
      				print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss: 		{:.6f}'.format(epoch,batch_idx*len(data),len(train_loader.dataset), 
                                                                    100.*batch_idx/len(train_loader), loss.item()))
    			if epoch == 4:
      				if batch_idx % iteration == 0:
        				saveimage(data["fg_bg_image"], f"/{batch}_3CFGBGORG.jpg")
        				saveimage(output[0], f"/{batch}_3CMSBGPDCT.jpg")
        				saveimage(data["ms_bg_image"], f"/{batch}_3CMSBGORG.jpg")
        				saveimage(output[1], f"/{batch}_3CDPBGPDCT.jpg")      
        				saveimage(data["dp_image"], f"/{batch}_3CDPBGORG.jpg")  
    			other_e = time()
    			other_time += other_e - other_s
  		end = time()
	 print(f'Total Execution time : {end-start:.2f} s')
         print(f'Model Execution Time : {model_time:.2f} s')
         print(f'Data Loading Time : {data_load_time:.2f} s')
         print(f'Other Execution Time : {other_time:.2f} s')	

# 8 Execute Model :
The below code is used to execute the mode.The Model is saved after the execution.The Model is saved after every epoch

	def executeModel(batch,train_dl,model, fromepoch, toepoch):
  		iteration= len(train_dl)
  		for epoch in range(fromepoch, toepoch):
    			#modeldata = trainmodeldepth(batch, model,scheduler, criterion1,criterion2, device, train_dl, optim, epoch,iteration,writer)
    			train( batch, model,scheduler, criterion1,criterion2, device, train_dl, optim, epoch,iteration,writer)
    			scheduler.step()  
  			torch.save(model.state_dict(), PATH/f"modelup3C_{batch}.pth")
			
# 9 Model Loss Output :

Below is the Loss Output generated out of the Model:

![](https://github.com/sudhakarmlal/EVA4/blob/master/Session15/Images/ModelExecutionLogs.gif)

# 10 Model Output Images :

Below are the sample ouput depth images generated out of Model1 and sample mask images generated out of Model2:

Depth Images(Model1):

![](https://github.com/sudhakarmlal/EVA4/blob/master/Session15/Images/out_images3out2.jpg)

v/s 

The actual Depth Images:
![](https://github.com/sudhakarmlal/EVA4/blob/master/Session15/Images/out_images3db_image.jpg)


Mask Images:

The below are the MaskedImages output generated out of the Model2

![](https://github.com/sudhakarmlal/EVA4/blob/master/Session15/Images/download.png)

# 11 Timings recorded for DataLoading,ModelExecution and Other timings :

![](https://github.com/sudhakarmlal/EVA4/blob/master/Session15/Images/ModelExectTime.gif)

 1) Open foreground image in GIMP    

![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/Screen1.png)  

  2) Open ‘view’ tab on the top menu. Open ‘zoom’, select ‘fit image to window’. As soon as you select it the image you opened will be enlarged to fit into the window 
  ![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/Screen2.png)    


3) Go to ‘Layer’ tab in the top menu. Select ‘Transparency’, Click on’ Add Alpha Chanel”. There will be no visible changes to the image you opened. 

  ![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/Screen3.png)    

  4) Select on’ Fuzzy Select tool’ on the tools section on your left top. Make a border on the image you opened. If you have made a border but it is not covering the whole image, then Shift + click on the part where there is no border. There will be a border on the image you opened. 

  ![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/Screen4.png)    



5) Go to ‘edit’ tab on the top menu. Click on ‘clear’.. As soon as you click it the background of the image will be cleared. 

  ![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/Screen5.png)    


6) Your foreground image is ready. Now it is time to export the file. For exporting go to ‘file’ tab on the top menu. Click on ‘export as’. As soon as you click a new window will appear. On the window you have to select the location where you have to export the file. If you want to change the image type then you have to go to ’select file type (by extension)’ and press ‘export’. Then you will be prompted by another window. Just press ‘export’ and the image will be exported 
  ![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/Screen6.png)    



Mask image creation 

1) for mask you can work on the same image you cleared the background. So first go to edit. Click on ’Fill with BG colour. BG stands for background. As soon as you click the background will become black. If the background does not become black press the exchange button on the tool section. It will exchange BG colour with FG colour    

  ![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/Screen7.png)    

  2) Go to ‘select’ in the top menu. Click on ‘invert’. There will be no visible changes to the image. 

3) Go to ‘edit’ in the top menu. Click on ‘Fill with FG colour. FG stands for foreground. As soon as you do that the image will become white. If the image does not become white press the exchange button on the tool section. It will exchange BG colour with FG colour. 

  ![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/Screen8.png)    


  4) Your mask is ready. Now it is time to export the file. For exporting go to ‘file’ tab on the top menu. Click on ‘export as’. As soon as you click a new window will appear. On the window you have to select the location where you have to export the file. If you want to change the image type then you have to go to ’select file type (by extension)’ and press ‘export’. Then you will be prompted by another window. Just press ‘export’ and the image will be exported   

  ![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/Screen9.png)  
 		

		
   ## c. how did you overlay the fg over bg and created 20 variants
     
     For each foreground image is overlayed randomly on each of the background image by using paste function of PIL library. We have chosen  (160x160) as dimension for background and (80x80) as dimension for foreground. While generating random position for 
     overlaying foreground on background, integer number is generated 20 times between 0 and 80 for both the x cordinate as well as y cordinate, using the function random.choice with  replace=False so that positions are not repeated. 
     
     In addition the each foreground image is horizonatlly flipped using function transpose from PIL library with option  PILImage.FLIP_LEFT_RIGHT and each of the flipped foreground image is also overlayed randomly on background image
     
     The  mask image corresponding to the foregorund image is overlayed on black background randomly but correpdoing position
     as the original foreground image is overlayed. Similary for flipped images the same process is repeated    
     
     The python script for generating overlay images is
     
     ![Script for generating fg bg iamges](/generate_fg_bg_images_jpg.py)
     
     It takes five arguments:
     
     argument1 : foreground images directory
     argument2: Background images directory
     argument3: Mask Images Directory
     argument4: Output directory for Overlayed foreground background image 
     argument5:  Output directory for Overlayed mask images on black backgournd
 

     
     
   ## d. how did you create your depth images? 
     
     The depth images are created using using the base notebook for depth model given modified. 
     
     - Changed code to process grey images by stacking 3 times 
     - For each batch, Iterate over images lying in fg_bg_jpg direcotry under the respective batch folders and store the images under depth_fg_bg_bng 
     - Used batch size as 128 and images are processed in a batch of 128 at a time 
     - Used plt.cla, plt.axis('off') and plt.clf so that Matplotlib saves the images faster
     
     The modified code is: ![Jupyter notebook for Dept hModel](/DenseDepth.ipynb)
     
 # 3. Show your dataset the way I have shown above in this readme     
     
     ## Python Code for generating gallery for  BackGround Scenary,ForeGround,coressponding Masks,ForeGround-Background,ForeGround-BackGround Mask and Depth Model Output images :

https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/GalleryUtil.ipynb


#### 1. Scenary or BackGround Images:

We have taken Home Images as background.Below is the gallery for the Home Images.A sample from the 100 backgrounds taken are shown below

![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/BackGrnd.png)

#### 2. ForeGround Images:

Foreground images are also download.Below is the gallery for the same.A sample taken from the 100 foreground images are shown below

![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/Foreground.png)

#### 3. Masked Images:

The Masked Images are generated for all the corresponding foreground images.A sample is taken for the 100 mask images  and are shown in the gallery below

![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/Mask.png)

#### 4. ForeGround-BackGround Images:

A total of 400K ForeGround-BackGround Images are generated by Overlaying foreground over background.Foreground is overlayed over background at twenty different positions.And also corresponding flip images for the foreground are generated for these twenty different positions.Algother since for one foreground there are  40 images on a specific background.A total of  40*100*100 =400K ForeGround-BackGround Images are generated.

A sample of  foreground-background images are shown in the Gallery below:

![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/foregroundbackground.png)


#### 5. ForeGround-BackGround Masked Images:
Similiar to the way the ForeGround-BackGround Images are generated by Overlaying Foreground over BackGround,The mask images are overlayed over the background.Since 20 different positions the Mask Images are overlayed and the corresponding  20 flipped images are also overlayed. So a total of 40*100*100(100 BackGround,100 ForeGround Images) i.e 400K masked images are generated.

A sample of  foreground-background masked images are shown in the gallery below:

![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/ForgroundBackGround_Mask.png)


#### 6. Depth Model Output Images:

A total of  400K depth model images are generated as an output of the depth model.

The gallery for sample of these 400K images are found below:

![](https://github.com/sudhakarmlal/EVA4/blob/master/Session14-15/Images/Depth.png)

5.Design Choices and Issues faced

- We have decided to take grey images as it will take less computation 
- We have decided to keep 10 batches of images so as to easier manage. For each batch, a separate zip file is created which will have fg, bg, fg_bg, mask, depth_fg_bg, black_msk. This will help in training as unit can be trained independently
- We have faced issues with colab and many of people report their good account is disabled. So we decided to run it only our own laptop with GPU and as the dataset is split 10 times, it was easier to upload to google drive, each batch of image
- The depth model when applied was very slow. So we changed teh batch size to 128 and chunks of files processed were 128 at a time. Also Pyplot methods clf() and cla() were used so that images were saved faster




      
      
      
      
      
      
      
      
      
     
     
     
     
     
     
     
     
     
     
     
     
