
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


#### This Mapping is required in order to generate 40,000 data objects for each of the  bg,fg,mask(fg-bg),depth,fg-bg in order to feed to the train_dataloader.

  
        Category : bg_jpg  Mean : 0.739088,  Std :  0.265235	
	Category : depth_fg_bg_jpg  Mean : 0.777681,  Std :  0.311899
	Category : fg_bg_jpg  Mean : 0.729304,  Std :  0.271675
	Category : fg_jpg  Mean : 0.851283,  Std :  0.270902
	Category : mask_black_jpg  Mean : 0.068877,  Std :  0.249513
	Category : mask_jpg  Mean : 0.276214,  Std :  0.436924
  
  
  
     ## a. how were fg created with transparency
     
     We used GIMP tool to generate foreground images with transparency. The full steps with screenshots are givne in:     
    

     ## b. how were masks created for fgs
     
  Masks were created using GIP tool, Full steps with screenshots are given in
  
  Creation of the data set Foreground image creation with transparency

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




      
      
      
      
      
      
      
      
      
     
     
     
     
     
     
     
     
     
     
     
     
