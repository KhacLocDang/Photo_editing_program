import cv2
import numpy as np
import dlib
from math import hypot
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

# Apply the transformations needed
import torchvision.transforms as T


def nothing(x):
    pass
s1=0
s2=0
Control = np.zeros([100,500], np.uint8)
cv2.namedWindow('Control')
cv2.createTrackbar('Sti-horr', 'Control',0,1,nothing)
cv2.createTrackbar('Fillter', 'Control',0,1,nothing)
cv2.createTrackbar('AEF', 'Control',0,1,nothing)
cv2.createTrackbar('Finish', 'Control',0,1,nothing)


while True:
	cv2.imshow('Control',Control)
	s1 = cv2.getTrackbarPos('Sti-horr','Control')
	s2 = cv2.getTrackbarPos('Fillter','Control')
	s3 = cv2.getTrackbarPos('AEF','Control')
	s4 = cv2.getTrackbarPos('Finish','Control')
	cv2.waitKey(1)
	if s4  == 1:
		break

        


if s1==0:
	# Create a black Remode, a window
	video = cv2.VideoCapture(0) 
	Remode = np.zeros([100, 300], np.uint8)
	cv2.namedWindow('Remode')

	# Loading Camera and Nose image and Creating mask
	nose_image = cv2.imread("pig_nose.png")
	_, frame = video.read()
	rows, cols, _ = frame.shape
	nose_mask = np.zeros((rows, cols), np.uint8)
	# Loading Face detector
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


	# create switch for ON/OFF functionality

	cv2.createTrackbar('TurnON-OFF', 'Remode',0,1,nothing)
	cv2.createTrackbar('pignose', 'Remode',0,1,nothing)


	while True:
		_, frame = video.read()
		cv2.imshow('Remode',Remode)    

		s = cv2.getTrackbarPos('TurnON-OFF','Remode')
		if s==1:
			break
		
		pignose= cv2.getTrackbarPos('pignose','Remode')
		if pignose==1:
			nose_mask.fill(0)
			gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = detector(frame)
			for face in faces:
				landmarks = predictor(gray_frame, face)
				# Nose coordinates
				top_nose = (landmarks.part(29).x, landmarks.part(29).y)
				center_nose = (landmarks.part(30).x, landmarks.part(30).y)
				left_nose = (landmarks.part(31).x, landmarks.part(31).y)
				right_nose = (landmarks.part(35).x, landmarks.part(35).y)

				nose_width = int(hypot(left_nose[0] - right_nose[0],
	                           left_nose[1] - right_nose[1]) * 1.7)
				nose_height = int(nose_width * 0.77)

				top_left = (int(center_nose[0] - nose_width / 2),
	                              int(center_nose[1] - nose_height / 2))
				bottom_right = (int(center_nose[0] + nose_width / 2),
	                       int(center_nose[1] + nose_height / 2))
				nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
				nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
				_, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
				nose_area = frame[top_left[1]: top_left[1] + nose_height,top_left[0]: top_left[0] + nose_width]
				nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
				final_nose = cv2.add(nose_area_no_nose, nose_pig)
				frame[top_left[1]: top_left[1] + nose_height,
	                    top_left[0]: top_left[0] + nose_width] = final_nose
		        
		
		cv2.imshow("Capturing",frame)
		key = cv2.waitKey(1)
		if key == ord('s'):
			cv2.imwrite("filename.png",frame)
		
		
		
	# 7. image saving
	#showPic = cv2.imwrite("filename.png",frame)
	#print(showPic)
	# 8. shutdown the camera
	video.release()
	cv2.destroyAllWindows()
if s1==1:
	cap = cv2.VideoCapture(0)

	panel = np.zeros([100, 700], np.uint8)
	cv2.namedWindow('panel')

	cv2.createTrackbar('TurnON-OFF', 'panel',0,1,nothing)
	cv2.createTrackbar('L - h', 'panel', 0, 179, nothing)
	cv2.createTrackbar('U - h', 'panel', 179, 179, nothing)

	cv2.createTrackbar('L - s', 'panel', 0, 255, nothing)
	cv2.createTrackbar('U - s', 'panel', 255, 255, nothing)

	cv2.createTrackbar('L - v', 'panel', 0, 255, nothing)
	cv2.createTrackbar('U - v', 'panel', 255, 255, nothing)

	cv2.createTrackbar('S ROWS', 'panel', 0, 480, nothing)
	cv2.createTrackbar('E ROWS', 'panel', 480, 480, nothing)
	cv2.createTrackbar('S COL', 'panel', 0, 640, nothing)
	cv2.createTrackbar('E COL', 'panel', 640, 640, nothing)

	while True:
	    _, frame = cap.read()

	    s_r = cv2.getTrackbarPos('S ROWS', 'panel')
	    e_r = cv2.getTrackbarPos('E ROWS', 'panel')
	    s_c = cv2.getTrackbarPos('S COL', 'panel')
	    e_c = cv2.getTrackbarPos('E COL', 'panel')

	    roi = frame[s_r: e_r, s_c: e_c]

	    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

	    l_h = cv2.getTrackbarPos('L - h', 'panel')
	    u_h = cv2.getTrackbarPos('U - h', 'panel')
	    l_s = cv2.getTrackbarPos('L - s', 'panel')
	    u_s = cv2.getTrackbarPos('U - s', 'panel')
	    l_v = cv2.getTrackbarPos('L - v', 'panel')
	    u_v = cv2.getTrackbarPos('U - v', 'panel')

	    lower_green = np.array([l_h, l_s, l_v])
	    upper_green = np.array([u_h, u_s, u_v])

	    mask = cv2.inRange(hsv, lower_green, upper_green)
	    mask_inv = cv2.bitwise_not(mask)

	    bg = cv2.bitwise_and(roi, roi, mask=mask)
	    fg = cv2.bitwise_and(roi, roi, mask=mask_inv)

	    cv2.imshow('bg', bg)

	    cv2.imshow('fg', fg)

	    cv2.imshow('panel', panel)

	    key = cv2.waitKey(1)
	    if key == ord('s'):
	        cv2.imwrite("filename.png",bg)

	    s = cv2.getTrackbarPos('TurnON-OFF','panel')
	    if s==1:
	    	break

   
	    

	cap.release()
	cv2.destroyAllWindows()



if s2==1:
	img = cv2.imread('filename.png')


	def verify_alpha_channel(img):
	    try:
	        img.shape[3] 
	    except IndexError:
	        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
	    return img


	def apply_hue_saturation(img, alpha, beta):
	    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	    h, s, v = cv2.split(hsv_image)
	    s.fill(199)
	    v.fill(255)
	    hsv_image = cv2.merge([h, s, v])

	    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
	    img = verify_alpha_channel(img)
	    out = verify_alpha_channel(out)
	    cv2.addWeighted(out, 0.25, img, 1.0, .23, img)
	    return img


	def apply_color_overlay(img, intensity=0.5, blue=0, green=0, red=0):
	    img = verify_alpha_channel(img)
	    img_h, img_w, img_c = img.shape
	    sepia_bgra = (blue, green, red, 1)
	    overlay = np.full((img_h, img_w, 4), sepia_bgra, dtype='uint8')
	    cv2.addWeighted(overlay, intensity, img, 1.0, 0, img)
	    return img


	def apply_sepia(img, intensity=0.5):
	    img = verify_alpha_channel(img)
	    img_h, img_w, img_c = img.shape
	    sepia_bgra = (20, 66, 112, 1)
	    overlay = np.full((img_h, img_w, 4), sepia_bgra, dtype='uint8')
	    cv2.addWeighted(overlay, intensity, img, 1.0, 0, img)
	    return img


	def alpha_blend(img_1, img_2, mask):
	    alpha = mask/255.0 
	    blended = cv2.convertScaleAbs(img_1*(1-alpha) + img_2*alpha)
	    return blended


	def portrait_mode(img):
	    #cv2.imshow('img', img)
	    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	    _, mask = cv2.threshold(gray, 120,255,cv2.THRESH_BINARY)

	    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
	    blured = cv2.GaussianBlur(img, (21,21), 11)
	    blended = alpha_blend(img, blured, mask)
	    img = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
	    return img

	def change_brightness(img, alpha, beta):
	    img = np.asarray(alpha*img + beta, dtype='uint8')   
	    img[img>255] = 255
	    img[img<0] = 0
	    return img


	Remode = np.zeros([100,500], np.uint8)
	cv2.namedWindow('Remode')
	cv2.createTrackbar('TurnON-OFF', 'Remode',0,1,nothing)
	cv2.createTrackbar('hue_sat', 'Remode',0,1,nothing)
	cv2.createTrackbar('sepia', 'Remode',0,1,nothing)
	cv2.createTrackbar('overlay', 'Remode',0,1,nothing)
	cv2.createTrackbar('portrait', 'Remode',0,1,nothing)
	cv2.createTrackbar('gray', 'Remode',0,1,nothing)
	cv2.createTrackbar('shining', 'Remode',0,1,nothing)
	cv2.createTrackbar('dark', 'Remode',0,1,nothing)
	cv2.imshow('Remode',Remode)


	while(True):
	    
	    #cv2.imshow('Remode',Remode)
	    s = cv2.getTrackbarPos('TurnON-OFF','Remode')
	    if s==1:
	        break
	    key = cv2.waitKey(20)
	    
	    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) 

	    s = cv2.getTrackbarPos('hue_sat','Remode')
	    if s==1:
	        hue_sat = apply_hue_saturation(img.copy(), alpha=3, beta=3)
	        cv2.imshow('hue_sat', hue_sat)
	        if key == ord('s'):
	            cv2.imwrite("filename.png",hue_sat)
	        
	  
	    s = cv2.getTrackbarPos('sepia','Remode')
	    if s==1:
	        sepia = apply_sepia(img.copy(), intensity=.8)
	        cv2.imshow('sepia',sepia)
	        if key == ord('s'):
	            cv2.imwrite("filename.png",sepia)


	    s = cv2.getTrackbarPos('overlay','Remode')
	    if s==1:
	        color_overlay = apply_color_overlay(img.copy(), intensity=.8, red=50, green=10)
	        cv2.imshow('color_overlay',color_overlay)
	        if key == ord('s'):
	            cv2.imwrite("filename.png",color_overlay)

	    s = cv2.getTrackbarPos('portrait','Remode')
	    if s==1:
	        portrait = portrait_mode(img.copy())
	        cv2.imshow('portrait',portrait)
	        if key == ord('s'):
	            cv2.imwrite("filename.png",portrait)

	    
	    s = cv2.getTrackbarPos('gray','Remode')
	    if s==1:
	        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	        cv2.imshow('gray',gray)
	        if key == ord('s'):
	            cv2.imwrite("filename.png",gray)
	    
	    s = cv2.getTrackbarPos('shining','Remode')
	    if s==1:
	        shining = change_brightness(img.copy(), alpha=.9, beta=20)
	        cv2.imshow('shining',shining)
	        if key == ord('s'):
	            cv2.imwrite("filename.png",shining)
	    
	    s = cv2.getTrackbarPos('dark','Remode')
	    if s==1:
	        dark = change_brightness(img.copy(), alpha=.3, beta=35)
	        cv2.imshow('dark',dark)
	        if key == ord('s'):
	            cv2.imwrite("filename.png",dark)
	    
	    #cv2.waitKey(20)



	cv2.destroyAllWindows()




if s3==1:
    def nothing(x):
        pass
    s1=0
    s2=0
    Control = np.zeros([100,500], np.uint8)
    cv2.namedWindow('Control')
    cv2.createTrackbar('DeleteBG', 'Control',0,1,nothing)
    cv2.createTrackbar('GrayBG', 'Control',0,1,nothing)
    cv2.createTrackbar('ChangeBG(Beta)', 'Control',0,1,nothing)
    cv2.createTrackbar('Finish', 'Control',0,1,nothing)

    while True:
            cv2.imshow('Control',Control)
            s3 = cv2.getTrackbarPos('DeleteBG','Control')
            s4 = cv2.getTrackbarPos('GrayBG','Control')
            s5 = cv2.getTrackbarPos('ChangeBG(Beta)','Control')
            s10 = cv2.getTrackbarPos('Finish','Control')
            cv2.waitKey(1)
            if s10  == 1:
                    break

    if s3==1:
        # Define the helper function
        def decode_segmap(image, source, bgimg, nc=21):
          
          label_colors = np.array([(0, 0, 0),  # 0=background
                       # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                       (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                       # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                       (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                       # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                       (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                       # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                       (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

          r = np.zeros_like(image).astype(np.uint8)
          g = np.zeros_like(image).astype(np.uint8)
          b = np.zeros_like(image).astype(np.uint8)
          
          for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
            
            
          rgb = np.stack([r, g, b], axis=2)
          
          # Load the foreground input image 
          foreground = cv2.imread(source)

          # Load the background input image 
          background = cv2.imread(bgimg)

          # Change the color of foreground image to RGB 
          # and resize images to match shape of R-band in RGB output map
          foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
          background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
          foreground = cv2.resize(foreground,(r.shape[1],r.shape[0]))
          background = cv2.resize(background,(r.shape[1],r.shape[0]))
          

          # Convert uint8 to float
          foreground = foreground.astype(float)
          background = background.astype(float)

          # Create a binary mask of the RGB output map using the threshold value 0
          th, alpha = cv2.threshold(np.array(rgb),0,255, cv2.THRESH_BINARY)

          # Apply a slight blur to the mask to soften edges
          alpha = cv2.GaussianBlur(alpha, (7,7),0)

          # Normalize the alpha mask to keep intensity between 0 and 1
          alpha = alpha.astype(float)/255

          # Multiply the foreground with the alpha matte
          foreground = cv2.multiply(alpha, foreground)  
          
          # Multiply the background with ( 1 - alpha )
          background = cv2.multiply(1.0 - alpha, background)  
          
          # Add the masked foreground and background
          outImage = cv2.add(foreground, background)

          # Return a normalized output image for display
          return outImage/255

        def segment(net, path, bgimagepath, show_orig=True, dev='cuda'):
          img = Image.open(path)
          
          if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
          # Comment the Resize and CenterCrop for better inference results
          trf = T.Compose([T.Resize(400), 
                           #T.CenterCrop(224), 
                           T.ToTensor(), 
                           T.Normalize(mean = [0.485, 0.456, 0.406], 
                                       std = [0.229, 0.224, 0.225])])
          inp = trf(img).unsqueeze(0).to(dev)
          out = net.to(dev)(inp)['out']
          om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
          
          rgb = decode_segmap(om, path, bgimagepath)
            
          plt.imshow(rgb); plt.axis('off');plt.savefig('BLankBG.png'); plt.show()
          

        dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

        segment(dlab, './filename.png','./blank.jpg', show_orig=False)
    if s4==1:
        # Define the helper function
        def decode_segmap(image, source, nc=21):
          
          label_colors = np.array([(0, 0, 0),  # 0=background
                       # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                       (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                       # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                       (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                       # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                       (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                       # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                       (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

          r = np.zeros_like(image).astype(np.uint8)
          g = np.zeros_like(image).astype(np.uint8)
          b = np.zeros_like(image).astype(np.uint8)
          
          for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
            
            
          rgb = np.stack([r, g, b], axis=2)

          # Load the foreground input image 
          foreground = cv2.imread(source)

          # Change the color of foreground image to RGB 
          # and resize image to match shape of R-band in RGB output map  
          foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
          foreground = cv2.resize(foreground,(r.shape[1],r.shape[0]))
          
          # Create a background image by copying foreground and converting into grayscale
          background = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
          
          # convert single channel grayscale image to 3-channel grayscale image
          background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
          
          # Convert uint8 to float
          foreground = foreground.astype(float)
          background = background.astype(float)

          # Create a binary mask of the RGB output map using the threshold value 0
          th, alpha = cv2.threshold(np.array(rgb),0,255, cv2.THRESH_BINARY)

          # Apply a slight blur to the mask to soften edges
          alpha = cv2.GaussianBlur(alpha, (7,7),0)

          # Normalize the alpha mask to keep intensity between 0 and 1
          alpha = alpha.astype(float)/255

          # Multiply the foreground with the alpha matte
          foreground = cv2.multiply(alpha, foreground)  
          
          # Multiply the background with ( 1 - alpha )
          background = cv2.multiply(1.0 - alpha, background)  
          
          # Add the masked foreground and background
          outImage = cv2.add(foreground, background)

          # Return a normalized output image for display
          return outImage/255

        def segment(net, path, show_orig=True, dev='cuda'):
          img = Image.open(path)
          
          if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
          # Comment the Resize and CenterCrop for better inference results
          trf = T.Compose([T.Resize(450), 
                           #T.CenterCrop(224), 
                           T.ToTensor(), 
                           T.Normalize(mean = [0.485, 0.456, 0.406], 
                                       std = [0.229, 0.224, 0.225])])
          inp = trf(img).unsqueeze(0).to(dev)
          out = net.to(dev)(inp)['out']
          om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
          
          rgb = decode_segmap(om, path)
            
          plt.imshow(rgb); plt.axis('off');plt.savefig('grayscale.png'); plt.show()
          

        dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()


        segment(dlab, './filename.png', show_orig=False)
    if s5==1:
        # Define the helper function
        def decode_segmap(image, source, bgimg, nc=21):
          
          label_colors = np.array([(0, 0, 0),  # 0=background
                       # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                       (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                       # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                       (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                       # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                       (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                       # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                       (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

          r = np.zeros_like(image).astype(np.uint8)
          g = np.zeros_like(image).astype(np.uint8)
          b = np.zeros_like(image).astype(np.uint8)
          
          for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
            
            
          rgb = np.stack([r, g, b], axis=2)
          
          # Load the foreground input image 
          foreground = cv2.imread(source)

          # Load the background input image 
          background = cv2.imread(bgimg)

          # Change the color of foreground image to RGB 
          # and resize images to match shape of R-band in RGB output map
          foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
          background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
          foreground = cv2.resize(foreground,(r.shape[1],r.shape[0]))
          background = cv2.resize(background,(r.shape[1],r.shape[0]))
          

          # Convert uint8 to float
          foreground = foreground.astype(float)
          background = background.astype(float)

          # Create a binary mask of the RGB output map using the threshold value 0
          th, alpha = cv2.threshold(np.array(rgb),0,255, cv2.THRESH_BINARY)

          # Apply a slight blur to the mask to soften edges
          alpha = cv2.GaussianBlur(alpha, (7,7),0)

          # Normalize the alpha mask to keep intensity between 0 and 1
          alpha = alpha.astype(float)/255

          # Multiply the foreground with the alpha matte
          foreground = cv2.multiply(alpha, foreground)  
          
          # Multiply the background with ( 1 - alpha )
          background = cv2.multiply(1.0 - alpha, background)  
          
          # Add the masked foreground and background
          outImage = cv2.add(foreground, background)

          # Return a normalized output image for display
          return outImage/255

        def segment(net, path, bgimagepath, show_orig=True, dev='cuda'):
          img = Image.open(path)
          
          if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
          # Comment the Resize and CenterCrop for better inference results
          trf = T.Compose([T.Resize(400), 
                           #T.CenterCrop(224), 
                           T.ToTensor(), 
                           T.Normalize(mean = [0.485, 0.456, 0.406], 
                                       std = [0.229, 0.224, 0.225])])
          inp = trf(img).unsqueeze(0).to(dev)
          out = net.to(dev)(inp)['out']
          om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
          
          rgb = decode_segmap(om, path, bgimagepath)
            
          plt.imshow(rgb); plt.axis('off'); plt.savefig('ChangeBG.png'); plt.show()
          

        dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()


        segment(dlab, './filename.png','./BG2.jpg', show_orig=False)



