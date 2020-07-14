import cv2
import numpy as np
import pandas as pd

class DrawBoundingBox:
    def __init__(self, lapse_folder, out_folder):
        self.ref_point = []
        self.lapse_folder = lapse_folder
        self.out_folder = out_folder
        self.df = pd.DataFrame()
        self.record = []
        self.image = None
    
    def shape_selection(self, event, x, y, flags, param):
        '''
            On mouse even callback for drawing bounding box manually on image and save it to csv
        '''
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_point = [(x, y)]
            self.record.append(x)
            self.record.append(y)
            hold = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.ref_point.append((x, y))
            self.record.append(x)
            self.record.append(y)
            # draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.ref_point[0], self.ref_point[1], (0, 255, 0), 2)

            #Reset the ref point
            self.ref_point = []
            cv2.imshow("image", self.image)
    
    
    
    def run(self, filename):
        vid = cv2.VideoCapture(self.lapse_folder + filename)
        self.record.append(filename)
        
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.shape_selection)
        
        try:
            while vid.isOpened():
                _, self.image = vid.read()
                clone = self.image.copy()
                # keep looping until the 'q' key is pressed
                key = None
                while key != ord("q"):
                    # display the image and wait for a keypress
                    cv2.imshow("image", self.image)
                    key = cv2.waitKey(1) & 0xFF

                    # press 'r' to reset the window
                    if key == ord("r"):
                        record = [filename]
                        self.image = clone.copy()

                    # Continue video
                    if key == 13 or key == -1:
                        # save to pandas dataframe
                        self.df = self.df.append([self.record], ignore_index=True)
                        self.record = [filename]
                        break
                # Exit loop if it's q
                if key == ord("q"):
                    break
                    
            if self.df.shape[0] > 0:
                self.df.columns = ['name', 'x1','y1','x2','y2']
                self.df.to_csv(self.out_folder + filename.replace('.mpg','').replace('Timelapses', 'bbox') + '_bbox.csv',index=False)
            
        except Exception as e:
            print(e)
            
        # close all open windows
        vid.release()
        cv2.destroyAllWindows() 

if __name__ == '__main__':
    x = DrawBoundingBox("./Timelapses/", './bbox/')
    # Run it on 1 video
    x.run('asun_1_20200710101504.mpg')