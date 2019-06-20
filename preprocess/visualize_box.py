import cv2
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.dirname(current_dir)))
sys.path.append(os.getcwd())

img_root_folder = './dataset/originalPics'
anno_txt_path = './dataset/FDDB-folds/FDDB-fold-01-ellipseList.txt'

def add_anno_on_img(img, annos):
    for anno in annos:
        center = (int(anno[3]), int(anno[4]))
        axes = (int(anno[1]), int(anno[0]))
        angle = anno[2]
        width = 2*axes[0] * 4 // 3
        height = 2*axes[1] * 4 //3 
        cv2.ellipse(img, center, axes, -angle, 0, 360, (255, 0, 0), 3)

        center_x, center_y = center
        for x_offset in [-1, 0, 1]:
            for y_offset in [-1, 0, 1]:
                if((x_offset, y_offset) == (0, 0)):
                    color = (255, 0, 0)
                    linewidth = 3
                else:
                    color = (0, 255, 0)
                    linewidth = 1
                left = center_x - width//2 + x_offset*width//3
                right = center_x + width//2 + x_offset*width//3
                top = center_y - height//2 + y_offset*height//3
                bottom = center_y + height//2 + y_offset*height//3

                cv2.rectangle(img, (left, top), (right, bottom), color, linewidth)
    cv2.imshow('img', img)
    cv2.waitKey()

f_anno =  open(anno_txt_path, 'r')
while(True):
    img_name = f_anno.readline().strip()
    print(img_name)
    img_path = img_name + '.jpg'
    img = cv2.imread(os.path.join(img_root_folder, img_path))
    if(img_name == ''):
        break
    # cv2.imshow('img', img)
    # cv2.waitKey()
    annos_num = int(f_anno.readline())
    annos = []
    for i in range(annos_num):
        annos.append([float(x) for x in f_anno.readline().split()])
    add_anno_on_img(img, annos)

f_anno.close()
