import os, shutil
from sklearn.model_selection import train_test_split


val_size = 0.2
#test_size = 0.2
postfix = 'jpg'
imgpath = r'./data/image'
txtpath =  r'./data/label'



output_train_img_folder =r'./data/tea3/images/train'
output_val_img_folder =  r'./data/tea3/images/val'
output_train_txt_folder =  r'./data/tea3/labels/train'
output_val_txt_folder =  r'./data/tea3/labels/val'

os.makedirs(output_train_img_folder, exist_ok=True)
os.makedirs(output_val_img_folder, exist_ok=True)
os.makedirs(output_train_txt_folder, exist_ok=True)
os.makedirs(output_val_txt_folder, exist_ok=True)


listdir = [i for i in os.listdir(txtpath) if 'txt' in i]
train, val = train_test_split(listdir, test_size=val_size, shuffle=True, random_state=0)

#todo：需要test放开

# train, test = train_test_split(listdir, test_size=test_size, shuffle=True, random_state=0)
# train, val = train_test_split(train, test_size=val_size, shuffle=True, random_state=0)

# 图片name-path字典
image_dict = {}
for dirpath,_,filenames in os.walk(imgpath):
    for filename in filenames:
        full_path = os.path.join(dirpath,filename)
        image_dict[filename] = full_path

for i in train:
    # img_source_path = os.path.join(imgpath, image_dict['{}.{}'.format(i[:-4], postfix)])
    img_source_path = image_dict['{}.{}'.format(i[:-4], postfix)]
    txt_source_path = os.path.join(txtpath, i)

    img_destination_path = os.path.join(output_train_img_folder, '{}.{}'.format(i[:-4], postfix))
    txt_destination_path = os.path.join(output_train_txt_folder, i)

    shutil.copy(img_source_path, img_destination_path)
    shutil.copy(txt_source_path, txt_destination_path)

for i in val:
    # img_source_path = os.path.join(imgpath, image_dict['{}.{}'.format(i[:-4], postfix)])
    img_source_path = image_dict['{}.{}'.format(i[:-4], postfix)]
    txt_source_path = os.path.join(txtpath, i)

    img_destination_path = os.path.join(output_val_img_folder, '{}.{}'.format(i[:-4], postfix))
    txt_destination_path = os.path.join(output_val_txt_folder, i)

    shutil.copy(img_source_path, img_destination_path)
    shutil.copy(txt_source_path, txt_destination_path)


#
# for i in train:
#     shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), r'E:\1-cheng\4-yolo-dataset-daizuo\multi-classify\bird-boat-horse-aeroplane-sheep\dataset20231219/images/train/{}.{}'.format(i[:-4], postfix))
#     shutil.copy('{}/{}'.format(txtpath, i), r'E:\1-cheng\4-yolo-dataset-daizuo\multi-classify\bird-boat-horse-aeroplane-sheep\dataset20231219/labels/train/{}'.format(i))
#
# for i in val:
#     shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), r'E:\1-cheng\4-yolo-dataset-daizuo\multi-classify\bird-boat-horse-aeroplane-sheep\dataset20231219/images/val/{}.{}'.format(i[:-4], postfix))
#     shutil.copy('{}/{}'.format(txtpath, i), r'E:\1-cheng\4-yolo-dataset-daizuo\multi-classify\bird-boat-horse-aeroplane-sheep\dataset20231219/labels/val/{}'.format(i))

#todo:需要test则放开

# for i in test:
#     shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), 'images/test/{}.{}'.format(i[:-4], postfix))
#     shutil.copy('{}/{}'.format(txtpath, i), 'labels/test/{}'.format(i))
