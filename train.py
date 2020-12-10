import torch
from torchvision import datasets,transforms
from utils import read_xml,tuple_to_position,generate_heatmap,AdamW
import argparse
import cv2
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('model/')
from net_vgg import *
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import time
import glob
import torch.nn as nn

# Hyper-parameter
parser = argparse.ArgumentParser()
parser.add_argument('--Exp_ID',type=int,default=1,help='The id of experiment if we use shell to search parameter')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nb_imgs', type=int, default=1000, help='Number of image, run <data_genrate.py> to change the value')
#parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=5, help='Patience')
parser.add_argument('--device_id', type=str,default="2", help='which gpu to use')
parser.add_argument('--details', action='store_true', default=True)
parser.add_argument('--model', type=str, default='vgg13_bn')
parser.add_argument('--val_ratio',type=float,default=0.1,help="validation data ratio")
parser.add_argument('--test_ratio',type=float,default=0.1,help="test data ratio")
parser.add_argument('--batch_size',type=int,default=32,help="train batch size")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] =args.device_id
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(' #######################\n',
'##The {:02d}th experiment##\n'.format(args.Exp_ID),
'#######################\n',
#'epochs: {}\n'.format(args.epochs),
'learning rate: {}\n'.format(args.lr),
'Weight decay: {}'.format(args.weight_decay),
)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Creat model
if args.model[0]=='v':
    print("Construct model {}".format(args.model))
    model=eval(args.model)()
    #model = nn.DataParallel(model, device_ids=args.device_id)
    model.cuda()
elif args.model=='resnet':
    pass

elif args.model=='wxjNet':
    pass
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
#filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

# optimizer = optim.SGD(model.parameters(),
#                        lr=args.lr,
#                       weight_decay=args.weight_decay,
#                        momentum=0.9,
#                       nesterov=True,)
# Data loading
def load_data():
    #print("loading data ...")
    if os.path.exists("data/image_set.npy") and os.path.exists("data/label_set.npy"):
        data=np.load("data/image_set.npy")
        label=np.load("data/label_set.npy")
        #print(data.shape,label.shape) (1000, 1, 224, 224) (1000, 10, 56, 56)
        Data=[]
        for i in range(data.shape[0]):
            Data.append([data[i],label[i]])
        nb_val = int(args.nb_imgs * args.val_ratio)
        nb_test = int(args.nb_imgs * args.test_ratio)
        nb_train = args.nb_imgs - nb_test - nb_val
        Train_data = Data[:nb_train]
        Val_data = Data[nb_train:nb_train + nb_val]
        Test_data = Data[nb_train + nb_val:]
        TrainDataloaders = torch.utils.data.DataLoader(Train_data, batch_size=args.batch_size, shuffle=False)
        ValDataloaders = torch.utils.data.DataLoader(Val_data, batch_size=1, shuffle=False)
        TestDataloaders = torch.utils.data.DataLoader(Test_data, batch_size=1, shuffle=False)
    else:

        img_path = "data/synth_data/data/"
        xml_path = "data/synth_data/annotation/"
        Data = []
        Image_set=[]
        Label_set=[]
        for i in range(args.nb_imgs):
            img = cv2.imread(img_path + str(i) + '.jpg', flags=cv2.IMREAD_GRAYSCALE)
            # cv2.imshow('image', img)
            # k = cv2.waitKey(0)
            # if k == 1:  # wait for ESC key to exit
            #     cv2.destroyAllWindows()
            img=img[np.newaxis,:,:]
            [xmin_tuple, ymin_tuple, xmax_tuple, ymax_tuple, label_tuple] = read_xml(file_name=str(i) + '.xml', path=xml_path)
            pos = tuple_to_position(xmin_tuple, ymin_tuple, xmax_tuple, ymax_tuple, label_tuple)
            label = generate_heatmap(height=224, width=224, position=pos, scale=4)
            #print(pos.keys(),label_tuple)
            Data.append([img, label])
            Image_set.append(img)
            Label_set.append(label)
            # plt.imshow(label[0],cmap='gray')
            # plt.show()
        Image_set=np.array(Image_set)
        Label_set=np.array(Label_set)
        np.save("data/image_set.npy",Image_set)
        np.save("data/label_set.npy",Label_set)
        nb_val=int(args.nb_imgs*args.val_ratio)
        nb_test=int(args.nb_imgs*args.test_ratio)
        nb_train=args.nb_imgs-nb_test-nb_val
        Train_data = Data[:nb_train]
        Val_data =Data[nb_train:nb_train+nb_val]
        Test_data =Data[nb_train+nb_val:]
        TrainDataloaders = torch.utils.data.DataLoader(Train_data, batch_size=args.batch_size, shuffle=False)
        ValDataloaders=torch.utils.data.DataLoader(Val_data, batch_size=1, shuffle=False)
        TestDataloaders = torch.utils.data.DataLoader(Test_data, batch_size=1, shuffle=False)
    #print("successfully load\n")
    return TrainDataloaders,ValDataloaders,TestDataloaders
TrainDataloaders,ValDataloaders,TestDataloaders=load_data()




def Loss(pred,gt,loss_type='MSE'):
    if loss_type=='MSE':
        return torch.norm(input=pred-gt, p=2)
    elif loss_type=="regular_MSE":
        pass

def compute_val_loss():
    """
    :return: average val data loss
    """
    Total_loss=0
    i=0
    for img, label in ValDataloaders:
        ht = model(img.float().cuda())
        loss = Loss(ht, label.float().cuda())
        Total_loss+=loss.item()
        i+=1
    return Total_loss/i

def compute_test_loss():
    """
    :return: average test data loss
    """
    Total_loss=0
    i=0
    for img, label in TestDataloaders:
        ht = model(img.float().cuda())
        loss = Loss(ht, label.float().cuda())
        Total_loss+=loss.item()
        i+=1
    return Total_loss/i

# Training
def train(load=False):
    if load:
        model.load_state_dict(torch.load('14.pkl'))
    else:
        remove_pkl()
    t_total=time.time()
    loss_values=[]
    bad_counter=0
    best = float("inf")
    best_epoch=0
    loss=0
    for epoch in range(args.epochs):
        for img, label in TrainDataloaders:
            # label=label.float().cuda()
            # label=nn.functional.relu(label - 0.6)/0.4
            optimizer.zero_grad()
            ht=model(img.float().cuda())
            loss=Loss(ht,label.float().cuda())
            print(loss.item())
            loss.backward()
            optimizer.step()

        print("###########################################")
        print("Epoch {:04d}, train loss {:.4f}, val loss {:.4f}, test loss {:.4f}".format(epoch + 1,loss, compute_val_loss(), compute_test_loss()))
        print("###########################################")
        loss_values.append(compute_val_loss())

        if loss_values[-1]<best:
            torch.save(model.state_dict(), '{}.pkl'.format(epoch + 1))
            best=loss_values[-1]
            best_epoch=epoch+1
            bad_counter=0
        else:
            bad_counter+=1
        if bad_counter==args.patience:
            break
    files=glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
    torch.save(model.state_dict(), 'net_model/{}.pkl'.format(args.Exp_ID))
    print("Test loss {:.4f}\n".format(compute_test_loss()))

def filter(x):
    filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
   # filter = np.array([[0.5, 0.6, 0.5], [0.6, 0.8, 0.6], [0.5, 0.6, 0.5]])
    kernel = np.zeros([10, 10, 3, 3])
    for i in range(10):
        kernel[i, i, :, :] = filter
    heatmap_filter = nn.Conv2d(10, 10, kernel_size=3, padding=1)
    heatmap_filter.weight=nn.Parameter(data=torch.Tensor(kernel).float().cuda(),requires_grad=False)
    heatmap_filter.bias=nn.Parameter(data=torch.Tensor([0]*10).float().cuda(),requires_grad=False)
    return heatmap_filter(x)

CHAR_as_ind={}
for i in range(1,11):
    CHAR_as_ind[i]=str(i-1)
def get_text_from_line(line):
    D=3 #the max value for center distance variant range
    line=np.max(line,axis=0)
    center=[]
    detected=0
    start=0
    for i in range(len(line)):
        if line[i] !=0 and detected==0:
            start=i
            detected=1
        elif line[i]==0 and detected==1:
            center.append((i+start)/2)
            detected=0
        else:
            pass
    #print(line,center)
    center_dis=[center[i+1]-center[i] for i in range(len(center)-1)]
    #print(center,center_dis)
    connect=[0]*len(center_dis)
    for i in range(1,len(center_dis)):
        if center_dis[i]-center_dis[i-1]<-D:
            connect[i-1]=0
            connect[i]=1
        elif center_dis[i]-center_dis[i-1]>D:
            connect[i - 1] = 1
            connect[i] = 0
        else:
            connect[i - 1] = 1
            connect[i] = 1
    text=CHAR_as_ind[line[int(center[0])]]
    for i in range(len(connect)):
        if connect[i]==0:
            text+=" "
        text+=CHAR_as_ind[line[int(center[i+1])]]
    return text

def get_text(heatmap):
    text_line=[]
    detected=0
    start=0
    for i in range(len(heatmap)):
        if np.any(heatmap[i] != 0) and detected==0:
            start=i
            detected=1
        elif np.all(heatmap[i]==0) and detected==1:
            #text_line+=get_text_from_line(heatmap[start:final+1,:])
            text_line.append(get_text_from_line(heatmap[start:i+1,:]))
            detected=0
        else:
            pass
    return text_line


def test():
    # model_CKPT=torch.load('1.pkl')
    # model.load_state_dict(model_CKPT['state_dict'])
    model.load_state_dict(torch.load('net_model/saved_model/1.pkl'))
    #Test_set=[0,11]
    a = time.time()
    for k,(img,label) in enumerate(TestDataloaders):
        if k<=1:

            ht = model(img.float().cuda(),"Train")
            # for _ in range(5):
            #     ht=filter(ht)
            heatmap = ht.cpu().detach().numpy()[0]
            print(heatmap.shape)
            #print(get_text(heatmap))

            # plt.imshow(heatmap)
            # plt.show()
            detail=0
            if not detail:
                fuse_heatmap=np.concatenate(heatmap,1)
                for i in range(1,10):
                    fuse_heatmap[:,i*56]=1
                ht0_4=fuse_heatmap[:,:56*5]
                ht5_9 = fuse_heatmap[:, 56 * 5:]
                plt.figure()
                plt.imshow(ht0_4)
                plt.figure()
                plt.imshow(ht5_9)
                plt.show()
            #print(fuse_heatmap.shape)
            #label = label.cpu().detach().numpy()
            else:
                for i in range(10):
                    plt.figure()
                    plt.imshow(heatmap[i])
                    # plt.figure()
                    # plt.imshow(label[0][i])
                #plt.show()
            if k==9:
                #print(text)
                break
    print(10 / (time.time() - a))

def test_img(path='cap.jpg'):
    model.load_state_dict(torch.load('net_model/1.pkl'))
    img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
    h,w=img.shape
    img=cv2.resize(img,dsize=(int(224*w/h),224))
    plt.figure()
    plt.imshow(img)
    img=img[np.newaxis,:,:]
    img = img[np.newaxis, :, :]
    img=torch.Tensor(img).float().cuda()
    ht = model(img)
    heatmap = ht.cpu().detach().numpy()
    fuse_heatmap = np.concatenate(tuple(heatmap[0]), 1)
    for i in range(1, 10):
        fuse_heatmap[:, i * 56] = 1
    ht0_4 = fuse_heatmap[:, :56 * 5]
    ht5_9 = fuse_heatmap[:, 56 * 5:]
    plt.figure()
    plt.imshow(ht0_4)
    plt.figure()
    plt.imshow(ht5_9)
    plt.show()

def remove_pkl():
    files = glob.glob('*.pkl')
    for file in files:
        os.remove(file)

def print_parameter():
    #model.load_state_dict(torch.load('net_model/1.pkl'))
    print("optimizer's state_dict")
    for var_name in model.state_dict():
        print(var_name, "\t", model.state_dict()[var_name].size())
    para=model.state_dict()["heatmap.weight"]
    para = para.cpu().detach().numpy()
    para=para[:,:,0,0]
    print(para.shape)
    np.save('para.npy',para)
    # plt.figure()
    # plt.imshow(para)
    # plt.show()
#train()
test()
#print_parameter()
#test_img()
# remove_pkl()