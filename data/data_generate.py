import numpy as np
import cv2
import sys
sys.path.append('../')
from utils import make_xml
def generate_one_pic(size,bgParameter=None,label=None,position=None,number=1,random=True,background=True,name='img1'):
    """
    函数生成一张含有样本的图以及对应的xml文件
    :param size: 生成图片的大小
    :param bgParameter: 背景参数
    :param label: 图片中各个样本的标签
    :param position: 图片中各个样本的位置
    :param number: 图片中样本数量
    :param random: 随机生成还是按position的参数生成
    :param background: 背景参数是否随机
    :param name：生成图片名字
    :return: None
    """
    if random:
        pass
    else:
        pass


def generate_one_line(L=4,division_proba=0.3):
    """
    :param L: 这一行有多少个字符
    :param division_proba: 将这一行分成两部分的概率
    :return:
    """
    xmin, xmax, ymin, ymax, label = [], [], [], [], []
    [division_proba, L] = [division_proba, L] if L > 4 else [0, 4]
    divide_point = np.random.randint(1, L - 1) if np.random.random() < division_proba else -1
    img_list=[]
    width=[]
    k=np.random.randint(15,21,1)[0]
    div = np.ones((400+2*k, 200))*255
    for i in range(10):
        m=cv2.imread('basic_char/'+str(i)+'.PNG',flags=cv2.IMREAD_GRAYSCALE)
        m = np.pad(m, ((k,k), (k,k)), 'constant', constant_values=((255, 255), (255, 255)))
        img_list.append(m)
        width.append(m.shape[1])

    img=[]
    char = np.random.randint(0, 10, L)
    y_start=0

    for i in range(L):
        if i!=divide_point:
            label.append(str(char[i]))
            xmin.append(0)
            ymin.append(y_start)
            xmax.append(400+2*k-1)
            ymax.append(y_start+width[char[i]]-1)
            img.append(img_list[char[i]])
            y_start+=width[char[i]]
        else:
            y_start+=200
            img.append(div)
    img = np.concatenate(img, axis=1)
    return img,np.array([xmin,ymin,xmax,ymax]),label

def generate_simple_image(img_size=224,char_size=32):
    L=img_size//char_size
    # 最多有多少文本行 = 一行最多有多少字符
    nb_line=np.random.randint(4,L)
    # 随机出文本行数
    nb_space=img_size-nb_line*char_size
    # 纵向的除了文本行外剩下的空间
    p=np.random.rand(nb_line+1)
    # nb_line个文本行，则一共有nb_line+1个空间需要填充，
    # 我们生成一个概率分布将剩余的空间分配到nb_line+1个位置
    space=p/p.sum()*nb_space
    space=space.astype(np.int32)
    # 多余的放到最后一行下面
    space[-1]+=nb_space-space.sum()
    # 开始生成一张图
    img=[]
    img.append(np.ones((space[0],img_size))*255)
    x_start=space[0]
    position=[]
    Label=[]
    for i in range(1,nb_line+1):
        #添加图片
        line,pos,label=generate_one_line(np.random.randint(4,L+1))
        height=pos[2][0]-pos[0][0]

        pos=(pos+1)*char_size/height-1
        pos=pos.astype(np.int32)
        pos[3] -= 1
        # to let xmin!=xmax

        pos[0]+=x_start
        pos[2]+=x_start

        x_start+=char_size
        x,y=line.shape
        line=cv2.resize(line,(int(char_size*y/x),char_size),interpolation=cv2.INTER_NEAREST)
        res=img_size-line.shape[1]
        w1=np.random.randint(0,res+1)
        pos[1]+=w1
        pos[3]+=w1
        w2=res-w1
        left=np.ones((char_size,w1))*255
        right = np.ones((char_size, w2)) * 255
        img.append(np.concatenate([left,line,right],axis=1))
        #添加间隔块
        img.append(np.ones((space[i], img_size)) * 255)
        x_start+=space[i]
        position.append(pos)
        Label.append(label)
    img=np.concatenate(img)
    position=np.concatenate(position,axis=1)
    Label=[j for i in Label for j in i]
    return img,position,Label

def generate_training_sample(N=10,save_path="synth_data/"):
    """
    创建N张图片存到save_path/data
    创建N个xml文件到save_path/annotation
    创建一个img_name_list.txt到save_path/下
    :param N:创建样本数量
    :return: none
    """
    f=open(save_path+'img_name_list.txt','w')
    for i in range(N):
        f.write(str(i)+'.jpg')
        img,P,L=generate_simple_image(224,2*np.random.randint(6,21))
        #img, P, L = generate_simple_image(224, 2 * np.random.randint(6, 21))
        cv2.imwrite(save_path+"data/"+str(i)+".jpg",img)
        #P[0,:].tolist(),P[1,:].tolist(),P[2,:].tolist(),P[3,:].tolist()
        make_xml(path=save_path+'annotation/',image_name=str(i),xmin_tuple=P[0,:],ymin_tuple=P[1,:],
                 xmax_tuple=P[2,:],ymax_tuple=P[3,:],label_tuple=L)
    f.close()
generate_training_sample(1)