import numpy as np
from math import log, sqrt
from matplotlib import pyplot as plt
import cv2
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element,SubElement,ElementTree
import math
import torch
from torch.optim.optimizer import Optimizer

#CHAR=[str(i) for i in range(10)]+[chr(i) for i in range(65,91)]+[chr(i) for i in range(97,123)]

# CHAR=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
#       'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
#        'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
#         'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

CHAR=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#参考：http://muyaan.com/2018/10/25/%E7%94%9F%E6%88%90%E4%BA%8C%E7%BB%B4%E9%AB%98%E6%96%AF%E5%88%86
# %E5%B8%83%E7%83%AD%E5%8A%9B%E5%9B%BE/
def gaussian(array_like_hm, mean, sigma):
    """
    :param array_like_hm:
    :param mean: 中心点，均值
    :param sigma: 方差
    :return:
    """
    array_like_hm -= mean
    x_term = array_like_hm[:,0] ** 2
    y_term = array_like_hm[:,1] ** 2
    exp_value = - (x_term + y_term) / 2 / pow(sigma, 2)
    #我们采用的标准正太分布，即x方向与y方向方差相同
    return np.exp(exp_value)

def draw_heatmap(width, height, m, dmax,edge_value):
    """
    :param width: 图片宽
    :param height: 图片高
    :param m: 中心点
    :param edge_value 为正态分布在距离中心点dmax处的值
    :return: 热点图
    """
    #####################################
    #这里并不是标准的正太分布，省去了前面的系数，但前面的系数对热点图并无影响
    #####################################
    sigma=sqrt(- pow(dmax, 2) / log(edge_value))
    x = np.arange(width, dtype=np.float)
    y = np.arange(height, dtype=np.float)
    xx, yy = np.meshgrid(x, y)
    # evaluate kernels at grid points
    array_like_hm = np.c_[xx.ravel(), yy.ravel()]
    zz = gaussian(array_like_hm, m, sigma)
    img = zz.reshape((height,width))
    return img

stantard_heatmap=draw_heatmap(99,99,(49,49),49,0.3)
pt = np.float32([[0,0],[0,98],[98,0],[98,98]])

def generate_heatmap(height,width,position,scale=4):
    """
    img: 图片
    position: 形如{'0':[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]],'S':[...,...}的字典前面是标签，
              后面是对应标签出现的坐标，(x1,y1)为左上角，(x2,y2)为右上角，(x3,y3)为左下角，(x4,y4)右下角
    scale: heatmap 从原img放缩的尺寸
    trunc_value: bounding box边界值
    trucntion: True代表除了boungding box区域其他区域值均为0
    return: 36张heatmap，对应0-9，A-Z,a-z大小写字母共62个字符的heatmap，shape: 62*img height*img width
    """
    x,y=height,width
    if x%scale or y%scale:
        print("img of size{:.0f}{:.0f}can't be divided by scale{:.0f}".format(x,y,scale))
        return
    L=[]
    for i in CHAR:
        a = np.zeros(shape=(x,y))
        if i  in position.keys():
            location=position[i]
            for p in location:

                # 生成变换矩阵
                M = cv2.getPerspectiveTransform(pt, np.float32(p))
                # 进行透视变换
                heat_map= cv2.warpPerspective(stantard_heatmap, M, (y, x))
                a+=heat_map
        # plt.imshow(a.transpose())
        # plt.show()
        shrink = cv2.resize(a.transpose(),(y // scale,x // scale),interpolation = cv2.INTER_LANCZOS4)
        #用周围8个像素进行插值
        L.append(shrink)
    return np.array(L)


def prettyXml(element, indent, newline, level = 0): # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if element.text == None or element.text.isspace(): # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
    #else:  # 此处两行如果把注释去掉，Element的text也会另起一行
        #element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element) # 将elemnt转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1): # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        prettyXml(subelement, indent, newline, level = level + 1) # 对子元素进行递归操作

def make_xml(path, image_name, xmin_tuple, ymin_tuple, xmax_tuple, ymax_tuple,label_tuple):
        """
        左上角与右下角
        :param xmin_tuple:
        :param ymin_tuple:
        :param xmax_tuple:
        :param ymax_tuple:
        :param image_name:
        :return:
        """
        root = Element('ID')
        root.text = image_name + '.jpg'
        for i in range(len(xmin_tuple)):
            node = SubElement(root,'Label')
            node_name=SubElement(node, 'label_name')
            node_name.text = label_tuple[i]
            node_xmin = SubElement(node, 'xmin')
            node_xmin.text = str(xmin_tuple[i])
            node_ymin = SubElement(node, 'ymin')
            node_ymin.text = str(ymin_tuple[i])
            node_xmax = SubElement(node, 'xmax')
            node_xmax.text = str(xmax_tuple[i])
            node_ymax = SubElement(node, 'ymax')
            node_ymax.text = str(ymax_tuple[i])
        prettyXml(root,'\t','\n')#加了这个在解析文件名出问题
        tree = ElementTree(root)
        #ET.dump(root)
        tree.write(path+image_name+'.xml', encoding="utf-8", xml_declaration=True)
        #make_xml((1,2),(1,2),(1,2),(1,2),('a','b'),'tt')

def read_xml(file_name,path=""):
    # DOMTree = parse(path+file_name)
    # data=DOMTree.documentElement
    # nodelist=data.getElementsByTagName('Label')
    root=ET.parse(path+file_name).getroot()
    #image_name=root.text
    xmin_tuple,ymin_tuple,xmax_tuple,ymax_tuple=[],[],[],[]
    label_tuple=[]
    for x in root:
        for node in x:
            xmin_tuple.append(int(node.text)) if node.tag=="xmin" else 0
            xmax_tuple.append(int(node.text)) if node.tag=="xmax" else None
            ymin_tuple.append(int(node.text)) if node.tag=="ymin" else None
            ymax_tuple.append(int(node.text)) if node.tag=="ymax" else None
            label_tuple.append(node.text) if node.tag=="label_name" else None
    return [xmin_tuple, ymin_tuple, xmax_tuple, ymax_tuple,label_tuple]#, image_name]
    #read_xml(make_xml((1,2),(1,2),(1,2),(1,2),('a','b'),'tt'))

def tuple_to_position(xmin_tuple, ymin_tuple, xmax_tuple, ymax_tuple, label_tuple):
    """
    transform the result of read_xml to input of generate_heatmap
    :param xmin_tuple:
    :param ymin_tuple:
    :param xmax_tuple:
    :param ymax_tuple:
    :param label_tuple:
    :return:
    """
    position={}
    for i in range(len(xmin_tuple)):
        label=label_tuple[i]
        if label in position.keys():
            position[label].append([[xmin_tuple[i],ymin_tuple[i]],[xmin_tuple[i],ymax_tuple[i]],[xmax_tuple[i],ymin_tuple[i]],[xmax_tuple[i],ymax_tuple[i]]])
        else:
            position[label]=[[[xmin_tuple[i],ymin_tuple[i]],[xmin_tuple[i],ymax_tuple[i]],[xmax_tuple[i],ymin_tuple[i]],[xmax_tuple[i],ymax_tuple[i]]]]
    return position

class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.add_(-step_size,  torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom) )

        return loss


def add_noise():
    pass
# ht=cv2.resize(stantard_heatmap,(6,6))
# print(ht[0][0])
# plt.imshow(stantard_heatmap)
# plt.show()
#test
# position={'0':[[[0,0],[0,20],[20,0],[20,20]]]}
# HM=generate_heatmap(8*18,8*25,position,8)
# print(HM.shape)
# plt.imshow(HM[0])
# print(HM[0][2:8,2:8])
# plt.show()