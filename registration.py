"""
author:majiabo & Durant
time: 2019.1.7
1 此文件由于配准，4x,10x,20x 医学图像。效果较好。
2 可以生成粗配和细配中间记录文件。
3 生成图像加入了图像面积和相关性要求。

"""
import numpy as np
import openslide
import cv2
import os
import glob
from tqdm import tqdm
import scipy.ndimage as ndi
from skimage import  morphology


class Data():
    def __init__(self,ratio,src_path,low_sub_path,high_sub_path,target_size,log = True):
        """
        ratio:匹配数据的对象的倍率 
        使用方法：
        （1）直接从高倍图中采集低倍数据时，使用sample_from_HR
        （2）配准模式
        """
        self.rectify_flag = False

        self.ratio = ratio
        self.src_path = src_path
        self.low_sub_path = low_sub_path
        self.high_sub_path = high_sub_path
        self.target_size = target_size
        self.g_template_size = (7,7)  #高斯模糊的模板大小
        self.g_template_sigma = 1.5   #高斯分布的方差
        self.regis_offset = 100       #配准时的冗余
        self.regis_log_path = './regis_log/'    #配准过程中产生的log 存放目录
        self.regis_log_step = 50      #生成log的间距
        self.regis_ccorr_score = 0.996   #若两幅图的相关性小于该值，则跳过之
        self.regis_match_score = './regis_match_score.txt'

        self.template_size = (1000,1000)  #粗配时的模板大小
        self.match_offset = 2000  #粗配时的偏移量
        self.sub_template_num = 15    #粗配是模板的数量

        self.log_flag = log   #logflag ，控制是否生成log
        self.rough_match_log = './rough_match_log/' #存放粗配中间结果的log

        self.Ex = []        #偏移的期望值
        self.Ey = []        
        self.Sx = []        #偏移的标准差
        self.Sy = []
        self.shift_x = []   #存放所有slide的所有子快的偏移量
        self.shift_y = []   #

        #init some function
        self._check_path_exist() #检查目录状态，若不存在，则创建
        self._get_name_list() # 返回namelist
        
    def _get_name_list(self):
        """
        返回4x,10x,20x的图像路径，顺序对应。
        """
        image_4x_path = self.src_path+'*4x.svs'
        image_4x_list = glob.glob(image_4x_path)
        image_10x_list = list()
        image_20x_list = list()
        for name in image_4x_list:
            name_head = name.split('-')[0]
            image_10x_name = name_head+'-10x.svs'
            image_20x_name = name_head+'-20x.svs'
            image_10x_list.append(image_10x_name)
            image_20x_list.append(image_20x_name)
        if self.ratio == 2.5:
            self.image_low_list = image_4x_list
            self.image_high_list = image_10x_list
        elif self.ratio == 2:
            self.image_low_list = image_10x_list
            self.image_high_list = image_20x_list
        elif self.ratio == 5:
            self.image_low_list = image_4x_list
            self.image_high_list = image_20x_list
        self.image_4x_list = image_4x_list
        self.image_10x_list = image_10x_list
        self.image_20x_list = image_20x_list

    def _extract_image_name(self,image_path):
        full_name = os.path.split(image_path)[1]
        name_no_ext = os.path.splitext(full_name)[0]
        name_id = name_no_ext.split('-')[0]
        return name_id

    def _check_path_exist(self):
        src_flag = os.path.isdir(self.src_path)
        low_sub_path_flag = os.path.isdir(self.low_sub_path)
        high_sub_path_flag = os.path.isdir(self.high_sub_path)
        if not src_flag:
            print("Not found slide,please check path...")
            exit()
        if not low_sub_path_flag:
            os.makedirs(self.low_sub_path)
            print("Not found LR image storage path,create path:",self.low_sub_path)
        if not high_sub_path_flag:
            os.makedirs(self.high_sub_path)
            print("Not found HR image storage path,create path:",self.high_sub_path)

        #检查regis_Log 路径是否存在
        regis_log_flag = os.path.isdir(self.regis_log_path)
        if not regis_log_flag:
            #not debug yet
            os.makedirs(self.regis_log_path)
        #检查粗配log 路径是否存在
        rough_match_log = os.path.isdir(self.rough_match_log)
        if not rough_match_log:
            os.makedirs(self.rough_match_log)
    def _combine_name(self,mode,slide_name,id):
        """
        生成名字，mode 为 low 或 high
        """
        assert mode == 'low' or mode == 'high'
        sub_name = slide_name+'_'+str(id)+'.tif'
        if mode == 'low':
            img_path = os.path.join(self.low_sub_path,sub_name)
        else:
            img_path = os.path.join(self.high_sub_path,sub_name)
        return [img_path,sub_name]

    def _BinarySP(self,img, threColor = 35, threVol= 1000):
        wj1 = img.max(axis=2)
        wj2 = img.min(axis=2)
        wj3 = wj1 - wj2
        imgBin = wj3 > threColor
        imgBin = ndi.binary_fill_holes(imgBin)
        s = np.array([[0,1,0],[1,1,1],[0,1,1]], dtype=np.bool)
        imgBin = ndi.binary_opening(imgBin, s)
        imgCon, numCon = ndi.label(imgBin)
        imgConBig = morphology.remove_small_objects(imgCon, min_size=threVol)
        imgBin = imgConBig > 0
        s = imgBin.sum()
        if s >1:
            flag = True
        else:
            flag = False
        return [flag,imgBin]

    def sample_from_HR(self,num_per_slide,name_record_path):
        """
        target: '10x' or '20x'
        直接从HR中降采样得到低倍图
        """
        assert self.ratio == 2.5 or self.ratio == 2 or self.ratio == 5
        if self.ratio == 2.5:
            name_list = self.image_10x_list
        else:
            name_list = self.image_20x_list
        print("正在处理数据数量:",len(name_list),' slides')
        #暂存name
        img_names_list = []
        for name in name_list:
            print('Processing:',name)
            slide_name = self._extract_image_name(name)

            slide = openslide.OpenSlide(name)
            w,h = slide.dimensions
            # because of PIL bugs,so read two times
            top_slide = np.array(slide.read_region(location = (0,0),level = 0,size = (w,h//2)).convert('RGB'))
            bottom_slide = np.array(slide.read_region(location = (0,h//2),level = 0,size = (w,h//2)).convert('RGB'))

            slide = np.vstack((top_slide,bottom_slide))

            x_range = np.random.randint(self.target_size[0]*2,high = h - self.target_size[0]*2,size = num_per_slide*4)
            y_range = np.random.randint(self.target_size[1]*2,high = w - self.target_size[1]*2,size = num_per_slide*4)
            
            image_count = 0
            for x,y in tqdm(zip(x_range,y_range)):
                sub_hr = slide[x:x+self.target_size[0],y:y+self.target_size[1]].copy()
                if not self._BinarySP(sub_hr):
                    continue
                sub_lr = slide[x-20:x+self.target_size[0]+20,y-20:y+self.target_size[1]+20].copy()
                sub_lr = cv2.GaussianBlur(sub_lr,self.g_template_size,self.g_template_sigma)
                sub_lr = sub_lr[20:-20,20:-20]
                lr_w = int(self.target_size[0]//ratio)
                lr_h = int(self.target_size[1]//ratio)

                sub_lr = cv2.resize(sub_lr,(lr_w,lr_h))
    
                [img_lr_path,sub_name] = self._combine_name('low',slide_name,image_count)
                img_hr_path,_ = self._combine_name('high',slide_name,image_count)
                
                sub_hr = cv2.cvtColor(sub_hr,cv2.COLOR_RGB2BGR)
                sub_lr = cv2.cvtColor(sub_lr,cv2.COLOR_RGB2BGR)

                cv2.imwrite(img_lr_path,sub_lr)
                cv2.imwrite(img_hr_path,sub_hr)

                img_names_list.append(sub_name)
                if image_count >= num_per_slide:
                    break
                image_count += 1
        #写文件名
        with open(name_record_path,'w') as f:
            for line in img_names_list:
                f.write(line)
                f.write('\n')

    def _rough_match(self):
        """
        初步配准，得到一个相对的偏移量，减少后续计算的数量
        num: 取点数量，越多相对越准确
        """
        src_size = (int(self.template_size[0]*self.ratio+self.match_offset*2),int(self.template_size[1]*self.ratio+self.match_offset*2))
        registration_size = tuple([int(x*self.ratio) for x in self.template_size])

        print("正在进行粗配...")
        for low_name,high_name in zip(self.image_low_list,self.image_high_list):
            low_slide = openslide.OpenSlide(low_name)
            high_slide = openslide.OpenSlide(high_name)
            low_slide_w, low_slide_h = low_slide.dimensions
            x_range = np.random.randint(self.template_size[0]*3,high = low_slide_w-self.template_size[0]*3,size = self.sub_template_num)
            y_range = np.random.randint(self.template_size[1]*3,high = low_slide_h-self.template_size[1]*3,size = self.sub_template_num)
            current_shift_x = list()
            current_shift_y = list()
            for low_x,low_y in zip(x_range,y_range):
                #获取HR leftTop coors
                high_x = int(low_x*self.ratio - self.match_offset)
                high_y = int(low_y*self.ratio - self.match_offset)
                #读取 HR and LR region
                high_image_crop = np.array(high_slide.read_region((high_x,high_y),level = 0,size = src_size).convert('RGB'))
                low_image_crop = np.array(low_slide.read_region((low_x,low_y),level = 0,size = self.template_size).convert('RGB'))
                # 将LR放大到与HR同尺度
                lr_to_hr_img = cv2.resize(low_image_crop,registration_size)

                result = cv2.matchTemplate(high_image_crop,lr_to_hr_img,cv2.TM_CCOEFF)
                _, maxval, _, maxloc = cv2.minMaxLoc(result)

                current_shift_x.append(self.match_offset - maxloc[0])
                current_shift_y.append(self.match_offset - maxloc[1])
                regis_crop = high_image_crop[maxloc[1]:maxloc[1]+registration_size[0],maxloc[0]:maxloc[0]+registration_size[1],:]
                if self.log_flag:
                    slide_name = self._extract_image_name(low_name)
                    regis_crop_path = self.rough_match_log+slide_name+'_'+str(low_x)+'_'+str(low_y)+'_HR.tif'
                    lr_to_hr_img_path = self.rough_match_log+slide_name+'_'+str(low_x)+'_'+str(low_y)+'_LR.tif'
                    merge_path = self.rough_match_log+slide_name+'_'+str(low_x)+'_'+str(low_y)+'_Merge.tif'
                    regis_crop = cv2.cvtColor(regis_crop,cv2.COLOR_RGB2BGR)
                    lr_to_hr_img = cv2.cvtColor(lr_to_hr_img,cv2.COLOR_RGB2BGR)
                    merged_img = cv2.addWeighted(regis_crop,0.5,lr_to_hr_img,0.5,0)
                    cv2.imwrite(regis_crop_path,regis_crop)
                    cv2.imwrite(lr_to_hr_img_path,lr_to_hr_img)
                    cv2.imwrite(merge_path,merged_img)
            self.shift_x.append(current_shift_x)
            self.shift_y.append(current_shift_y)

    def _write_analyze_log(self,log_path):
        import csv
        with open(log_path,'w') as f:
            writer = csv.writer(f)
            slide_names = [self._extract_image_name(name) for name in self.image_low_list]
            slide_names.insert(0,'item')
            shift_x = list(self.shift_x)
            shift_x.insert(0,'x_shift')
            shift_y = list(self.shift_y)
            shift_y.insert(0,'y_shift')
            ex = list(self.Ex)
            ex.insert(0,'Ex')
            ey = list(self.Ey)
            ey.insert(0,'Ey')
            sx = list(self.Sx)
            sx.insert(0,'Sx')
            sy = list(self.Sy)
            sy.insert(0,'Sy')
            writer.writerow(slide_names)
            writer.writerow(shift_x)
            writer.writerow(shift_y)
            writer.writerow(ex)
            writer.writerow(ey)
            writer.writerow(sx)
            writer.writerow(sy)

    def _analyze_shift(self,x_thresh = 50,y_thresh = 50):
        for x_shift,y_shift in zip(self.shift_x,self.shift_y):
            temp_ex = np.mean(np.array(x_shift))
            temp_ey = np.mean(np.array(y_shift))
            temp_sx = np.std(np.array(x_shift))
            temp_sy = np.std(np.array(y_shift))
            self.Ex.append(temp_ex)
            self.Ey.append(temp_ey)
            self.Sx.append(temp_sx)
            self.Sy.append(temp_sy)
        if self.log_flag:
            print("正在生成原始数据log...")
            log_path = '_analyze_log_origion.csv'
            self._write_analyze_log(log_path)

        rectify_shift_x = []
        rectify_shift_y = []

        # 排除异常数据，保留相对正常数据
        if self.rectify_flag:
            for x_shift,y_shift,ex,ey in zip(self.shift_x,self.shift_y,self.Ex,self.Ey):
                temp_x_shift_list = []
                temp_y_shift_list = []
                for x_s_item,y_s_item in zip(x_shift,y_shift):
                    if abs(ex - x_s_item) < x_thresh and abs(ey - y_s_item) < y_thresh:
                        temp_x_shift_list.append(x_s_item)
                        temp_y_shift_list.append(y_s_item)
                rectify_shift_x.append(temp_x_shift_list)
                rectify_shift_y.append(temp_y_shift_list)
            # 重新计算
            self.shift_x = rectify_shift_x
            self.shift_y = rectify_shift_y
            self.Ex = []
            self.Ey = []
            self.Sx = []
            self.Sy = []
            for x_shift,y_shift in zip(self.shift_x,self.shift_y):
                temp_ex = np.mean(np.array(x_shift))
                temp_ey = np.mean(np.array(y_shift))
                temp_sx = np.std(np.array(x_shift))
                temp_sy = np.std(np.array(y_shift))
                self.Ex.append(temp_ex)
                self.Ey.append(temp_ey)
                self.Sx.append(temp_sx)
                self.Sy.append(temp_sy)
            #记录新的log
            if self.log_flag:
                print("正在生成修正后的数据log...")
                log_path = '_analyze_log_rectify.csv'
                self._write_analyze_log(log_path)

    def _map_coors(self,x,y,ex,ey,size):
        """
        进行坐标映射，x,y为低倍做个
        """
        h_x = int(x*self.ratio - ex) - self.regis_offset
        h_y = int(y*self.ratio - ey) - self.regis_offset
        h_w = size[0]+2*self.regis_offset
        h_h = size[1]+2*self.regis_offset
        return h_x,h_y,h_w,h_h

    def registration(self,names_record_path,num_per_slide = 10000):
        """
        被多次修改，效率低
        """
        print("正在生成数据中...")
        #粗配
        self._rough_match()
        #分析,清理错误数据
        self._analyze_shift(x_thresh=50,y_thresh=50)
    
        pair_match_ccorr = {} # 记录相关系数
        img_names_list = []  #记录图像名
        low_size = [int(x/self.ratio) for x in self.target_size]
        regis_low_size = [200,200]
        regis_high_size = tuple([int(x*self.ratio) for x in regis_low_size])
        for low_src_name,high_src_name,ex,ey in zip(self.image_low_list,self.image_high_list,self.Ex,self.Ey):
            print("正在处理:{}".format(low_src_name))
            low_slide = openslide.OpenSlide(low_src_name)
            high_slide = openslide.OpenSlide(high_src_name)
            
            w,h = low_slide.dimensions

            x_range = np.random.randint(self.target_size[0]*2,high = h - self.target_size[0]*2,size = num_per_slide*4)
            y_range = np.random.randint(self.target_size[1]*2,high = w - self.target_size[1]*2,size = num_per_slide*4)
            count_id = 0
            slide_name = self._extract_image_name(low_src_name)
            for x,y in zip(x_range,y_range):
                low_sub = np.array(low_slide.read_region((x,y),level = 0,size = regis_low_size).convert('RGB'))
                flag,imgBin = self._BinarySP(low_sub,threColor=30,threVol=3500)
                if not flag:
                    continue
                lr_to_hr = cv2.resize(low_sub,regis_high_size)
                h_x,h_y,h_w,h_h = self._map_coors(x,y,ex,ey,regis_high_size)
                high_sub = np.array(high_slide.read_region(location = (h_x,h_y),level = 0,size = (h_w,h_h)).convert('RGB'))
                # 开始局部配准
                result = cv2.matchTemplate(high_sub,lr_to_hr,cv2.TM_CCOEFF)
                _, maxval, _, maxloc = cv2.minMaxLoc(result)
                pair_hr = high_sub[maxloc[1]:(maxloc[1]+regis_high_size[0]),maxloc[0]:(maxloc[0]+regis_high_size[1]),:].copy()
                
                score = cv2.matchTemplate(pair_hr,lr_to_hr,cv2.TM_CCORR_NORMED)
                score = float(score)
                lr_img_path,img_name = self._combine_name('low',slide_name,count_id)
                hr_img_path,_ = self._combine_name('high',slide_name,count_id)
                if self.log_flag:
                    if count_id % self.regis_log_step == 0:
                        temp_low_sub = cv2.cvtColor(lr_to_hr,cv2.COLOR_RGB2BGR)
                        temp_high_sub = cv2.cvtColor(pair_hr,cv2.COLOR_RGB2BGR)
                        temp_merge_sub = cv2.addWeighted(temp_low_sub,0.5,temp_high_sub,0.5,0)
                        img_temp_name = img_name.split('.')[0]
                        cv2.imwrite(os.path.join(self.regis_log_path,img_temp_name+'_low.tif'),temp_low_sub)
                        cv2.imwrite(os.path.join(self.regis_log_path,img_temp_name+'_high.tif'),temp_high_sub)
                        cv2.imwrite(os.path.join(self.regis_log_path,img_temp_name+'_merge.tif'),temp_merge_sub)
                # 判断阈值是否符合要求
                if score < self.regis_ccorr_score:
                    print("!!!图像相关性小于阈值...")
                    continue
                else:
                    #截图
                    temp_low = cv2.cvtColor(low_sub,cv2.COLOR_RGB2BGR)
                    temp_high = cv2.cvtColor(pair_hr,cv2.COLOR_RGB2BGR)
                    temp_low = temp_low[:low_size[0],:low_size[1],:].copy()
                    temp_high = temp_high[:self.target_size[0],:self.target_size[1],:].copy()
                    flag , _ =  self._BinarySP(temp_high,threVol=3500)
                    if flag:
                        cv2.imwrite(lr_img_path,temp_low)
                        cv2.imwrite(hr_img_path,temp_high)
                        img_names_list.append(img_name)
                        pair_match_ccorr[img_name] = score
                        count_id += 1
                if count_id >= num_per_slide:
                    break

        #记录文件名
        with open(names_record_path,'w') as f:
            for line in img_names_list:
                f.write(line)
                f.write('\n')
        with open(self.regis_match_score,'w') as f:
            keys = pair_match_ccorr.keys()
            for key in keys:
                f.write(key)
                f.write(':')
                f.write(str(pair_match_ccorr[key]))
                f.write('\n')












            


