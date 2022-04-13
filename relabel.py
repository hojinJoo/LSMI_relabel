from argparse import ZERO_OR_MORE
from audioop import mul
import os, shutil, json, cv2, rawpy, math, torch,sys
from tkinter.messagebox import NO
import numpy as np
from tqdm import tqdm
from make_mixture_map import apply_wb_raw
import torch.nn.functional as F
import statistics

DATAPATH = '/media/NAS3/CIPLAB/dataset/LSMI/galaxy/'
SAVEPATH = './results'
META = './meta.json'
CAMERA = "galaxy"
RAW = CAMERA+".dng"
VISUALIZE = True
ZERO_MASK = -1       # masking value for unresolved pixel where G = 0 in all image pairs
ONLY_LIGHT2 = -2
SAVE_SUBTRACTED_IMG = False
RAW_EXT = os.path.splitext(RAW)[1]
TEMPLETE = rawpy.imread(RAW)
NO_GT = -3


if CAMERA == 'sony':
    BLACK_LEVEL = 128
    BLACK_LEVEL_RAW = 512
else:
    BLACK_LEVEL = min(TEMPLETE.black_level_per_channel)
    BLACK_LEVEL_RAW = BLACK_LEVEL


SATURATION = TEMPLETE.white_level
RAW_PATTERN = TEMPLETE.raw_pattern.astype('int8')


def get_AE(pred,gt,camera='galaxy') :
    pred = pred.squeeze()
    gt = torch.from_numpy(gt)
    
    if camera == 'galaxy':
        pred = torch.clamp(pred, 0, 1023)
        gt = torch.clamp(gt, 0, 1023)
    elif camera == 'sony' or camera == 'nikon':
        pred = torch.clamp(pred, 0, 16383)
        gt = torch.clamp(gt, 0, 16383)

    # print(pred.size())
    # print(gt.size())
    cos_similarity = F.cosine_similarity(pred+1e-4,gt+1e-4,dim=2)
    cos_similarity = torch.clamp(cos_similarity, -1, 1)
    rad = torch.acos(cos_similarity)
    ang_error = torch.rad2deg(rad)
    
    return ang_error
    
def get_MAE(pred,gt,mask,camera='galaxy'):
    """
    pred : w,h,c
    gt : w,h,c
    """
    
    ae = get_AE(pred,gt)
    mask = np.logical_not(mask)
    ae = ae[mask]
    mean_angular_error = ae.mean().item()

    return mean_angular_error


def get_PSNR(pred, gt,camera='galaxy'):
    """
    pred & gt   : (w,h,c) numpy array 3 channel RGB
    returns     : average PSNR of two images
    """
    white_level = SATURATION
    
    
    pred = torch.from_numpy(pred)
    gt = torch.from_numpy(gt)
    
    # 여기서는 0 clamp
    pred = torch.clamp(pred,0,white_level)
    gt = torch.clamp(gt,0,white_level)
    
    mse = torch.mean((pred-gt)**2)
    psnr = 20 * torch.log10(white_level / torch.sqrt(mse))

    # pred_np = pred.cpu().numpy()
    # gt_np = gt.cpu().numpy()

    # psnr_cv = cv2.PSNR(pred_np,gt_np,white_level)
    psnr = psnr.item()
    
    return psnr


def get_coefficent(mixed_illumination_map,mode ='best',*lights) :
    """
    mixed_illum     : mixed illumnintaion map img : place_12 / place_1_wb_recal
    *light          : Lights info from meta.json
    mode            : return mode
    """
    assert mode in ['R','B','best']
    coefficients = {}
    err = {}
    lights_dict = {}
    
    if len(lights) == 2 :
        light1,light2 = lights
        light1 = light1[None,None,:]
        light2 = light2[None,None,:]

        lights_dict['light_1'] = light1
        lights_dict['light_2'] = light2
        
        no_gt = np.where(mixed_illumination_map[:,:,1] == NO_GT)
        
        both_zero = np.where(mixed_illumination_map[:,:,1] == ZERO_MASK)

        for l in [1,2] :
            light_with_a = lights_dict[f'light_{l}']
            light_other = lights_dict[f'light_{3-l}']
            for c_idx, c in enumerate(['R','B']) :
                if c_idx == 1 : c_idx=2
                denominator = light_with_a[:,:,c_idx] - light_other[:,:,c_idx]
                
                dividend = mixed_illumination_map[:,:,c_idx] - light_other[:,:,c_idx] 

                # 사실상 L1과 L2가 같을 수가 없음
                coefficient_a = dividend / denominator
                coefficient_other = 1 - coefficient_a
                
                #coefficient_a = np.where(denominator==0, ZERO_MASK,coefficient_a)
                coefficient_a[both_zero] = ZERO_MASK
                coefficient_a[no_gt] = NO_GT

                coefficient_other[both_zero] = ZERO_MASK
                coefficient_other[no_gt] = NO_GT
 
                
                if l == 1 :
                    coefficient_map = np.stack((coefficient_a,coefficient_other),axis=-1)
                else : 
                    coefficient_map = np.stack((coefficient_other,coefficient_a),axis=-1)

                coefficients[f'light_{l}_channel_{c}'] = coefficient_map
                # if l==2 and c =='B' :
                #     print(np.min(coefficient_a),np.max(coefficient_a))
                #     M = np.where(coefficient_a == np.max(coefficient_a))
                #     m = np.where(coefficient_a == np.min(coefficient_a))
                #     z = np.zeros_like(mixed_illumination_map,np.int8)
                   
                #     z = cv2.rectangle(z,(m[0][0].item()-25,m[1][0].item()-25), (m[0][0].item()+25,m[1][0].item()+25),(255,255,255),5)
                #     z = cv2.rectangle(z,(M[0][0].item()-25,M[1][0].item()-25), (M[0][0].item()+25,M[1][0].item()+25),(255,0,0,),5)
                #     cv2.imwrite("asd.png",z)
                #     #print(m)
                    
                #     print(denominator)
                #     print(f"m DIVIDEND : {dividend[m[0],m[1]]}")
                #     print("COEFF",coefficient_a[m[0],m[1]])
                #     print("m MIXED_MAP",mixed_illumination_map[m[0],m[1],:])
                #     print(f"M DIVIDEND : {dividend[M[0],M[1]]}")
                #     print("M MIXED_MAP",mixed_illumination_map[M[0],M[1],:])
                #     exit()
                    
                    

        if mode == 'best' :
            # find least error
            for l in range(1,3) :
                for c in ['R','B'] :
                    err[f"light_{l}_channel_{c}"] = np.sum(np.abs( \
                    coefficients[f"light_{l}_channel_{c}"][2-l] - coefficients[f"light_{3-l}_channel_{c}"][2-l]))
            
            # min_error_key = min(err,key=err.get)
            # return coefficients[min_error_key]

            return coefficients
        #     # print(err)
        #     # exit()
        #     min_error_key = min(err,key=err.get)
        #     # print(min_error_key)

        #     # all = coefficients[min_error_key][:,:,0].flatten().shape[0]
        #     # #print(coefficients[min_error_key][:,:,0].flatten().shape)
        #     # not_in_range = np.logical_or(coefficients[min_error_key][:,:,0]>1,coefficients[min_error_key][:,:,0]<0)
        #     # part = coefficients[min_error_key][not_in_range][:,0].shape[0]
        #     # print(100 * part/all)
        #     # exit()
        #     # print(np.min(coefficients[min_error_key][0]),np.max(coefficients[min_error_key][0]),np.min(coefficients[min_error_key][1]),np.max(coefficients[min_error_key][1]))
        #     # print(np.sum(coefficients[min_error_key][0]>0.5))
        #     # exit()
        #     return coefficients[min_error_key]
        # else : 
        #     return coefficients[f"light_1_channel_{mode}"]

        
    
    # 3 Lights
    else :
        light1,light2,light3 = lights
        light1 = light1[None,None,:]
        light2 = light2[None,None,:]
        light3 = light3[None,None,:]

        lights_dict['light_1'] = light1
        lights_dict['light_2'] = light2
        lights_dict['light_3'] = light3
        
        
        light_with_a = light1
        light_with_b = light2
        light_other = light3
    
        
        term_A =  (light_with_b[:,:,2] - light_other[:,:,2])\
                    / np.clip((light_with_b[:,:,0] - light_other[:,:,0]),np.finfo(float).tiny,np.finfo(float).max)

        denominator_a = light_with_a[:,:,2] - light_other[:,:,2] + term_A * (light_with_a[:,:,0] - light_other[:,:,0])
        coefficient_a = (mixed_illumination_map[:,:,2] - light_other[:,:,2] - term_A * (mixed_illumination_map[:,:,0] - light_other[:,:,0]) ) \
                        / np.clip(denominator_a, np.finfo(float).tiny, np.finfo(float).max)
        
        coefficient_a = np.where(denominator_a==0, ZERO_MASK,coefficient_a)

        denominator_b = light_with_b[:,:,0] - light_other[:,:,0]
        coefficient_b = ( mixed_illumination_map[:,:,0] - light_other[:,:,0] - coefficient_a * ( light_with_a[:,:,0] -light_other[:,:,0]) ) \
                        / np.clip(denominator_b, np.finfo(float).tiny, np.finfo(float).max)
        
        coefficient_b = np.where(denominator_b==0, ZERO_MASK,coefficient_b)

        coefficient_other = np.where(np.logical_and(coefficient_a == ZERO_MASK, coefficient_b==ZERO_MASK),ZERO_MASK,1.0 - coefficient_a - coefficient_b)
        
        coefficient_map = np.stack((coefficient_a,coefficient_b,coefficient_other),axis=-1)
        
        return coefficient_map
    
def get_illumination_map_3_v2(imgs,single_imgs,Lights) :
    img_12, img_13,img = imgs
    img_1,img_2,img_3 = single_imgs
    L1,L2,L3 = Lights

    ratio_alpha_12_denom =  img_1[...,1] + img_2[...,1] + 1e-6
    ratio_alpha_12 = img_1[...,1]  / ratio_alpha_12_denom
    ratio_alpha_12 = (1 - ratio_alpha_12) / (ratio_alpha_12 + 1e-6)

    ratio_alpha_13_denom =  img_1[...,1] + img_3[...,1] + 1e-6
    ratio_alpha_13 = img_1[...,1]  / ratio_alpha_13_denom
    ratio_alpha_13 = (1 - ratio_alpha_13) / (ratio_alpha_13 + 1e-6)

    
    # OLD ~
    # coeff_denominator =  img_1[...,1] + img_2[...,1] + img_3[...,1] + 1e-6
    # alpha = img_1[...,1] / coeff_denominator
    # alpha = alpha[...,None]
    # beta = img_2[...,1] / coeff_denominator
    # beta = beta[...,None]

    # reflectance_gt = alpha * img_1 / L1 + beta * img_2 / L2 + (1 - alpha - beta) * img_3 / L3
    # ~ OLD
    
    # NEW ~
    coeff_denominator = 1 + ratio_alpha_12 + ratio_alpha_13
    alpha = 1 / coeff_denominator
    alpha = alpha[...,None]
    beta = ratio_alpha_12 / coeff_denominator
    beta = beta[...,None]
    reflectance_gt = 1 * img_1 / L1 + beta * img_2 / L2 + (1 - alpha - beta) * img_3 / L3
    # ~ NEW


    reflectance_zero = (reflectance_gt[...,1] == 0)

    mul_factor = img[...,1] / (reflectance_gt[...,1]  + 1e-6) 

    mul_factor[reflectance_zero] = 1
    mul_factor = mul_factor[...,None]

    recal = mul_factor * reflectance_gt

    mixed_map = img / (recal + 1e-6)

    wb_red_zero = np.where(np.logical_and(np.logical_and(recal[:,:,0]<=1e-2,recal[:,:,1]!=0),img[...,1] !=0))
    wb_blue_zero = np.where(np.logical_and(np.logical_and(recal[:,:,2]<=1e-2,recal[:,:,1]!=0),img[...,1] !=0))

    mixed_map[wb_red_zero[0],wb_red_zero[1],0] = img[wb_red_zero[0],wb_red_zero[1],0] / img[wb_red_zero[0],wb_red_zero[1],1]
    mixed_map[wb_blue_zero[0],wb_blue_zero[1],2] = img[wb_blue_zero[0],wb_blue_zero[1],2] / img[wb_blue_zero[0],wb_blue_zero[1],1]

    both_zero = np.logical_and(recal[...,1]==0, img[...,1]==0)
    no_gt = np.logical_or(np.logical_and(recal[...,1]!=0,img[...,1]==0),np.logical_and(recal[...,1] ==0, img[...,1]!=0))
    mixed_map[both_zero] = ZERO_MASK
    mixed_map[no_gt] = NO_GT

    return mixed_map

def get_illumination_map_3(img,imgs,Lights) :
    img_1,img_2,img_3 = imgs
    L1,L2,L3 = Lights
    coeff_denominator =  img_1[...,1] + img_2[...,1] + img_3[...,1] + 1e-6
    alpha = img_1[...,1] / coeff_denominator
    alpha = alpha[...,None]
    beta = img_2[...,1] / coeff_denominator
    beta = beta[...,None]

    reflectance_gt = alpha * img_1 / L1 + beta * img_2 / L2 + (1 - alpha - beta) * img_3 / L3

    reflectance_zero = (reflectance_gt[...,1] == 0)

    mul_factor = img[...,1] / (reflectance_gt[...,1]  + 1e-6) 

    mul_factor[reflectance_zero] = 1
    mul_factor = mul_factor[...,None]

    recal = mul_factor * reflectance_gt

    mixed_map = img / (recal + 1e-6)

    wb_red_zero = np.where(np.logical_and(np.logical_and(recal[:,:,0]<=1e-2,recal[:,:,1]!=0),img[...,1] !=0))
    wb_blue_zero = np.where(np.logical_and(np.logical_and(recal[:,:,2]<=1e-2,recal[:,:,1]!=0),img[...,1] !=0))

    mixed_map[wb_red_zero[0],wb_red_zero[1],0] = img[wb_red_zero[0],wb_red_zero[1],0] / img[wb_red_zero[0],wb_red_zero[1],1]
    mixed_map[wb_blue_zero[0],wb_blue_zero[1],2] = img[wb_blue_zero[0],wb_blue_zero[1],2] / img[wb_blue_zero[0],wb_blue_zero[1],1]

    both_zero = np.logical_and(recal[...,1]==0, img[...,1]==0)
    no_gt = np.logical_or(np.logical_and(recal[...,1]!=0,img[...,1]==0),np.logical_and(recal[...,1] ==0, img[...,1]!=0))
    mixed_map[both_zero] = ZERO_MASK
    mixed_map[no_gt] = NO_GT

    return mixed_map

def get_illumination_map(wb, img,img_1,img_2,L1,L2) :
    """
    wb      : white balanced img_1
    img     : img_12
    """
    
    # denominator = wb[:,:,1] + 1e-6

    # mul_factor = img[:,:,1] / denominator
    # mul_factor = mul_factor[:,:,None]

    # recal = np.multiply(wb,mul_factor)


    # # NOTE : recal [0,10,4] , img[8,10,8] -> mixed_map[0.8,1,1.5], G로 Normalize함
    # wb_red_zero = np.where(np.logical_and(recal[:,:,0]==0.0,recal[:,:,1]!=0))
    # wb_blue_zero = np.where(np.logical_and(recal[:,:,2]==0.0,recal[:,:,1]!=0))


    # mixed_map = img / (recal+1e-6)

    # mixed_map[wb_red_zero[0],wb_red_zero[1],0] = img[wb_red_zero[0],wb_red_zero[1],0] / img[wb_red_zero[0],wb_red_zero[1],1]
    # mixed_map[wb_blue_zero[0],wb_blue_zero[1],2] = img[wb_blue_zero[0],wb_blue_zero[1],2] / img[wb_blue_zero[0],wb_blue_zero[1],1]
    
    # light2_cond = np.logical_and(img[:,:,1] !=0, wb[:,:,1] ==0)
    
    # # 불을 켰는데 더 어두워진 경우
    # no_gt_cond = np.logical_and(img[:,:,1] == 0, wb[:,:,1] !=0)
    # both_zero = np.logical_and(img[:,:,1]==0, wb[:,:,1]==0)

    # mixed_map[both_zero] = ZERO_MASK  
    # mixed_map[light2_cond] = ONLY_LIGHT2
    # mixed_map[no_gt_cond] = NO_GT
    
    # 새 방법
    img_1 = img_1.astype("int16")
    
    alpha_denominator = img_1[...,1] + img_2[...,1] + 1e-6
    alpha = img_1[...,1] / alpha_denominator
    alpha = alpha[...,None]
    
    
    reflectance_gt = alpha * img_1 / L1 + (1-alpha) * img_2 / L2

    reflectance_zero = (reflectance_gt[:,:,1]==0)

    mul_factor = img[:,:,1] / (reflectance_gt[:,:,1] + 1e-6)

    mul_factor[reflectance_zero] = 1
    mul_factor = mul_factor[...,None]

    recal = mul_factor * reflectance_gt


    mixed_map = img / (recal + 1e-6)

    wb_red_zero = np.where(np.logical_and(np.logical_and(recal[:,:,0]<=1e-2,recal[:,:,1]!=0),img[...,1] !=0))
    wb_blue_zero = np.where(np.logical_and(np.logical_and(recal[:,:,2]<=1e-2,recal[:,:,1]!=0),img[...,1] !=0))

    mixed_map[wb_red_zero[0],wb_red_zero[1],0] = img[wb_red_zero[0],wb_red_zero[1],0] / img[wb_red_zero[0],wb_red_zero[1],1]
    mixed_map[wb_blue_zero[0],wb_blue_zero[1],2] = img[wb_blue_zero[0],wb_blue_zero[1],2] / img[wb_blue_zero[0],wb_blue_zero[1],1]

    # light2까지 고려했기 때문에 이전처럼 light2cond 는 no_gt
    both_zero = np.logical_and(recal[...,1]==0, img[...,1]==0)
    no_gt = np.logical_or(np.logical_and(recal[...,1]!=0,img[...,1]==0),np.logical_and(recal[...,1] ==0, img[...,1]!=0))
    mixed_map[both_zero] = ZERO_MASK
    mixed_map[no_gt] = NO_GT
    
    
    # print((alpha*(img_1/L1))[0,0])
    # print(((1-alpha) * (img_2/L2))[0,0])
    # print(alpha[0,0])
    # print(img_1[0,0])
    # print(img_2[0,0])
    # print(img[0,0])
    # print(mixed_map[0,0])
    # print(recal[0,0])
    # print(np.min(mixed_map),np.max(mixed_map))
    # exit()
    # img_mask = np.zeros_like(img,np.uint8)
    # r = np.logical_or(mixed_map[:,:,0] > L1[0], mixed_map[:,:,0] < L2[0])
    # b = np.logical_or(mixed_map[:,:,2] > L1[2], mixed_map[:,:,2] < L2[2])
    # rb = np.logical_and(r,b)
    
    
    # img_mask[r] = [255,255,255]
    # cv2.imwrite('img_.png',img_mask)
    # exit()

    # M = np.where(mixed_map == np.min(mixed_map))
    # print("WB_1",wb[M[0],M[1],:])
    # print("IMG_12",img[M[0],M[1],:])
    # print("IMG_1",img_1[M[0],M[1],:])
    # print("IMG_2",img_2[M[0],M[1],:])
    # print(f"Alpah {alpha[M[0],M[1],:]}")
    # print(f"recal {recal[M[0],M[1],:]}")
    # print("MIXED_MAP",mixed_map[M[0],M[1],:])

    # m = np.where(mixed_map == np.max(mixed_map))
    # print(np.max(mixed_map))
    # z = np.zeros_like(img,np.int8)
    # z[m[0],m[1]] = [255,255,255]
    # z = cv2.rectangle(z,(m[0].item()-25,m[1].item()-25), (m[0].item()+25,m[1].item()+25),(255,255,255),5)
    # cv2.imwrite("asd.png",z)
    # print(m)
    # print("WB_1",wb[m[0],m[1],:])
    # print("IMG_12",img[m[0],m[1],:])
    # print("IMG_1",img_1[m[0],m[1],:])
    # print("IMG_2",img_2[m[0],m[1],:])
    # print(f"Alpah {alpha[m[0],m[1],:]}")
    # print(f"recal {recal[m[0],m[1],:]}")
    # print("MIXED_MAP",mixed_map[m[0],m[1],:])
    # exit()

    return mixed_map

def get_pixelwise_illumination(coefficients,mixed_illumination_map,mask,L1,L2) :
    illum_t = torch.tensor([])
    ae_t = torch.tensor([])
    coeff_t = torch.tensor([])
    for l in [1,2] :
        for c in ['R','B'] :
            coefficient_map_tmp = coefficients[f"light_{l}_channel_{c}"]
            
            
            l1_coeff = coefficient_map_tmp[:,:,0,None]
            l2_coeff = coefficient_map_tmp[:,:,1,None]


            multi_illumination = l1_coeff * L1 + l2_coeff * L2
            multi_illumination = multi_illumination[None,...]
            multi_illumination = torch.from_numpy(multi_illumination)
            
            illum_t = torch.cat((illum_t,multi_illumination),dim=0)

            ae = get_AE(multi_illumination,mixed_illumination_map)
            ae = ae[None,...]
            
            ae_t = torch.cat((ae_t,ae),dim=0)
            
            coefficient_map_tmp = torch.from_numpy(coefficient_map_tmp)
            coefficient_map_tmp = coefficient_map_tmp[None,...]
            coeff_t = torch.cat((coeff_t,coefficient_map_tmp),dim=0)


    
    min_idx = torch.min(ae_t,dim=0).indices
    idx_c = torch.zeros(3).unsqueeze(0)
    idx_coeff = torch.zeros(2).unsqueeze(0)
    coeff_idx = min_idx[...,None] + idx_coeff
    coeff_idx = coeff_idx[None,...]
    coeff_idx = coeff_idx.type(torch.int64)
    min_idx = min_idx[...,None] + idx_c
    min_idx = min_idx[None,...]
    min_idx = min_idx.type(torch.int64)
    
    
    
    # print(min_idx[0,0,0])
    # print(ae_t[:,0,0])
    # print(ae_t.size())
    
    least_coeff = torch.gather(coeff_t,0,coeff_idx).squeeze()

    least_ae_illumination = torch.gather(illum_t,0,min_idx).squeeze()
    
    # print(illum_t[:,0,0])
    # print(least_ae_illumination[0,0])
    # input(mixed_illumination_map[0,0])
    
    # print(illum_t[:,0,0,:])
    # print(least_ae_illumination[0,0])
    # print(least_ae_illumination.size())
    # exit()

    return least_ae_illumination, least_coeff
    


def dict_save(d) :
    ret = {}
    #keys = ["MAE_COEFFICIETNS","MAE_MIXED_ILLUMUNATION_MAP","PSNR_COEFFICIENTS","PSNR_MIXED_ILLUMINATION_MAP"]
    keys = ["MAE_COEFFICIENTS"]
    for k in keys :
        ret[k] = {}
        tmp = ret[k]
        tmp["AVERAGE"] = statistics.mean((place[k]) for place in d.values())
        tmp_min  = min((place[k]) for place in d.values())
        #print(tmp_min)
        tmp_max = max((place[k]) for place in d.values())
        min_place = [place for place in d.keys() if d[place][k] == tmp_min]
        max_place = [place for place in d.keys() if d[place][k] == tmp_max]
        tmp["MINIMUM"] = [tmp_min,min_place]
        tmp["MAXIMUM"] = [tmp_max,max_place]

    return ret

def main() :
    with open(META, 'r') as json_file:
        jsonData = json.load(json_file)
    
    # with open("./place_dict.json",'r') as r :
    #     d= json.load(r)
    
    M = "MAE_MIXED_ILLUMUNATION_MAP"

    places = sorted([f for f in os.listdir(DATAPATH) if os.path.isdir(os.path.join(DATAPATH,f))])
    place_dict = {}
    coefficient_dict = {}
    
    for place in tqdm(places):
    #     # place = 'Place544'    

        try : 
            placeInfo = jsonData[place]
            
            if placeInfo["NumOfLights"] == 2:
                continue

                Light1 = np.array(placeInfo["Light1"])
                Light2 = np.array(placeInfo["Light2"])
                
                single_image = place + '_1'
                multi_image = place + '_12'
    
                img_1 = cv2.imread(DATAPATH + place +'/'+ single_image + ".tiff", cv2.IMREAD_UNCHANGED) - BLACK_LEVEL
                img_12 = cv2.imread(DATAPATH + place +'/' + multi_image + ".tiff", cv2.IMREAD_UNCHANGED) - BLACK_LEVEL        

                img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB)
                img_12 = cv2.cvtColor(img_12,cv2.COLOR_BGR2RGB)

                img_2 = np.clip(img_12.astype("int16") - img_1.astype("int16"),0,SATURATION-BLACK_LEVEL)           
                

                img_2_wb = (img_2/Light2)
                img_1_wb = (img_1 / Light1)
                
                # 지금 여러 케이스 나눠가지고 masking하는 게 3개나 있는데 
                # 나중에 쓰는 거라면 masking 다시 하나로 통일 ??

                mixed_illumination_map = get_illumination_map(img_1_wb, img_12, img_1,img_2,Light1,Light2)
                fname_illumination = SAVEPATH +'/relabelled'+'/'+multi_image + "_mixedillum.npy"
                np.save(fname_illumination,mixed_illumination_map)
                # print(mixed_illumination_map)
                
                both_zero = (mixed_illumination_map[:,:,1] == ZERO_MASK)
                no_gt = (mixed_illumination_map[:,:,1] == NO_GT)
                mask = np.logical_or(both_zero, no_gt)
                # print(np.sum(mask))

                # coefficient_map = get_coefficent(mixed_illumination_map,'best',Light1,Light2)
                coefficients = get_coefficent(mixed_illumination_map,'best',Light1,Light2)


                fname_multi_illum = SAVEPATH+'/relabelled'+ '/' + multi_image + "_coefficient_ratio.npy"
                multi_illumination,coefficient_map = get_pixelwise_illumination(coefficients,mixed_illumination_map,mask,Light1,Light2)
                # illum_zero = (multi_illumination == 0.0 )
                # img_12_wb = img_12 / ( multi_illumination + 1e-6 )
                # img_12_wb[illum_zero] = 0.0 # or g normalize
                mae_12 = get_MAE(multi_illumination,mixed_illumination_map,mask)
                coefficient_map = coefficient_map.numpy()
                multi_illumination = multi_illumination.numpy()
                np.save(fname_multi_illum,coefficient_map)
                

                # np.save('./light.npy',multi_illumination)
                # exit()
                # for l in range(1,3) :
                #     for c in ['R','B'] :
                #         coefficient_map_tmp= coefficients[f"light_{l}_channel_{c}"]
                        
                #         l1_coeff = coefficient_map_tmp[:,:,0,None]
                #         l2_coeff = coefficient_map_tmp[:,:,1,None]
                        

                #         multi_illumination = l1_coeff * Light1 + l2_coeff * Light2
                #         illum_zero = (multi_illumination == 0.0 )
                #         img_12_wb_tmp = img_12 / ( multi_illumination + 1e-6 )
                #         img_12_wb_tmp[illum_zero] = 0.0 # or g normalize
                        
                #         mae_12_tmp = get_MAE(multi_illumination,mixed_illumination_map,mask)
                #         print(place,mae_12_tmp)
                #         if mae_12_tmp < mae_12 :
                #             mae_12 = mae_12_tmp
                #             img_12_wb = img_12_wb_tmp
                #             coefficient_map = coefficient_map_tmp
                #             print(f"{place} : final MAE : {mae_12}")
                

                


                # original
                # l1_coeff = coefficient_map[:,:,0,None]
                # l2_coeff = coefficient_map[:,:,1,None]

                # multi_illumination = l1_coeff * Light1 + l2_coeff * Light2
                # 여기까지

  

                # # OPT1
                # multi_illumination = np.where(multi_illumination==0.0, 1, multi_illumination)
                                
                # img_12_wb = img_12 / multi_illumination
                
                # OPT2
                
                illum_zero = (multi_illumination == 0.0 )
                img_12_wb = img_12 / ( multi_illumination + 1e-6 )
                img_12_wb[illum_zero] = 0.0 # or g normalize
                
                img_12_mixed_denominator = np.where(mixed_illumination_map <= 0.0, 1, mixed_illumination_map)
                img_12_mixed_wb = img_12 / img_12_mixed_denominator
                
                # ZERO MASK + NO_GT => MASK
                place_dict[place] = {}
                p = place_dict[place]

                # mae_12 = get_MAE(img_12_wb,img_1_wb,mask)
#                psnr_12 = get_PSNR(img_12_wb,img_1_wb)

              #  mae_mixed = get_MAE(img_12_mixed_wb,img_1_wb,mask)
          #      psnr_mixed = get_PSNR(img_12_mixed_wb,img_1_wb)
                all = coefficient_map[:,:,0].flatten().shape[0]
                not_in_range_1 = np.logical_or(coefficient_map[:,:,0]>1,coefficient_map[:,:,0]<0)
                part_1 = coefficient_map[not_in_range_1][:,0].shape[0]
                not_in_range_2 = np.logical_or(coefficient_map[:,:,1]>1,coefficient_map[:,:,1]<0)
                part_2 = coefficient_map[not_in_range_2][:,1].shape[0]


                p["COEFFICIENT_RATIO"] = {'Coefficient_1' : 100 *part_1/all,'Coefficient_2' : 100*part_2/all}
                p["MAE_COEFFICIENTS"] = mae_12
            #    p["MAE_MIXED_ILLUMUNATION_MAP"] = mae_mixed
         #       p["PSNR_COEFFICIENTS"] = psnr_12
        #        p["PSNR_MIXED_ILLUMINATION_MAP"] = psnr_mixed
              
                print(mae_12)
                with open(os.path.join(SAVEPATH,"relabelled",f'result_{place}.json'),'w') as out_file :
                    json.dump(p,out_file,indent=4)


                if VISUALIZE :
                    # print(type(img_1_wb))
                    # print(type(img_12_wb))
                    # print(type(img_12_mixed_wb))
                    # input()
                    img_1_wb = img_1_wb[...,::-1]
                    img_12_wb = img_12_wb[...,::-1]
                    img_12_mixed_wb = img_12_mixed_wb[...,::-1]
                    img_2_wb = img_2_wb[...,::-1]

                    img_1_wb = np.clip(img_1_wb,0,SATURATION-BLACK_LEVEL)
                    img_12_wb = np.clip(img_12_wb,0,SATURATION-BLACK_LEVEL)
                    img_12_mixed_wb = np.clip(img_12_mixed_wb,0,SATURATION-BLACK_LEVEL)
                    img_2_wb = np.clip(img_2_wb,0,SATURATION-BLACK_LEVEL)
              
                    cv2.imwrite(os.path.join(SAVEPATH,'relabelled',place+"_1_wb.png"),img_1_wb)
                    cv2.imwrite(os.path.join(SAVEPATH,'relabelled',place+"_12_wb.png"),img_12_wb)
                    cv2.imwrite(os.path.join(SAVEPATH,'relabelled',place+"_12_mixed_illum_wb.png"),img_12_mixed_wb)
                    cv2.imwrite(os.path.join(SAVEPATH,'relabelled',place+"_2__wb.png"),img_2_wb)
            elif placeInfo["NumOfLights"] == 3:
                if place != "Place1095" :
                    continue
                print(place)
                Light1 = np.array(placeInfo["Light1"])
                Light2 = np.array(placeInfo["Light2"])
                Light3 = np.array(placeInfo["Light3"])
                
                single_image = place + '_1'
                multi_image_12 = place + '_12'
                multi_image_13 = place + '_13'
                multi_image_123 = place + '_123'

                img_1 = cv2.imread(DATAPATH + place +'/'+ single_image + ".tiff", cv2.IMREAD_UNCHANGED) - BLACK_LEVEL
                img_12 = cv2.imread(DATAPATH + place +'/' + multi_image_12 + ".tiff", cv2.IMREAD_UNCHANGED) - BLACK_LEVEL        
                img_13 = cv2.imread(DATAPATH + place +'/' + multi_image_13 + ".tiff", cv2.IMREAD_UNCHANGED) - BLACK_LEVEL        
                img_123 = cv2.imread(DATAPATH + place +'/' + multi_image_123 + ".tiff", cv2.IMREAD_UNCHANGED) - BLACK_LEVEL        
                

                img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB)
                img_12 = cv2.cvtColor(img_12,cv2.COLOR_BGR2RGB)
                img_13 = cv2.cvtColor(img_13,cv2.COLOR_BGR2RGB)
                img_123 = cv2.cvtColor(img_123,cv2.COLOR_BGR2RGB)
                

                img_1_int16 = img_1.astype("int16")
                img_12_int16 = img_12.astype("int16")
                img_13_int16 = img_13.astype("int16")
                img_123_int16 = img_123.astype("int16")

                img_2_int16 = np.clip(img_12_int16 - img_1_int16,0,SATURATION-BLACK_LEVEL).astype("int16")
                img_3_int16 = np.clip(img_123_int16 - img_12_int16,0,SATURATION-BLACK_LEVEL).astype("int16")
                img_3_int16_l2 = np.clip(img_13_int16 - img_1_int16,0,SATURATION-BLACK_LEVEL).astype("int16")


                img_1_wb = (img_1_int16 / Light1)
                img_2_wb = (img_2_int16 / Light2)
                img_3_wb = (img_3_int16 / Light3)
                img_3_2_wb = (img_3_int16_l2 / Light3)


                # mixed_illumination_map = get_illumination_map_3(img_123_int16,(img_1_int16,img_2_int16,img_3_int16),(Light1,Light2,Light3))        
                mixed_illumination_map = get_illumination_map_3_v2((img_12_int16,img_13_int16,img_123_int16),(img_1_int16,img_2_int16,img_3_int16_l2),(Light1,Light2,Light3))        
                fname_illumination = SAVEPATH + "/relabelled/" + multi_image_123 + "_mixedillum.npy"
                np.save(fname_illumination,mixed_illumination_map)

                both_zero = (mixed_illumination_map[:,:,1] == ZERO_MASK)
                no_gt = (mixed_illumination_map[:,:,1] == NO_GT)
                mask = np.logical_or(both_zero, no_gt) 
                
                img_123_mixed_denominator = np.where(mixed_illumination_map <= 0.0, 1, mixed_illumination_map)
                img_123_mixed_wb = img_123_int16 / img_123_mixed_denominator

                if VISUALIZE :
                    img_1_wb = img_1_wb[...,::-1]
                    img_2_wb = img_2_wb[...,::-1]
                    img_3_wb = img_3_wb[...,::-1]
                    img_3_2_wb = img_3_2_wb[...,::-1]
                    img_123_mixed_wb = img_123_mixed_wb[...,::-1]

                    img_1_wb = np.clip(img_1_wb,0,SATURATION-BLACK_LEVEL)
                    img_2_wb = np.clip(img_2_wb,0,SATURATION-BLACK_LEVEL)
                    img_3_wb = np.clip(img_3_wb,0,SATURATION-BLACK_LEVEL)
                    img_3_2_wb = np.clip(img_3_2_wb,0,SATURATION-BLACK_LEVEL)
                    img_123_mixed_wb = np.clip(img_123_mixed_wb,0,SATURATION-BLACK_LEVEL)

                    cv2.imwrite(os.path.join(SAVEPATH,'relabelled',place+"_1_wb.png"),img_1_wb)
                    cv2.imwrite(os.path.join(SAVEPATH,'relabelled',place+"_2_wb.png"),img_2_wb)
                    cv2.imwrite(os.path.join(SAVEPATH,'relabelled',place+"_3_wb.png"),img_3_wb)
                    cv2.imwrite(os.path.join(SAVEPATH,'relabelled',place+"_3_2_wb.png"),img_3_2_wb)
                    cv2.imwrite(os.path.join(SAVEPATH,'relabelled',place+"_123_mixed_illum_ratio_wb.png"),img_123_mixed_wb)
                    
                    
        except Exception as e :
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            print(place)
            continue

    exit()
    
    result = dict_save(place_dict)
    
    with open('./results/place_dict.json','w') as out_file:
        json.dump(place_dict,out_file,indent=4)
    
    with open('./results/result.json','w') as out_file :
        json.dump(result,out_file,indent=4)
   
    # with open('./results/coefficients_ratio.json','w') as out_file :
    #     json.dump(coefficient_dict,out_file,indent=4)
if __name__ == "__main__" :
    print("Generating",CAMERA,"mixture map...")
    main()
     
    