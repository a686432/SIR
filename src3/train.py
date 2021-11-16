import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import  transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import sys
import os
import argparse
from data_loader import *
import numpy as np
import time
import math
import random
import config
from utils import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
from myeval import EvalTool
from GAN import Discriminator
import misc
from multi_gpu import BalancedDataParallel


################################### load and save model ##################################
def save_model(model, path):
    print("Saving...")
    torch.save(model.module.state_dict(), path)

def save_model_m(model, path):
    print("Saving...")
    torch.save(model.state_dict(), path)

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
                print(name)
                continue
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state[name].copy_(param)

def load_model_m(model, path):
    model.load_state_dict(torch.load(path,map_location='cuda:0'),strict=False)
def load_model(model, path):
    model.load_state_dict(torch.load(path,map_location='cuda:0'))



###########################################################################################
def train(epoch):
    sys.stdout.write('\n')
    print ("Training... Epoch = %d" % epoch)

    model.train()
    criterion.train()

    x_step_sum = len(train_loader)
    for batch_idx,(data, target, ori_img) in tqdm(enumerate(train_loader), total = len(train_loader)):
        x_step = batch_idx + (epoch - 1) * len(train_loader)
        batch_size = data.shape[0]
        conf , shape, albedo, camera_exp = model(data)
        metrics,train_result = criterion(camera_exp, shape, target,albedo,ori_img,conf)


######################################## parse loss ########################################
        loss_3d = metrics['loss_3d']
        re_s =  metrics['regular_loss_shape'] 
        re_e = metrics['regular_loss_exp'] 
        loss_edge = metrics['loss_edge']
        pixel_loss = metrics['pixel_loss']
        res_shape_loss = metrics['res_shape_loss']
        images = train_result['images_s']
        illum_images = train_result['illum_images_s']
        ver_color = train_result['ver_color_s']
        center_norm = train_result['center_norm']
        loss_perc_im = metrics['loss_perc_im'] 
        loss_albedo_reg = metrics['loss_albedo_reg']
        albedo_smooth_loss = metrics['albedo_smooth_loss']
        albedo_sharp_loss = metrics['albedo_sharp_loss']
        shape_smooth_loss = metrics['shape_smooth_loss']
        loss_fr = metrics['loss_fr']
        loss_fr_all = metrics['loss_fr_all']
        albedo_res_loss = metrics['albedo_res_loss'] 
        center_loss = metrics['loss_center']
        weighted_center = metrics['weighted_center_loss'] 
        


        loss = loss_3d.mean() + *config.tik_shape_weight*re_s.mean() + config.tik_exp_weight*re_e.mean()+ \
                config.weight_edge_lm*loss_edge.mean()+config.weight_pixel_loss*pixel_loss.mean()+ \
                res_shape_loss.mean()*config.weight_res_shape_loss+loss_albedo_reg.mean()*config.weight_albedo_reg + \
                albedo_smooth_loss.mean()*config.weight_albedo_smooth + loss_fr_all.mean()*config.mix_loss_weight_fr+ \
                config.weight_loss_perc_im*loss_perc_im.mean() + albedo_res_loss.mean()* config.weight_res_albedo_loss + \
                config.weight_albedo_sharp * albedo_sharp_loss.mean()
        
            
            
        if config.use_weighted_center_loss:
            scheduler.step()
            optimizer4nn.zero_grad()
            optimizer4center.zero_grad()
            loss.backward()
            #print(weighted_center)
            #param = fr_loss_sup.parameters()
            # for group in optimizer4center.param_groups:
            #     print(group["params"][0].grad.abs().mean())
            # for param in fr_loss_sup.parameters():
            #     print(param.grad.abs().mean())
            #     print(param.abs().mean()) 
            gt_id = target['id'].long()
            #print(gt_id.shape)
            p = optimizer4center.param_groups[0]["params"][0]
            # for group in optimizer4center.param_groups:
            #    print(group["params"][0].grad.abs().mean())
            #print(p.grad.data[gt_id].shape)
            p.grad.data[gt_id] = p.grad.data[gt_id] * weighted_center.reshape(batch_size,1)
            # for group in optimizer4center.param_groups:
            #    print(group["params"][0].grad.abs().mean())
            # for param in fr_loss_sup.parameters():
            #     print(param.grad.abs().mean())
            #     #print(param.abs().mean()) 
            
            #print(param[0])
                   
            # param.grad.data = param.grad.data / weighted_center.reshape(batch_size,1)
            optimizer4nn.step()
            optimizer4center.step()
            # for group in optimizer4center.param_groups:
            #     print('ddddddddd',group["params"][0].grad.abs().mean())
            
            
            # optimizer4center.zero_grad()
            # param = fr_loss_sup.parameters():
            # print(param.grad.data)
            # weighted_center_loss.backward()
            # optimizer4center.step()

        elif config.use_center_loss:
            scheduler.step()
            optimizer4nn.zero_grad()
            optimizer4center.zero_grad()
            loss.backward()
            optimizer4nn.step()
            optimizer4center.step()
        else:
            scheduler.step()
            optimizer4nn.zero_grad()
            loss.backward()
            optimizer4nn.step()



        '''
            optimizer4nn.zero_grad()
            loss.backward(retain_graph=True)
            optimizer4nn.step()
            model.zero_grad()
            criterion.zero_grad()
            optimizer4cl.zero_grad()
            weighted_centers.backward()
            optimizer4cl.step()
        '''


        if config.re_type_gan:
            scheduler_d_optimizer.step()
        # st3 = time()-st-st1-st2
        # print('backward time:', st3)
        
        if x_step % board_loss_every == 0:# and x_step!=0:
            if config.mode == "bfm17":
                exp_para =  camera_exp[:, 7:107]
                albedo = albedo.data.cpu().numpy()
            else:
                exp_para =  camera_exp[:, 7:36]
                albedo, _ = albedo
                albedo = albedo.data.cpu().numpy()
                
            shape = shape.data.cpu().numpy()
            exp_para = exp_para.data.cpu().numpy()
            
            norm_shape = np.linalg.norm(shape, axis=1).mean()
            norm_exp = np.linalg.norm(exp_para, axis=1).mean()
            albedo_norm = np.linalg.norm(albedo, axis=1).mean()
            albedo_std = np.linalg.norm(albedo, axis=1).std()
            std_shape = np.linalg.norm(shape, axis=1).std()
            loss_3d = loss_3d.data.cpu().numpy().mean()
            loss_edge = loss_edge.data.cpu().numpy().mean()
            pixel_loss = pixel_loss.data.cpu().numpy().mean()
            norm_mean, norm_std = eval_3d(model)
            res_shape_loss = res_shape_loss.data.cpu().numpy().mean()
            loss_perc_im = loss_perc_im.data.cpu().numpy().mean()
            loss_albedo_reg = loss_albedo_reg.data.cpu().numpy().mean()
            albedo_smooth_loss = albedo_smooth_loss.data.cpu().numpy().mean()
            shape_smooth_loss = shape_smooth_loss.data.cpu().numpy().mean()
            loss_fr = loss_fr.data.cpu().numpy().mean()
            albedo_res_loss = albedo_res_loss.data.cpu().numpy().mean()
            albedo_sharp_loss = albedo_sharp_loss.data.cpu().numpy().mean()
            center_norm = center_norm.data.cpu().numpy().mean()
            center_loss = center_loss.data.cpu().numpy().mean()
            with open(config.log_file, 'a+') as f:
                 f.write('step:{} fr_loss:{} ld_loss:{}'.format(x_step,loss_fr,loss_3d)+'\n')  
            #writer.add_scalar('loss_max', loss_max, x_step)
            writer.add_scalar('reject_rate', 1.0*reject_num/board_loss_every, args.batch_size*num_gpus* x_step)
            writer.add_scalar('norm/shape', norm_shape, args.batch_size*num_gpus* x_step)
            writer.add_scalar('norm/shape_std', std_shape, args.batch_size*num_gpus* x_step)
            writer.add_scalar('norm/norm_std', norm_shape/std_shape, args.batch_size*num_gpus* x_step)
            writer.add_scalar('norm/exp', norm_exp, args.batch_size*num_gpus* x_step)
            writer.add_scalar('norm/texture', albedo_norm, args.batch_size*num_gpus* x_step)
            writer.add_scalar('norm/texture_std', albedo_std, args.batch_size*num_gpus* x_step)
            writer.add_scalar('norm/mean', norm_mean, args.batch_size*num_gpus* x_step)
            writer.add_scalar('norm/std', norm_std, args.batch_size*num_gpus* x_step)
            writer.add_scalar('norm/center_norm', center_norm, args.batch_size*num_gpus* x_step)
            #writer.add_scalar('loss/loss_d',dfe,args.batch_size*num_gpus* x_step)
            writer.add_scalar('loss/loss_3d',loss_3d,args.batch_size*num_gpus* x_step)
            writer.add_scalar('loss/loss_edge',loss_edge,args.batch_size*num_gpus* x_step)
            writer.add_scalar('loss/pixel_loss',pixel_loss,args.batch_size*num_gpus* x_step)
            writer.add_scalar('loss/res_loss',res_shape_loss,args.batch_size*num_gpus* x_step)
            writer.add_scalar('loss/loss_perc_im',loss_perc_im,args.batch_size*num_gpus* x_step)
            writer.add_scalar('loss/loss_albedo_reg',loss_albedo_reg,args.batch_size*num_gpus* x_step)
            writer.add_scalar('loss/albedo_smooth_loss',albedo_smooth_loss,args.batch_size*num_gpus* x_step)
            writer.add_scalar('loss/albedo_sharp_loss',albedo_sharp_loss,args.batch_size*num_gpus* x_step)
            writer.add_scalar('loss/shape_smooth_loss',shape_smooth_loss,args.batch_size*num_gpus* x_step)
            writer.add_scalar('loss/loss_fr',loss_fr,args.batch_size*num_gpus* x_step)
            writer.add_scalar('loss/albedo_res_loss',albedo_res_loss,args.batch_size*num_gpus* x_step)
            writer.add_scalar('loss/center_loss',center_loss,args.batch_size*num_gpus* x_step)





            #writer.add_scalar('loss/loss_g',regular_loss.data.cpu().numpy()/config.mix_loss_weight_advargs.batch_size*num_gpus* x_step)
            reject_num = 0
            writer.add_scalar('loss', loss.item(), args.batch_size*num_gpus* x_step)

            if not config.weight_pixel_loss == 0:
                #abedlo,abedlo_s = abedlo
                albedo = train_result['albedo']
                albedo_a = train_result['albedo_a']
                albedo_s = train_result['albedo_s']
                #colors = C['tex'].T; triangles = C['tri'].T-1; uv_coords=C['uv']
                #print(colors.shape,uv_coords.shape,triangles.shape)
                if config.use_ConvTex:
                    uv_albedo_map = albedo[0].permute(2,0,1).clamp(min=0,max=1)
                    uv_albedo_map_a = albedo_a[0].permute(2,0,1).clamp(min=0,max=1)
                    uv_albedo_map_s = albedo_s[0].permute(2,0,1).clamp(min=0,max=1)
                    #uv_albedo_map_s = albedo.clamp(min=0,max=1)
                    # print(uv_albedo_map.shape)
                    # print(uv_texture_map.shape)
                    writer.add_image('abedlo',uv_albedo_map,args.batch_size*num_gpus* x_step)
                    writer.add_image('abedlo_s',uv_albedo_map_s ,args.batch_size*num_gpus* x_step)
                    writer.add_image('abedlo_a',uv_albedo_map_a ,args.batch_size*num_gpus* x_step)


                # else:
                #     uv_coords = process_uv(uv_coords, config.uv_h, config.uv_w)
                #     attribute = abedlo[0].clamp(min=0,max=1).data.cpu().numpy().reshape(-1,3)
                #     uv_albedo_map = render_colors(uv_coords, config.triangles, attribute, config.uv_h, config.uv_w, c=3)
                #     attribute = ver_color.view(batch_size,-1,3)[0].clamp(min=0,max=1).data.cpu().numpy().reshape(-1,3)
                #     uv_texture_map = render_colors(uv_coords, config.triangles, attribute, config.uv_h, config.uv_w, c=3)
                #     writer.add_image('abedlo',np.squeeze(uv_albedo_map).transpose(2,0,1) ,args.batch_size*num_gpus* x_step)
                #     writer.add_image('texture',np.squeeze(uv_texture_map).transpose(2,0,1) ,args.batch_size*num_gpus* x_step)




                # im = images[0,...,:3]
                # images[0,...,:3].permute(2,0,1)
                #print(uv_texture_map)
                #print(images.shape)
                ori_img = ori_img.to(images.device)
                print(images[0,...,3].max())
                mix_image = torch.where(images[0,...,3].unsqueeze(-1)==1, images[0,...,:3],ori_img.permute(0,2,3,1)[0,...,:3] ).clamp(min=0,max=1)
                
                illum_images = torch.where(images[0,...,3].unsqueeze(-1)==1, illum_images[0,...,:3],ori_img.permute(0,2,3,1)[0,...,:3] ).clamp(min=0,max=1)
            
                writer.add_image('image',images[0,...,:3].permute(2,0,1) ,args.batch_size*num_gpus* x_step)
                writer.add_image('ori_image',ori_img[0] ,args.batch_size*num_gpus* x_step)
                writer.add_image('mix_image',mix_image.permute(2,0,1) ,args.batch_size*num_gpus* x_step)
                writer.add_image('mix_re_image',illum_images.permute(2,0,1) ,args.batch_size*num_gpus* x_step)

                if config.use_confidence_map:
                    conf_a, conf_lm = conf
                    conf_a_map = conf_a[0].sigmoid()
                    conf_lm_map = conf_lm[0].sigmoid()
                    writer.add_image('conf_a',conf_a_map,args.batch_size*num_gpus* x_step)
                    writer.add_image('conf_lm',conf_lm_map ,args.batch_size*num_gpus* x_step)

            

        if x_step % board_eval_every == 0 and x_step!=0:
            model.eval()
            criterion.eval()
            eval_tool.update_tb(model, args.batch_size*num_gpus* x_step, eval_ytf = (x_step % (board_eval_every*10) == 0), emb_idx = 1, mask = 12)
            model.train()
            criterion.train()
        if x_step % board_save_every == 0 and x_step!=0:
            save_model(model, config.result_dir + 'model/train_stage3_x_step_%d_' % x_step + config.prefix + '.pkl')
            save_model(criterion, config.result_dir + 'model/train_stage3_x_step_criterion_%d_' % x_step + config.prefix + '.pkl')
            save_model_m(optimizer4nn, config.result_dir + 'model/train_stage3_x_step_optimizer4nn_%d_' % x_step + config.prefix + '.pkl')
         

def process_uv(uv_coords, uv_h = 112, uv_w = 112):
    uv_coords[:,0] = uv_coords[:,0]*(uv_h - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords

def eval_3d(model):
    predicts = np.array([])
    model.eval()
    #print("Evaluating...")
    for batch_idx,(data, target,imname) in enumerate(eval_loader_micc):
        data, target =  data.to(device), target.to(device)
        _, pred_shape, feat, pose_expr= model(data)
        pred_shape = pred_shape.data.cpu().numpy()
        norm = np.linalg.norm(pred_shape, axis=1)
        #predicts = norm
        predicts = np.append(predicts, norm)
        #print(np.linalg.norm(pred_shape, axis=1))
    #print(predicts.shape)
    model.train()
    return predicts.mean(),predicts.std()

def eval_vis(model,criterion):
    model.eval()
    criterion.eval()
    for image_name in image_path_list:
        image_name = image_name.replace('\n','')
        image = Image.open(image_name)
        image = config.transform_eval(image).unsqueeze(0).to(device)
    result = model(image)
    #self.bfm()
    _, shape_para, albedo_para, pred_camera_exp = result
    criterion.loss_vdc_3ddfa.pixel_loss
    #shape_para = result[2]
    if "bfm17" == config.mode:
        exp_para = pred_camera_exp[:, 7:107]
        camera_para = pred_camera_exp[:, 0:7]
        illum = pred_camera_exp[:,107:134]
    else:
        exp_para = pred_camera_exp[:, 7:36]
        camera_para = pred_camera_exp[:, 0:7]
        illum = pred_camera_exp[:,36:63]


    model.train()
    criterion.train()



def isPointInTri(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: (2,). [u, v] or [x, y] 
        tri_points: (3 vertices, 2 coords). three vertices(2d points) of a triangle. 
    Returns:
        bool: true for in triangle
    '''
    tp = tri_points

    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)

def get_point_weight(point, tri_points):
    ''' Get the weights of the position
    Methods: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
     -m1.compute the area of the triangles formed by embedding the point P inside the triangle
     -m2.Christer Ericson's book "Real-Time Collision Detection". faster.(used)
    Args:
        point: (2,). [u, v] or [x, y] 
        tri_points: (3 vertices, 2 coords). three vertices(2d points) of a triangle. 
    Returns:
        w0: weight of v0
        w1: weight of v1
        w2: weight of v3
     '''
    tp = tri_points
    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    w0 = 1 - u - v
    w1 = v
    w2 = u

    return w0, w1, w2



def render_colors(vertices, triangles, colors, h, w, c = 3):
    ''' render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3] 
        colors: [nver, 3]
        h: height
        w: width    
    Returns:
        image: [h, w, c]. 
    '''
    assert vertices.shape[0] == colors.shape[0]
    
    # initial 
    image = np.zeros((h, w, c))
    depth_buffer = np.zeros([h, w]) - 999999.

    for i in range(triangles.shape[0]):
        tri = triangles[i, :] # 3 vertex indices

        #print(tri,vertices[tri, :2])
        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(vertices[tri, 0]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(vertices[tri, 1]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], vertices[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], vertices[tri, :2])
                point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]

                if point_depth > depth_buffer[v, u]:
                    depth_buffer[v, u] = point_depth
                    #print(w0, w1, w2,w0*colors[tri[0], :] + w1*colors[tri[1], :] + w2*colors[tri[2], :])
                    image[v, u, :] = w0*colors[tri[0], :] + w1*colors[tri[1], :] + w2*colors[tri[2], :]
    return image






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face recognition with CenterLoss')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--verify', '-v', default=False, action='store_true', help='Verify the net')
    parser.add_argument('--gpu', default="0", help="select the gpu")
    parser.add_argument('--net', default="sphere", help="type of network")
    parser.add_argument('--loss', default="cos", help="type of loss fuction")
    parser.add_argument('--loadfile', '-l' , default="/data3/jdq/fs2_81000.cl", help="model parameter filename")
    parser.add_argument('--savefile', '-S' , default="../dict.cl", help="model parameter filename")
    parser.add_argument('--param-fp-train', default='./train.configs/param_aligned.pkl')
    parser.add_argument('--filelists-train', default='./train.configs/train_aug_120x120.list.train')
    parser.add_argument('--epoch', '-e' , default=50, help="training epoch")
    parser.add_argument('--lfw_vallist', '-vl' , default="/data1/jdq/lfw_crop/")
    parser.add_argument('--lfw_pairlist', '-pl' , default="../lfw_pair.txt")
    parser.add_argument("-b", "--batch_size", help="batch_size", default=80, type=int)
    parser.add_argument('--number_of_class','-nc', default=8631,type=int, help="The number of the class")
    ###################################### init ##########################################

    ###################################### transform ######################################
    print("********************")
    args = parser.parse_args()
    dict_file=args.loadfile     
    if config.use_rawnet:
        from rawnet import sphere64a
    else:
        from net import sphere64a
    ############################################ devices ####################################
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    num_gpus=len(args.gpu.split(","))
    gpu_ids = range(num_gpus)
    print('num of GPU is ' + str(num_gpus))
    print('GPU is ' + str(gpu_ids))
    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.set_device(gpu_ids[0])
    config.use_cuda = use_cuda
    config.device = device
    config.device_ids = gpu_ids

    ######################################### model & loss #######################################
    print("loading model...")
    #torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    model = sphere64a(pretrained=False,model_root=dict_file, stage = 3)
    if config.start_from_warmpixel:
        load_model_m(model,config.checkpoint_warm_pixel+'.pkl') 
    elif config.start_from_warm3d:
        load_model_m(model,config.checkpoint_warm3d+'.pkl') 
    model = model.to(device)
    #model = torch.nn.DataParallel(model,device_ids=gpu_ids)
    if num_gpus>1:
        model = BalancedDataParallel(16,model,device_ids=gpu_ids)
    else:
        model = torch.nn.DataParallel(model,device_ids=gpu_ids)

    #
    if config.use_face_recognition_constraint:
   
        from fr_loss import CosLinear, CosLoss, CenterLoss,softmaxLinear,softmaxLoss
        #criterion = mixed_loss_FR_batch(fr_ip = ip, fr_loss =  fr_loss_sup=fr_loss_sup,d=D).to(device)
        #fr_loss_sup = CenterLoss(num_classes = args.number_of_class, dim_hidden = 99)
        from loss_batch import mixed_loss_FR_batch
        if config.use_center_loss:
            fr_loss_sup = CenterLoss(num_classes = args.number_of_class, dim_hidden = 199)
            # ip = softmaxLinear(in_features = 199, out_features = args.number_of_class)
            # fr_loss = softmaxLoss()
            ip = CosLinear(in_features = 199, out_features = args.number_of_class)
            fr_loss = CosLoss(num_cls = args.number_of_class, alpha=0.2)
            criterion = mixed_loss_FR_batch(fr_ip=ip,fr_loss = fr_loss,fr_loss_sup=fr_loss_sup)
        else: 
            #ip = softmaxLinear(in_features = 199, out_features = args.number_of_class)
            # fr_loss = softmaxLoss()
            criterion = mixed_loss_FR_batch(fr_ip=ip,fr_loss = fr_loss)
            ip = CosLinear(in_features = 199, out_features = args.number_of_class)
            fr_loss = CosLoss(num_cls = args.number_of_class, alpha=0.2)
        
        if config.start_from_warm3d:
            load_model(criterion.loss_3d_func,config.checkpoint_warm3d+'_c.pkl')
        if config.start_from_warmpixel:
            load_model_m(criterion,config.checkpoint_warm_pixel+'_c.pkl')
        criterion.loss_3d_func.reset_albedo()
            
    elif config.use_mix_data:
        from loss_batch import mixed_loss_FR_batch
        criterion = mixed_loss_FR_batch()
        if config.start_from_warm3d:
            load_model(criterion.loss_3d_func,config.checkpoint_warm3d+'_c.pkl')
        if config.start_from_warmpixel:
            load_model(criterion.loss_3d_func,config.checkpoint_warm_pixel+'_c.pkl')
    else:
        from loss_batch import loss_vdc_3ddfa
        criterion = loss_vdc_3ddfa()
        if config.start_from_warm3d:
            load_model_m(criterion,config.checkpoint_warm3d+'_c.pkl')
        if config.start_from_warmpixel:
            load_model_m(criterion,config.checkpoint_warm_pixel+'_c.pkl')
        criterion.reset_albedo()        
        
        # load_model(criterion,'/data/jdq/model_s/train_warm3d_criterion_26511_Clinear.pkl')
        # load_model(criterion,'/data/jdq/model_s/train_warm3d_criterion_26511_Clinear.pkl')
    
    criterion = criterion.to(device)
    #criterion = nn.DataParallel(criterion,device_ids=gpu_ids)
    if num_gpus>1:
        criterion = BalancedDataParallel(16,criterion,device_ids=gpu_ids)
    else:
        criterion = nn.DataParallel(criterion,device_ids=gpu_ids)
    
    
    ######################################### dataset ############################################

    if not args.verify:
        if config.use_mix_data:
            trainset = VGG2MixDataset(
                max_number_class=args.number_of_class,
                indexfile="../file_path_list_vgg2.txt",
                transform =  config.transform_train,
                ddfa_root=config.ddfa_root,
                ddfa_filelists=args.filelists_train,
                ddfa_param_fp=args.param_fp_train,
                mix = True)
        else:
            trainset = DDFADataset(
                root = config.ddfa_root, 
                filelists = args.filelists_train,
                param_fp = args.param_fp_train,
                transform = config.transform_train)


        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size*num_gpus, shuffle=True, num_workers=4)
        evalset = AFLW2000DataSet(root=config.aflw_data_root_path,transform=config.transform_eval)
        eval_loader = torch.utils.data.DataLoader(evalset, batch_size=8*num_gpus, shuffle=False, num_workers=4)
        evalset_micc = MICCDataSet(root=config.micc_image_root, filelist=config.micc_filelist, transform=config.transform_train)
        eval_loader_micc = torch.utils.data.DataLoader(evalset_micc, batch_size=8*num_gpus, shuffle=False, num_workers=4)


    
    lr=args.lr
    if config.use_mix_data:
        lr_change1 = int(0.4 * len(train_loader))
        lr_change2 = int(1.0 * len(train_loader))
        lr_change3 = int(1.5 * len(train_loader)) 
        lr_change4 = int(2.0 * len(train_loader))
    else:
        lr_change1 = int(config.lr_change1 * len(train_loader))
        lr_change2 = int(config.lr_change2 * len(train_loader))
        lr_change3 = int(config.lr_change3 * len(train_loader)) 
        lr_change4 = int(config.lr_change4 * len(train_loader))

    if config.use_center_loss:
        optimizer4nn = torch.optim.Adam([{'params':filter(lambda p: p.requires_grad, model.parameters())},\
            {'params':filter(lambda p: p.requires_grad, criterion.module.loss_3d_func.parameters())},\
            {'params':filter(lambda p: p.requires_grad, criterion.module.fr_loss.parameters())},\
            {'params':filter(lambda p: p.requires_grad, criterion.module.fr_ip.parameters())},\
            ],lr=args.lr, weight_decay=5e-4)
        optimizer4center = torch.optim.Adam(filter(lambda p: p.requires_grad, criterion.module.fr_loss_sup.parameters()),lr=config.center_lr)

    else:
        optimizer4nn = torch.optim.Adam([{'params':filter(lambda p: p.requires_grad, model.parameters())},{'params':filter(lambda p: p.requires_grad, criterion.parameters())}],lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer4nn, milestones=[lr_change1,lr_change2,lr_change3,lr_change4], gamma=0.1)


    #optimizer4nn = nn.DataParallel(optimizer4nn,device_ids=gpu_ids)
    
   
    

    ########################################################## GAN ################################################################
    if config.re_type_gan:
        lr_p = pow(20,1.0/lr_change1)
        lr_d = lambda x_step:(lr_p ** x_step) / (int( x_step > lr_change1)*4+1) / (int( x_step > lr_change2)*4+1)   
        discriminator_activation_function = torch.relu
        d_hidden_size = 1024
        d_output_size = 1
        sgd_momentum = 0.9
        D = Discriminator(input_size= 99,
                            hidden_size=d_hidden_size,
                            output_size=d_output_size,
                            f=discriminator_activation_function).cuda()
        D = torch.nn.DataParallel(D,device_ids=gpu_ids)
        d_optimizer = torch.optim.Adam(D.parameters(), lr=config.d_learning_rate,weight_decay=5e-4)
        #scheduler_d_optimizer = optim.lr_scheduler.MultiStepLR(d_optimizer, milestones=[lr_change1,lr_change2], gamma=0.2)
        scheduler_d_optimizer = optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lr_d)
        criterion_B = torch.nn.BCELoss().cuda()
        criterion_B = torch.nn.DataParallel(criterion_B,device_ids=gpu_ids)
    ############################################################################## log ########################################


    


    iter_num=0
    train_loss=0
    correct=0
    total=0
    eval_loss=0
    eval_loss_v=0
    
    board_loss_every = len(train_loader)//100 #32686//100
   
    board_eval_every = len(train_loader)//10 #32686//100
    board_save_every = len(train_loader)
    

    if config.use_mix_data:
        board_loss_every =  len(train_loader) // 600 
        board_eval_every = len(train_loader) // 60 #32686//100
        board_save_every = len(train_loader) // 6


    print('board_loss_every '+ str(board_loss_every)+'...')
    print('board_eval_every '+ str(board_eval_every)+'...')


    writer = SummaryWriter(config.result_dir+'tmp_log5/train_recog_' + config.prefix + '_'+ str(config.tik_shape_weight) + '_' + str(config.weight_edge_lm)+ '_' + time.strftime('%m-%d:%H-%M-%S',time.localtime(time.time())) + '/')

    eval_tool = EvalTool(transform = config.transform_eval, criterion=criterion, tb_writer = writer, batch_size=16*num_gpus)

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer4nn, patience=board_loss_every*40, verbose=True)
    loss_mean = 10000
    reject_num = 0
    feat_norm = 0
    norm_shape = 0
    norm_exp = 0


    if config.start_from_warm3d:
        pass
    ################################################### Train ##################################
    if args.verify:
        if not os.path.exists(dict_file):
            print("Cannot find the model!")
        else:
            print("Loading...\n")
            print("evaluting...")
    else:
        for epoch in range(int(args.epoch)):
            train(epoch+1)
