
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from PIL import Image
import torchvision
from torchvision import transforms
#from torchvision.models import resnet10s1
import pytorch_lightning as pl
from model.facenet import Backbone,ConfNet
from model.bfm import BFM17_batch
from model.loss import loss_vdc_3ddfa
from model.renderer import Renderer
from dataset import MICCDataSet, VGG2Dataset,AFLW2000DataSet,DataSet_300WLP, VGG2MixDataset, LFWDataSet, CFPDataSet, AgeDBDataSet, VISDataSet, YTFDataset, BosregDataSet
import matplotlib
import matplotlib.pyplot as plt
import io
import cv2
from torch.optim.lr_scheduler import StepLR
import os
from model.fr_metrics import evaluate_fr
from model.rmse_metric import get_rmse_from_param
plt.switch_backend('agg')
from script.testSirTerm import Test_SIR_term_Bos

class Encoder(pl.LightningModule):
    def __init__(self, hp):
        super(Encoder, self).__init__()
        '''
        self.bfm: The 3DMM model: The input are 3DMM parameter and output the face geometric and texture.
        self.model: The network the regress the 
        '''
        self.hp = hp
        self.bfm = BFM17_batch(hp=hp)
        #print(self.bfm.ev_shape)
        self.model = Backbone(hp.model.net_depth, hp.model.drop_ratio,
            bfm_ev_shape=self.bfm.ev_shape,
            bfm_ev_exp=self.bfm.e_exp,
            bfm_ev_tex=self.bfm.albedo_ev,
            hp = hp
            )
        
        self.Renderer  = Renderer(self.bfm.tris,hp)
        #self.confnet =  ConfNet(nf=32)
        
        self.loss = loss_vdc_3ddfa(self.bfm,self.Renderer)
       
        
        


    def forward(self, image):
        #print('bbbbbbbbb')
        parameters = self.model(image)
        recon_face, texture = self.bfm(parameters)
        confmap = None
        return parameters,recon_face,texture, confmap




    def training_step(self, batch, batch_idx):
        image  = batch['image']
        raw_img = batch['raw_img']
        landmark = batch['landmark']
        identity = batch['id']
        
        parameters,recon_face, texture,confmap = self(image)
        loss = self.loss(parameters,recon_face,landmark,texture,raw_img,confmap,identity=identity)
        

        self.logger.experiment.add_scalar("Loss/Loss landmark", loss['landmark'].item(), self.global_step)
        self.logger.experiment.add_scalar("Loss/Loss kl", loss['kl_loss'].item(), self.global_step)
        self.logger.experiment.add_scalar("Loss/Loss face_recognition", loss['fr_loss'].item(), self.global_step)
        self.logger.experiment.add_scalar("Loss/para regular shape Loss", loss['para_regular_shape'].item(), self.global_step)
        self.logger.experiment.add_scalar("Loss/para regular exp Loss", loss['para_regular_exp'].item(), self.global_step)
        #self.logger.experiment.add_scalar("Loss/para_res_shape", loss['para_res_shape'].item(), self.global_step)
        self.logger.experiment.add_scalar("Loss/Pixel Loss", loss['pixel_loss'].item(), self.global_step)
        self.logger.experiment.add_scalar("Loss/perceptual_loss", loss['perceptual_loss'].item(), self.global_step)
        self.logger.experiment.add_scalar("Loss/para regular tex Loss", loss['para_regular_tex'].item(), self.global_step)
        #self.logger.experiment.add_scalar("Loss/smooth_loss", loss['smooth_loss'].item(), self.global_step)
        #self.logger.experiment.add_scalar("Loss/sym_loss", loss['sym_loss'].item(), self.global_step)
        #self.logger.experiment.add_scalar("Loss/basis_regular_loss", loss['basis_regular_loss'].item(), self.global_step)
        self.logger.experiment.add_scalar("Loss/center_loss", loss['center_loss'].item(), self.global_step)
        #self.logger.experiment.add_scalar("Loss/basis_res_loss", loss['basis_res_loss'].item(), self.global_step)
        self.logger.experiment.add_scalar("Norm/shape para norm", parameters['shape_para'].norm(dim=1).mean().item(), self.global_step)
        self.logger.experiment.add_scalar("Norm/exp para norm", parameters['exp_para'].norm(dim=1).mean().item(), self.global_step)
        self.logger.experiment.add_scalar("Norm/tex para norm", parameters['tex_para'].norm(dim=1).mean().item(), self.global_step)
        self.logger.experiment.add_scalar("Norm/centers norm", self.loss.center_loss.centers.norm(dim=1).mean().item(), self.global_step)
        
        # self.logger.experiment.add_scalar("ID Loss", loss_id.item(), self.global_step)
        # self.logger.experiment.add_scalar("Reconstruction Loss", loss_rec.item(), self.global_step)
        # self.logger.experiment.add_scalar("GAN Loss", loss_GAN.item(), self.global_step)

        total_loss = loss['landmark'] + \
             self.hp.weight.para_regular_shape *loss['para_regular_shape'] + \
             self.hp.weight.para_regular_exp *loss['para_regular_exp'] + \
             self.hp.weight.pixel_loss * loss['pixel_loss'] + \
             self.hp.weight.para_regular_tex * loss['para_regular_tex'] + \
             self.hp.weight.perceptual_loss * loss['perceptual_loss'] + \
             self.hp.weight.fr_loss * loss['fr_loss'] + \
             self.hp.weight.kl_loss * loss['kl_loss'] + \
             self.hp.weight.center_loss * loss['center_loss']
            #  self.hp.weight.para_res_shape * loss['para_res_shape'] + \
            #  self.hp.weight.smooth_loss * loss['smooth_loss'] + \
            #  self.hp.weight.sym_loss * loss['sym_loss'] + \
            #  self.hp.weight.basis_regular_loss * loss['basis_regular_loss'] + \
              #+ \
            #  self.hp.weight.basis_res_loss * loss['basis_res_loss'] 
        # total_loss = self.hp.weight.fr_loss * loss['fr_loss'] + loss['landmark']*0
        return total_loss


    def validation_step(self, batch, batch_idx,dataset_idx):
        #print(batch, batch_idx,dataset_idx)
        #exit()
        if dataset_idx == 0:
            data,  target , raw_image, _ = batch
            total = 0
            parameters,recon_face, texture,confmap = self(data)
            pred_lms = self.bfm.get_landmark_68(face_pixels=recon_face).transpose(2,1)
            x_max, x_min, y_max, y_min = target[...,0].max(1)[0], target[...,0].min(1)[0], target[...,1].max(1)[0], target[...,1].min(1)[0]
            d_normalize = torch.sqrt((x_max - x_min) * (y_max - y_min))
            pts =  (pred_lms - target).norm(dim=-1).mean(1)
            rst = pts / d_normalize.float()
            if batch_idx == 0:
                illum = parameters['illum']
                #texture= self.bfm.tex_mu.expand(recon_face.shape[0],self.bfm.tex_mu.shape[0])*0+0.3
                result = self.Renderer(recon_face,texture,illum,shape=self.bfm.get_shape(parameters['shape_para']))
                seg_mask = self.loss.seg(raw_image)
                vis = result['images'].permute(0,3,1,2)[:,3,:,:].unsqueeze(1)*seg_mask.int()
                #print(torch.unique(result['images'].permute(0,3,1,2)[:,3,:,:]))
                mix_image = torch.where(vis!=0,
                            result['images'].permute(0,3,1,2)[:,:3,:,:],
                            raw_image).clamp(min=0,max=1)
                #result = self.Renderer(self.bfm.mu_shape.reshape(-1,3).T.unsqueeze(0),self.bfm.tex_mu.unsqueeze(0),illum)
                render_image = torchvision.utils.make_grid(result['images'].permute(0,3,1,2)[:,:3,:,:], nrow=4)
                mix_image = torchvision.utils.make_grid(mix_image, nrow=4)
                #seg_mask = torchvision.utils.make_grid(seg_mask, nrow=4)
                self.logger.experiment.add_image("Rendered Image", render_image, self.global_step)
                self.logger.experiment.add_image("Mixed Image", mix_image, self.global_step)
                # print(confmap[0].shape)
                # self.logger.experiment.add_image("conf Image", confmap[0], self.global_step)

            return {'rsts': rst.cpu() , 'shape_para': parameters['shape_para'].cpu()}
        elif dataset_idx == 4:
            data, target , raw_image = batch
            parameters,recon_face, texture,confmap = self(data)
            ver = self.bfm.get_shape(parameters['shape_para'])
            return {'shape': ver.cpu(), 'target': target.cpu()}
        else:
            data, data2, label = batch
            parameters1,_ , _, _ = self(data)
            parameters2,_ , _, _ = self(data2)
            return {'ip1':parameters1['shape_para'].cpu(),'ip2':parameters2['shape_para'].cpu(),'label':label.cpu() }


    def validation_epoch_end(self, outputs):
        ########################################################################################################
        print("Evaluating aflw2000...")
        # print(outputs)
        output_aflw = outputs[0]
        #exit()
        count = np.zeros(1000)
        rst = torch.stack([x['rsts'].reshape(-1) for x in output_aflw]).reshape(-1)
        shape_para = torch.stack([x['shape_para'] for x in output_aflw]).reshape(-1,199)
        for i in range(1000):
            count[i] = torch.sum(rst < i * 1.0 / 1000) *1.0 / rst.shape[0]
        mean_rst = torch.mean(rst)
        #import numpy as np
    
        prn = np.load(self.hp.val.prn_rst)
        _3ddfa = np.load(self.hp.val.ddfa_rst)
    

        x_range = 1000
        
        x = np.linspace(0, x_range / 1000., x_range)
        y = count * 100
        
        y_prn = prn['arr_0'] * 100
        y_3ddfa = _3ddfa['arr_0'] * 100
        
        plt.figure()
        plt.grid()
        plt.xlim(0,0.1)
        plt.ylim(0,100)
        plt.plot(x,y[:x_range], color='red', label='ours')
        plt.plot(x,y_prn[:x_range], color='green', label='prn')
        plt.plot(x,y_3ddfa[:x_range], color='yellow', label='3ddfa')
        plt.legend(loc= 'lower right' )
        plt.xlabel("NME normalized by bounding box size")
        plt.ylabel("Number of images (%)")
        plt.title("Alignment Accuracy on AFLW2000 Dataset(68 points)")


        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        
        img = Image.open(buf)
        img_tensor = transforms.ToTensor()(img)        
        plt.close()
        self.log('alfw.error', mean_rst.item())
        accept = torch.sum((shape_para.norm(dim=1)>600).int()* (shape_para.norm(dim=1)<800).int())

        self.logger.experiment.add_scalar("val_shape_para.mean", shape_para.norm(dim=1).mean().item(), self.global_step)
        self.logger.experiment.add_scalar("accpet rate", accept.item(), self.global_step)
        self.logger.experiment.add_image("Validation Image", img_tensor, self.global_step)
        ##########################################################################################################################
        
        print("Evaluating LFW...")
        output_lfw = outputs[1]
        lfw_issame = torch.stack([x['label'] for x in output_lfw]).reshape(-1)
        ip1 = torch.stack([x['ip1'] for x in output_lfw]).reshape(-1,199)
        ip2 = torch.stack([x['ip2'] for x in output_lfw]).reshape(-1,199)
        lfw_acc, lfw_thd = evaluate_fr(ip1,ip2,lfw_issame,metrics='Eu')
        self.logger.experiment.add_scalar("lfw.accuarcy", lfw_acc, self.global_step)
        # print(lfw_acc)
        ##########################################################################################################################
        print("Evaluating CFP-FP...")
        output_cfp = outputs[2]
        cfp_fp_issame = torch.stack([x['label'] for x in output_cfp]).reshape(-1)
        ip1 = torch.stack([x['ip1'] for x in output_cfp]).reshape(-1,199)
        ip2 = torch.stack([x['ip2'] for x in output_cfp]).reshape(-1,199)
        cfp_acc, cfp_thd =  evaluate_fr(ip1,ip2,cfp_fp_issame,metrics='Eu')
        self.logger.experiment.add_scalar("cfp_fp.accuarcy", cfp_acc, self.global_step)
        # print(cfp_acc)
        ########################################################################################################################

        print("Evaluating Age_DB...")
        output_age = outputs[3]
        agedb_30_issame = torch.stack([x['label'] for x in output_age]).reshape(-1)
        ip1 = torch.stack([x['ip1'] for x in output_age]).reshape(-1,199)
        ip2 = torch.stack([x['ip2'] for x in output_age]).reshape(-1,199)
        agedb_acc, agedb_thd  = evaluate_fr(ip1,ip2,agedb_30_issame,metrics='Eu')
        self.logger.experiment.add_scalar("agedb_30.accuarcy",agedb_acc, self.global_step)



        output_micc = outputs[4]
        micc_obj = torch.stack([x['shape'] for x in output_micc])
        micc_obj = micc_obj.reshape(micc_obj.shape[0]*micc_obj.shape[1],-1,3)
        identity = torch.stack([x['target'] for x in output_micc]).reshape(-1)
        error =  get_rmse_from_param(self.hp,micc_obj.data.cpu().numpy(),identity.data.cpu().numpy(),self.bfm.tris.data.cpu().numpy())
        self.log("val_loss",error)
        
        # print(agedb_acc)
        




    def configure_optimizers(self):
        lr = self.hp.model.learning_rate
        optimizer4nn = torch.optim.Adam([
            {'params':filter(lambda p: p.requires_grad, self.model.parameters())},
            {'params':filter(lambda p: p.requires_grad, self.bfm.parameters()), 'lr': lr*0.5},
            #{'params':filter(lambda p: p.requires_grad, self.confnet.parameters())},
            {'params':filter(lambda p: p.requires_grad, self.loss.fr_loss.parameters())},
            {'params':filter(lambda p: p.requires_grad, self.loss.center_loss.parameters()), 'lr': 1}],
            lr=lr, weight_decay=5e-4)
        scheduler = {
         'scheduler': StepLR(optimizer4nn, step_size=2, gamma=0.1),
         'monitor': 'metric_to_track',
         'interval': 'epoch',
         'frequency': 1,
         'strict': True,
        }
        return [optimizer4nn], [scheduler]

    def train_dataloader(self):
        normallize_mean = [0.485, 0.456, 0.406]
        normallize_std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(
            mean = normallize_mean,
            std = normallize_std)
        transform_train = transforms.Compose([
            transforms.CenterCrop((224,224)),    
            transforms.ToTensor(),
            normalize
        ])
        #dataset = DataSet_300WLP(self.hp.data.train_300WLP_dir, transform=transform_train)
        #dataset = DataSet_300WLP(self.hp.data.dataset_list, transform=transform_train)
        dataset = VGG2MixDataset(self.hp, transform=transform_train)
        return DataLoader(dataset, batch_size=self.hp.model.batch_size, num_workers=self.hp.model.num_workers, shuffle=True)

    def val_dataloader(self):
        normallize_mean = [0.485, 0.456, 0.406]
        normallize_std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(
            mean = normallize_mean,
            std = normallize_std)
        transform_test = transforms.Compose([
            transforms.CenterCrop((224,224)),    
            transforms.ToTensor(),
            normalize
        ])
        
        dataset = AFLW2000DataSet(self.hp.data.valset_dir, transform=transform_test)
        dataset_lfw = LFWDataSet(self.hp.data.valset_lfw_dir, transform=transform_test)
        dataset_cfp = CFPDataSet(self.hp.data.valset_cfp_dir, transform=transform_test)
        dataset_age = AgeDBDataSet(self.hp.data.valset_age_dir, transform=transform_test)
        dataset_micc = MICCDataSet(self.hp.data.valset_micc_dir, transform=transform_test)
        dataloader = DataLoader(dataset, batch_size=20,num_workers=self.hp.model.num_workers, shuffle=False)
        dataloader_lfw = DataLoader(dataset_lfw, batch_size=20,num_workers=self.hp.model.num_workers, shuffle=False)
        dataloader_cfp = DataLoader(dataset_cfp, batch_size=10,num_workers=self.hp.model.num_workers, shuffle=False)
        dataloader_age = DataLoader(dataset_age, batch_size=20,num_workers=self.hp.model.num_workers, shuffle=False)
        dataloader_micc = DataLoader(dataset_micc, batch_size=1,num_workers=self.hp.model.num_workers, shuffle=False)
        # loaders = {'alfw': dataloader,
        #            'lfw': dataloader_lfw,
        #            'cfp': dataloader_cfp,
        #            'age': dataloader_age,                      
        #            }
        return [dataloader,dataloader_lfw,dataloader_cfp,dataloader_age,dataloader_micc]


    def test_dataloader(self):
        normallize_mean = [0.485, 0.456, 0.406]
        normallize_std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(
            mean = normallize_mean,
            std = normallize_std)
        transform_test = transforms.Compose([
            transforms.CenterCrop((224,224)),    
            transforms.ToTensor(),
            normalize
        ])
        dataset = AFLW2000DataSet(self.hp.data.valset_dir, transform=transform_test)
        print(self.hp.data.test_dir)
        vis_dataset = VISDataSet(self.hp.data.test_dir, transform=transform_test)
        dataloader = DataLoader(dataset, batch_size=20,num_workers=self.hp.model.num_workers, shuffle=False)
        vis_dataloader = DataLoader(vis_dataset, batch_size=1,num_workers=self.hp.model.num_workers, shuffle=False)
        dataset_lfw = LFWDataSet(self.hp.data.valset_lfw_dir, transform=transform_test)
        dataset_cfp = CFPDataSet(self.hp.data.valset_cfp_dir, transform=transform_test)
        dataset_age = AgeDBDataSet(self.hp.data.valset_age_dir, transform=transform_test)
        dataset_micc = MICCDataSet(self.hp.data.valset_micc_dir, transform=transform_test)
        dataset_ytf = YTFDataset(self.hp.data.valset_ytf_dir, transform=transform_test)
        dataset_BosphorusDB = BosregDataSet(self.hp.data.valset_bos_dir, transform=transform_test)
        dataloader_lfw = DataLoader(dataset_lfw, batch_size=20,num_workers=self.hp.model.num_workers, shuffle=False)
        dataloader_cfp = DataLoader(dataset_cfp, batch_size=10,num_workers=self.hp.model.num_workers, shuffle=False)
        dataloader_age = DataLoader(dataset_age, batch_size=20,num_workers=self.hp.model.num_workers, shuffle=False)
        dataloader_micc = DataLoader(dataset_micc, batch_size=1,num_workers=self.hp.model.num_workers, shuffle=False)
        dataloader_ytf = DataLoader(dataset_ytf, batch_size=20,num_workers=self.hp.model.num_workers, shuffle=False)
        dataloader_BosphorusDB = DataLoader(dataset_BosphorusDB, batch_size=1,num_workers=self.hp.model.num_workers, shuffle=False)
        return [vis_dataloader,dataloader,dataloader_micc,dataloader_lfw,dataloader_cfp,dataloader_age,dataloader_ytf,dataloader_BosphorusDB]


    def test_step(self, batch, batch_idx,dataset_idx):
        

        if dataset_idx == 0:
            #return
            data, raw_image,imgname = batch
            parameters,recon_face, texture,confmap = self(data)
            parameters['shape_para'] = parameters['shape_para']/1.5
            ver = self.bfm.get_exp_obj(parameters['shape_para'],parameters['exp_para'])
            #shape_p = parameters['shape_para']*0
            
            # shape_p[:,batch_idx] = self.bfm.ev_shape[batch_idx]*5
            ver = self.bfm.get_shape(parameters['shape_para'])
            print(parameters['shape_para']/self.bfm.ev_shape)
           
            # print(self.bfm.e_shape.shape)
            # exit(0)
            # cv2.imwrite(os.path.join(self.hp.log.test_vis_dir,f'{batch_idx*recon_face.shape[0]+idx}.png'),raw_image[idx].permute(1,2,0).data.cpu().numpy()[...,::-1]*255)
            # self.bfm.writeobj(os.path.join(self.hp.log.test_vis_dir,f'{batch_idx*recon_face.shape[0]+idx}.obj'),vertices.reshape(-1,3))
           
            for idx,vertices in enumerate(ver):
                if parameters['shape_para'][idx].norm(dim=-1)<10000 and parameters['shape_para'][idx].norm(dim=-1)>600:
                    # pass
                    #print(raw_image[idx].shape)
                    # exit()
                    #print(parameters['shape_para'][idx])
                    #print(os.path.basename(imgname[idx])[:-4]+'.png')
                    cv2.imwrite(os.path.join(self.hp.log.test_vis_dir,os.path.basename(imgname[idx])[:-4]+'.png'),raw_image[idx].permute(1,2,0).data.cpu().numpy()[...,::-1]*255)
                    self.bfm.writeobj_s(os.path.join(self.hp.log.test_vis_dir,os.path.basename(imgname[idx])[:-4]+'.obj'),vertices.reshape(-1,3))
            #exit()
            print(self.bfm.ev_shape)
        elif dataset_idx == 1:
            return
            data, target , raw_image, imgname = batch
            parameters,recon_face, texture,confmap = self(data)
            pred_lms = self.bfm.get_landmark_68(face_pixels=recon_face).transpose(2,1)
            x_max, x_min, y_max, y_min = target[...,0].max(1)[0], target[...,0].min(1)[0], target[...,1].max(1)[0], target[...,1].min(1)[0]
            d_normalize = torch.sqrt((x_max - x_min) * (y_max - y_min))
            pts =  (pred_lms - target).norm(dim=-1).mean(1)
            rst = pts / d_normalize.float()
            ver = self.bfm.get_shape(parameters['shape_para']).data.cpu().numpy().reshape(parameters['shape_para'].shape[0],-1,3)
            # if batch_idx == 0:
            illum = parameters['illum']*0
            illum[:,:12] = illum[:,:12] + 1
            
            texture = self.bfm.tex_mu.expand(recon_face.shape[0],self.bfm.tex_mu.shape[0])*0+0.3
            result = self.Renderer(recon_face,texture,illum,shape=self.bfm.get_shape(parameters['shape_para']))
            #print(torch.unique(result['images'].permute(0,3,1,2)[:,3,:,:]))
            mix_image = torch.where(result['images'].permute(0,3,1,2)[:,3,:,:].unsqueeze(1)!=0,
                        result['images'].permute(0,3,1,2)[:,:3,:,:]*0.65 + raw_image*0.35,
                        raw_image).clamp(min=0,max=1)
            #result = self.Renderer(self.bfm.mu_shape.reshape(-1,3).T.unsqueeze(0),self.bfm.tex_mu.unsqueeze(0),illum)
            #render_image = torchvision.utils.make_grid(result['images'].permute(0,3,1,2)[:,:3,:,:], nrow=4)
            # render_image = torchvision.utils.make_grid(raw_image, nrow=4)
            # mix_image = torchvision.utils.make_grid(mix_image, nrow=4)
            # self.logger.experiment.add_image("Rendered Image", render_image, self.global_step)
            # self.logger.experiment.add_image("Mixed Image", mix_image, self.global_step)
            for idx,vertices in enumerate(ver):
                if parameters['shape_para'][idx].norm(dim=-1)<500 and parameters['shape_para'][idx].norm(dim=-1)>0:
                    print(vertices.shape)
                    #cv2.imwrite(os.path.join(self.hp.log.test_vis_dir,os.path.basename(imgname[idx][:-4]+'_ours.png')),mix_image[idx].permute(1,2,0).data.cpu().numpy()[...,::-1]*255)
                    self.bfm.writeobj_s(os.path.join(self.hp.log.test_vis_dir,os.path.basename(imgname[idx][:-4]+'_ours.obj')),vertices)
            #        pass
            return {'rsts': rst.cpu() , 'shape_para': parameters['shape_para'].cpu()}
        elif dataset_idx == 2:
            return
            data, target , raw_image = batch
            parameters,recon_face, texture,confmap = self(data)
            ver = self.bfm.get_shape(parameters['shape_para'])
            return {'shape': ver.cpu(), 'target': target.cpu()}
        elif  dataset_idx == 6:
            return
            data, imagename = batch
            parameters,_ , _, _ = self(data)
            #print(parameters['shape_para'])
            return {'ip':parameters['shape_para'].cpu()*3,'imagename':imagename }
        elif dataset_idx == 7:
            data, target, imagename = batch
            parameters,_ , _, _ = self(data)
            return {'shape_para':parameters['shape_para'].cpu()/1.25,'target': target.cpu()}
        else:
            return
            data, data2, label = batch
            parameters1,_ , _, _ = self(data)
            parameters2,_ , _, _ = self(data2)
            obj1 = self.bfm.get_shape(parameters1['shape_para'])
            obj2 = self.bfm.get_shape(parameters2['shape_para'])
            return {'ip1':parameters1['shape_para'].cpu(),'ip2':parameters2['shape_para'].cpu(),'label':label.cpu(),'obj_ip1':obj1.cpu(),'obj_ip2':obj2.cpu()}
        
                    #print(raw_image[idx].shape)
                    # exit()
                #print(parameters['shape_para'][idx])
                    #cv2.imwrite(os.path.join(self.hp.log.test_dir,f'{batch_idx*recon_face.shape[0]+idx}.png'),raw_image[idx].permute(1,2,0).data.cpu().numpy()[...,::-1]*255)
                    #self.bfm.writeobj(os.path.join(self.hp.log.test_dir,f'{batch_idx*recon_face.shape[0]+idx}.obj'),vertices.reshape(-1,3))
                
            # if batch_idx == 0:
            #     illum = parameters['illum']
            #     #tex = self.bfm.tex_mu.expand(recon_face.shape[0],self.bfm.tex_mu.shape[0])
            #     result = self.Renderer(recon_face,texture,illum,shape=self.bfm.get_shape(parameters['shape_para']))
            #     #print(torch.unique(result['images'].permute(0,3,1,2)[:,3,:,:]))
            #     # print(raw_image.shape,result['images'].shape)
            #     # exit()
            #     mix_image = torch.where(result['images'].permute(0,3,1,2)[:,3,:,:].unsqueeze(1)!=0,
            #                 result['images'].permute(0,3,1,2)[:,:3,:,:],
            #                 raw_image).clamp(min=0,max=1)
            #     #result = self.Renderer(self.bfm.mu_shape.reshape(-1,3).T.unsqueeze(0),self.bfm.tex_mu.unsqueeze(0),illum)
            #     render_image = torchvision.utils.make_grid(result['images'].permute(0,3,1,2)[:,:3,:,:], nrow=4)
            #     mix_image = torchvision.utils.make_grid(mix_image, nrow=4)
            #     self.logger.experiment.add_image("Rendered Image", render_image, self.global_step)
            #     self.logger.experiment.add_image("Mixed Image", mix_image, self.global_step)
            #     # print(confmap[0].shape)
            #     # self.logger.experiment.add_image("conf Image", confmap[0], self.global_step)

            


    def test_epoch_end(self, outputs):
        # print(outputs)
        # rsts = outputs['rsts']
        # shape_para = outputs['shape_para']
        #outputs = outputs[1]
     
        # w_sfm = self.bfm.w_shape_r.data.cpu().numpy() +  self.bfm.w_shape_t.data.cpu().numpy()
        # mu_sfm = self.bfm.mu_shape.data.cpu().numpy()
        # np.savez('result/sfm.npz',w_shape=w_sfm,mu_shape=mu_sfm)
        # exit()
        shape_para = torch.stack([x['shape_para'] for x in outputs[7]]).reshape(-1,199)
        labels = torch.stack([x['target'] for x in outputs[7]]).reshape(-1)
        tst = Test_SIR_term_Bos()
        sir_term = tst.calculate_sir_term(shape_para[:4666],labels[:4666],shape_para[4666:],self.bfm.ev_shape.data.cpu().numpy())

        print(sir_term)
        '''
        print("Evaluating aflw2000...")
        count = np.zeros(1000)
        rst = torch.stack([x['rsts'].reshape(-1) for x in outputs[1]]).reshape(-1)
        shape_para = torch.stack([x['shape_para'] for x in outputs[1]]).reshape(-1,199)
        print(shape_para.shape)
        for i in range(1000):
            count[i] = torch.sum(rst < i * 1.0 / 1000) *1.0 / rst.shape[0]
        mean_rst = torch.mean(rst)
        #import numpy as np
    
        prn = np.load(self.hp.val.prn_rst)
        _3ddfa = np.load(self.hp.val.ddfa_rst)
    

        x_range = 1000
        
        x = np.linspace(0, x_range / 1000., x_range)
        y = count * 100
        
        y_prn = prn['arr_0'] * 100
        y_3ddfa = _3ddfa['arr_0'] * 100
        
        plt.figure()
        plt.grid()
        plt.xlim(0,0.1)
        plt.ylim(0,100)
        plt.plot(x,y[:x_range], color='red', label='ours')
        plt.plot(x,y_prn[:x_range], color='green', label='prn')
        plt.plot(x,y_3ddfa[:x_range], color='yellow', label='3ddfa')
        plt.legend(loc= 'lower right' )
        plt.xlabel("NME normalized by bounding box size")
        plt.ylabel("Number of images (%)")
        plt.title("Alignment Accuracy on AFLW2000 Dataset(68 points)")


        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        
        img = Image.open(buf)
        img_tensor = transforms.ToTensor()(img)        
        plt.close()
        print(shape_para)
        print(shape_para.mean(dim=0),shape_para.std(dim=0))
        self.log('val_loss', mean_rst.item())
        self.log('shape_para', shape_para.norm(dim=1).mean().item())
        self.logger.experiment.add_image("Validation Image", img_tensor, self.global_step)
        
        
        


        
        print("Evaluating LFW...")
        output_lfw = outputs[3]
        lfw_issame = torch.stack([x['label'] for x in output_lfw]).reshape(-1)
        batch_size = lfw_issame.shape[0]
        ip1 = torch.stack([x['ip1'] for x in output_lfw]).reshape(-1,199)
        ip2 = torch.stack([x['ip2'] for x in output_lfw]).reshape(-1,199)
        lfw_acc, lfw_thd = evaluate_fr(ip1,ip2,lfw_issame,metrics='cos')
        
        
        self.log("lfw.accuarcy", lfw_acc)

        ip1 = torch.stack([x['obj_ip1'] for x in output_lfw]).reshape(-1,159447)
        ip2 = torch.stack([x['obj_ip2'] for x in output_lfw]).reshape(-1,159447)
        lfw_acc, lfw_thd = evaluate_fr(ip1,ip2,lfw_issame,metrics='cos')
        
        
        Eudistence = (ip1.reshape(batch_size,-1,3)-ip2.reshape(batch_size,-1,3))**2
        Eudistence = Eudistence.sum(dim=-1).sqrt().mean(-1)
        np.savez('result/lfw_obj.npz', ip1 = ip1.data.cpu().numpy(),ip2 = ip2.data.cpu().numpy())
        np.save('result/lfw.npy',Eudistence)

        self.log("lfw.obj.accuarcy", lfw_acc)
        # print(lfw_acc)
        ##########################################################################################################################
        print("Evaluating CFP-FP...")
        output_cfp = outputs[4]
        cfp_fp_issame = torch.stack([x['label'] for x in output_cfp]).reshape(-1)
        ip1 = torch.stack([x['ip1'] for x in output_cfp]).reshape(-1,199)
        ip2 = torch.stack([x['ip2'] for x in output_cfp]).reshape(-1,199)
        cfp_acc, cfp_thd =  evaluate_fr(ip1,ip2,cfp_fp_issame,metrics='cos')
        self.log("cfp_fp.accuarcy", cfp_acc)

        ip1 = torch.stack([x['obj_ip1'] for x in output_cfp]).reshape(-1,159447)
        ip2 = torch.stack([x['obj_ip2'] for x in output_cfp]).reshape(-1,159447)
        cfp_acc, cfp_thd =  evaluate_fr(ip1,ip2,cfp_fp_issame,metrics='cos')
        self.log("cfp_fp.obj.accuarcy", cfp_acc)
        # print(cfp_acc)
        ########################################################################################################################

        print("Evaluating Age_DB...")
        output_age = outputs[5]
        agedb_30_issame = torch.stack([x['label'] for x in output_age]).reshape(-1)
        ip1 = torch.stack([x['ip1'] for x in output_age]).reshape(-1,199)
        ip2 = torch.stack([x['ip2'] for x in output_age]).reshape(-1,199)
        agedb_acc, agedb_thd  = evaluate_fr(ip1,ip2,agedb_30_issame,metrics='cos')
        self.log("agedb_30.accuarcy",agedb_acc)


        ip1 = torch.stack([x['obj_ip1'] for x in output_age]).reshape(-1,159447)
        ip2 = torch.stack([x['obj_ip2'] for x in output_age]).reshape(-1,159447)
        agedb_acc, agedb_thd  = evaluate_fr(ip1,ip2,agedb_30_issame,metrics='cos')
        self.log("agedb_30.obj.accuarcy",agedb_acc)
        
        print("Evaluating YTF...")
        feat_dict = {}
        feat_count_dict = {}
        output_ytf = outputs[6]
        for i,x in enumerate(output_ytf):
            for idx,imgname in enumerate(x['imagename']):
                imgpath = imgname.split('/')[-3]+'/'+imgname.split('/')[-2]
                print(x['ip1'][idx])
                print(imgpath)
                if feat_dict.__contains__(imgpath):
                    feat_count_dict[imgpath] += 1
                    feat_dict[imgpath] += x['ip1'][idx]
                else:
                    feat_count_dict[imgpath] = 1 
                    feat_dict[imgpath] = x['ip1'][idx]
                print(feat_dict[imgpath])
            sys.stdout.write(str(i)+'/'+str(len(output_ytf))+'\r')


        for key in feat_dict.keys():
            feat_dict[key] = feat_dict[key] / feat_count_dict[key]
        print(feat_dict)
        import pickle  
        f = open("ytf_result_n1.pkl","wb")
        pickle.dump(feat_dict,f)
        f.close()
        '''

        # import pickle
        # feat_dict = pickle.load(open("ytf_result_n1.pkl","rb"))
        # with open('/home_old/jdq/PG/propressing/splits_corrected.txt') as f:
        #     split_lines = f.readlines()[1:]

        # ip1 = []
        # ip2 = []
        # labels = []
        # print(feat_dict)
        # for batch_idx,line in enumerate(split_lines):
        #     words = line.replace('\n', '').replace(' ', '').split(',')
        #     name1 =words[2]
        #     name2 =words[3]
        #     #print(name1,name2)
        #     sameflag = int(words[5])
        #     ip1.append(feat_dict[name1])
        #     ip2.append(feat_dict[name2])
        #     labels.append(int(sameflag))
        #     sys.stdout.write(str(batch_idx)+'/'+str(len(split_lines))+'\r')
        # ip1 = torch.stack(ip1)
        # ip2 = torch.stack(ip2)
        # #print(ip2.shape,ip1.shape)
        # labels = np.array(labels).flatten()   
        # ytf_acc, ytf_thd  = evaluate_fr(ip1,ip2,labels,metrics='cos')
        # self.log("ytf.accuarcy",ytf_acc)



        # output_micc = outputs[2]
        # micc_obj = torch.stack([x['shape'] for x in output_micc])
        # micc_obj = micc_obj.reshape(micc_obj.shape[0]*micc_obj.shape[1],-1,3)
        # identity = torch.stack([x['target'] for x in output_micc]).reshape(-1)
        # error =  get_rmse_from_param(self.hp,micc_obj.data.cpu().numpy(),identity.data.cpu().numpy(),self.bfm.tris.data.cpu().numpy())
        # self.log("micc.rmse",error)

        # self.logger.experiment.add_scalar("Validation Loss", mean_rst.item(), self.global_step)
        # self.logger.experiment.add_image("Validation Image", img_tensor, self.global_step)
    