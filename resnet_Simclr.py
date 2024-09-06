from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
import resnet_imgnet
# from nt_xentloss import NTXentLoss
def _getResnet(name):
    resnet = {
        "resnet18":resnet_imgnet.resnet18_imgnet(),
        "resnet50":resnet_imgnet.resnet50_imgnet(),
    }
    return resnet[name]


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False

class SimclrResnetv2(nn.Module):
    
    def __init__(self,backbone_name,project_dim,pretrained_backbone=False):
        super().__init__()
        # self.backbone = _getResnet(backbone_name)
        self.convnet = _getResnet(backbone_name)
        self.pjh_inputdim = self.convnet.fc.in_features
        self.clsnum = self.convnet.fc.out_features
        self.convnet.fc = nn.Identity()
        
        projection_layers = [
            ('fc1', nn.Linear(self.pjh_inputdim, self.pjh_inputdim, bias=False)),
            ('bn1', nn.BatchNorm1d(self.pjh_inputdim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.pjh_inputdim, project_dim, bias=False)),
            ('bn2', BatchNorm1dNoBias(project_dim)),
        ]

        self.projection = nn.Sequential(OrderedDict(projection_layers))
        
    # V2!!!
#for simclr&branches training,MLCKD
    def forward(self,xis,xjs):
        features1,output1 = self.convnet(xis)
        features2,output2 = self.convnet(xjs)
        projection1 = self.projection(output1)
        projection2 = self.projection(output2)
        return features1,features2,projection1,projection2    
    
# #for linear evaluation,fine tuning DP
#     def forward(self,xis):
#         _,output1 = self.convnet(xis)
#         # projection1 = self.projection_head(output1)
#         return output1
    

    
    def get_project_input_dim(self):
        return self.pjh_inputdim
    


    def build_simclrResnet_from_backbone(self,filepath):
        checkpoint=torch.load(filepath)
        checkpoint = checkpoint["state_dict"]   
        self.convnet.fc = nn.Linear(self.pjh_inputdim, self.clsnum)
        self.convnet.load_state_dict(checkpoint)
        self.convnet.fc = nn.Identity()

    def build_simclrResnet_from_backbone_validate(self,checkpoint):
        self.convnet.fc = nn.Linear(self.pjh_inputdim, self.clsnum)
        self.convnet.load_state_dict(checkpoint)
        self.convnet.fc = nn.Identity()

        
        
    def load_pretrain_withoutFC2(self,filepath):
        pretrained_dict=torch.load(filepath)
        pretrained_dict = pretrained_dict['net']
        backbone_dict=self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone_dict}
        backbone_dict.update(pretrained_dict)
        self.load_state_dict(backbone_dict)
        self.convnet.fc = nn.Identity()

    def load_pretrain_withoutFC1(self,filepath):
        pretrained_dict=torch.load(filepath)
        pretrained_dict = pretrained_dict['state_dict']

        backbone_dict=self.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone_dict}

        backbone_dict.update(pretrained_dict)
        self.load_state_dict(backbone_dict)
        self.convnet.fc = nn.Identity()


class SepConv(nn.Module):
    
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

   


class Branch_resnet256_halfV2(nn.Module):
    def __init__(self,backbone_name,project_dim):
        super().__init__()
        self.mainframe = SimclrResnetv2(backbone_name,project_dim)

        self.expansion = 4
        self.expansionhalf = 1
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(64 * self.expansion, 64 * self.expansionhalf, kernel_size=1, bias=False),
            SepConv(
                channel_in=64 * self.expansionhalf,
                channel_out=128 * self.expansionhalf
            ),
            SepConv(
                channel_in=128 * self.expansionhalf,
                channel_out=256 * self.expansionhalf
            ),
            SepConv(
                channel_in=256 * self.expansionhalf,
                channel_out=512 * self.expansionhalf
            ),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(128 * self.expansion, 128 * self.expansionhalf, kernel_size=1, bias=False),
            SepConv(
                channel_in=128 * self.expansionhalf,
                channel_out=256 * self.expansionhalf,
            ),
            SepConv(
                channel_in=256 * self.expansionhalf,
                channel_out=512 * self.expansionhalf,
            ),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(256 * self.expansion, 256 * self.expansionhalf, kernel_size=1, bias=False),
            SepConv(
                channel_in=256 * self.expansionhalf,
                channel_out=512 * self.expansionhalf,
            ),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(512 * self.expansion, 512 * self.expansionhalf, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self,xis,xjs):
        features1,features2,projection1,projection2 = self.mainframe(xis,xjs)
        # for i in range(4):
        #     print(features1[i].shape)
        
        auxiliary1 = []
        feat1 = features1[0]
        a1 = self.branch1(feat1)
        auxiliary1.append(torch.squeeze(a1))
        
        feat2 = features1[1]
        a1 = self.branch2(feat2)
        auxiliary1.append(torch.squeeze(a1))
        
        feat3 = features1[2]
        a1 = self.branch3(feat3)
        auxiliary1.append(torch.squeeze(a1))
        
        feat4 = features1[3]
        a1 = self.branch4(feat4)
        auxiliary1.append(torch.squeeze(a1))
        
        
        
        auxiliary2 = []
        feat1 = features2[0]
        a2 = self.branch1(feat1)
        auxiliary2.append(torch.squeeze(a2))
        
        feat2 = features2[1]
        a2 = self.branch2(feat2)
        auxiliary2.append(torch.squeeze(a2))
        
        feat3 = features2[2]
        a2 = self.branch3(feat3)
        auxiliary2.append(torch.squeeze(a2))
        
        feat4 = features2[3]
        a2 = self.branch4(feat4)
        auxiliary2.append(torch.squeeze(a2))
        
        return projection1,projection2,auxiliary1,auxiliary2
    
    def load_mainnetwork1(self,filepath):
        checkpoint = torch.load(filepath)
        # self.mainframe.convnet.fc = nn.Linear(self.mainframe.pjh_inputdim, self.mainframe.clsnum)
        self.mainframe.load_state_dict(checkpoint["net"])
        # self.mainframe.convnet.fc = nn.Identity()        


    def load_mainnetwork2(self,filepath):
        checkpoint = torch.load(filepath)
        self.mainframe.convnet.fc = nn.Linear(self.mainframe.pjh_inputdim, self.mainframe.clsnum)
        self.mainframe.load_state_dict(checkpoint["state_dict"])
        self.mainframe.convnet.fc = nn.Identity()    

    def load_mainframeV2(self,filepath):
        pretrained_dict=torch.load(filepath)
        pretrained_dict = pretrained_dict['state_dict']
        backbone_dict=self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone_dict}
        backbone_dict.update(pretrained_dict)
        self.load_state_dict(backbone_dict)
        self.convnet.fc = nn.Identity()

    def load_completenetwork(self,filepath):
        checkpoint = torch.load(filepath)
        # print(checkpoint['net'])
        # print(checkpoint)
        self.load_state_dict(checkpoint['net']) 

class Branch_resnet64V2(nn.Module):
    def __init__(self,backbone_name,project_dim):
        super().__init__()
        self.mainframe = SimclrResnetv2(backbone_name,project_dim)
        self.mainframe.projection = nn.Identity()########
        self.expansion = 4
        self.expansionhalf = 1
        
        self.branch1 = nn.Sequential(
            # nn.Conv2d(64 * self.expansion, 64 * self.expansionhalf, kernel_size=1, bias=False),
            SepConv(
                channel_in=64 * self.expansionhalf,
                channel_out=128 * self.expansionhalf
            ),
            SepConv(
                channel_in=128 * self.expansionhalf,
                channel_out=256 * self.expansionhalf
            ),
            SepConv(
                channel_in=256 * self.expansionhalf,
                channel_out=512 * self.expansionhalf
            ),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.branch2 = nn.Sequential(
            # nn.Conv2d(128 * self.expansion, 128 * self.expansionhalf, kernel_size=1, bias=False),
            SepConv(
                channel_in=128 * self.expansionhalf,
                channel_out=256 * self.expansionhalf,
            ),
            SepConv(
                channel_in=256 * self.expansionhalf,
                channel_out=512 * self.expansionhalf,
            ),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.branch3 = nn.Sequential(
            # nn.Conv2d(256 * self.expansion, 256 * self.expansionhalf, kernel_size=1, bias=False),
            SepConv(
                channel_in=256 * self.expansionhalf,
                channel_out=512 * self.expansionhalf,
            ),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.branch4 = nn.Sequential(
            # nn.Conv2d(512 * self.expansion, 512 * self.expansionhalf, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self,xis,xjs):
        features1,features2,_,_ = self.mainframe(xis,xjs)
        # for i in range(4):
        #     print(features1[i].shape)
        
        auxiliary1 = []
        feat1 = features1[0]
        a1 = self.branch1(feat1)
        auxiliary1.append(torch.squeeze(a1))
        
        feat2 = features1[1]
        a1 = self.branch2(feat2)
        auxiliary1.append(torch.squeeze(a1))
        
        feat3 = features1[2]
        a1 = self.branch3(feat3)
        auxiliary1.append(torch.squeeze(a1))
        
        feat4 = features1[3]
        a1 = self.branch4(feat4)
        auxiliary1.append(torch.squeeze(a1))
        
        
        
        auxiliary2 = []
        feat1 = features2[0]
        a2 = self.branch1(feat1)
        auxiliary2.append(torch.squeeze(a2))
        
        feat2 = features2[1]
        a2 = self.branch2(feat2)
        auxiliary2.append(torch.squeeze(a2))
        
        feat3 = features2[2]
        a2 = self.branch3(feat3)
        auxiliary2.append(torch.squeeze(a2))
        
        feat4 = features2[3]
        a2 = self.branch4(feat4)
        auxiliary2.append(torch.squeeze(a2))
        
        return _,_,auxiliary1,auxiliary2




if __name__ == "__main__":
    teacher_model = Branch_resnet256_halfV2(args.modelname_student,args.projection)
    file = './checkpoint/branches_train_4branches_resnet50-112-epoch-99.pth'
    teacher_model.load_completenetwork(file)