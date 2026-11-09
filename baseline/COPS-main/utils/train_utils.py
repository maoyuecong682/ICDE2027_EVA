import torch
import numpy as np

# 自动选择设备：原版把 device='cuda' 写死成默认值，无 GPU 环境会直接崩
_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_pred_logit(testloader, net, device=_DEVICE ):
    ##### here we record the variance of each prediction 
    net.eval()
    logit_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            out = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            logit_list.append(out)
        logit_list=np.concatenate(logit_list, 0)
        print(logit_list.shape)
    return logit_list
def get_pred_raw(testloader, net, device=_DEVICE ):
    ##### here we record the variance of each prediction 
    net.eval()
    logit_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            out = outputs.detach().cpu().numpy()
            logit_list.append(out)
        logit_list=np.concatenate(logit_list, 0)
        print(logit_list.shape)
    return logit_list

def get_pred_un_sig(testloader, net, device=_DEVICE ):
    ##### here we record the variance of each prediction 
    net.eval()
    logit_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            out =  torch.sigmoid(outputs).detach().cpu().numpy()
            logit_list.append(out)
        logit_list=np.concatenate(logit_list, 0)
        print(logit_list.shape)
    return logit_list

def get_pred_un(testloader, net, device=_DEVICE ):
    ##### here we record the variance of each prediction 
    net.eval()
    logit_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            out =  torch.sigmoid(outputs).detach().cpu().numpy()
            logit_list.append(out)
        logit_list=np.concatenate(logit_list, 0)
        print(logit_list.shape)
    return logit_list
