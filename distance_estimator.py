import torch
from torch.autograd import Variable
import math


PATH_TO_DE_MODEL = "./model/model.pth"

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

class distance_estimator():
    def __init__(self):
        self.de_model = linearRegression(3, 1)
        self.de_model.load_state_dict(torch.load(PATH_TO_DE_MODEL, map_location = 'cpu'))

    def mod_hw_add_diag(self, h,w):
        FRAME_WIDTH = 640
        FRAME_HEIGHT = 480
        FRAME_DIAGONAL = math.sqrt(FRAME_WIDTH ** 2 + FRAME_HEIGHT ** 2 )
        h1 = 1/(h/FRAME_HEIGHT)
        w1 = 1/(w/FRAME_WIDTH)
        diag = 1/(((h ** 2 + w ** 2) ** (1/2))/FRAME_DIAGONAL) # + dataset_x_train['w']** 2)
        print("diag = ", diag)
        return h1,w1,diag

    def estimator(self, h,w):
        h,w,diag = self.mod_hw_add_diag(h, w)
        print(h,w,diag)
        input_tensor = torch.tensor([[h, w, diag]])
        dist = self.de_model(Variable((input_tensor)[0].cpu()).float())[0].cpu().detach().numpy()
        return dist.tolist()