import torch
import torchvision
import time
import matplotlib.pyplot as plt



def weight_init(m):
    """
    init model's prams, if a layer is a conv layer: set the weight to N(0, 0.02), set the bias to constant 0.
    if a layer is a batch norm: set the weight to N(1, 0.02), set the bias to constant 0.
    usage: model.apply(weight_init)

    params:
        m: model's layer
    return:
        None
    """
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    if class_name.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


def plot_img_tensor(img_tensor):
    """
    plot a batch of image tensor


    params:
        tensors: torch.tensor of size (N, img_channel, width, height)

    return:
        None
    """
    grid = torchvision.utils.make_grid(img_tensor, normalize=True)
    img_data = grid.permute(1, 2, 0).numpy()
    plt.imshow(img_data)
    plt.axis("off")



def save_checkpoint(dict, path="model.pth.tar"):
    """
    save the model and optim through a dictionary
     {"model": model.state_dict(), "optim": optim.state_dict()} (*)


    params:
        data: a dict of the structure (*) storing the model and optim

    return:
        None
    """
    now = time.strftime("%D_%H:%M")
    print(f"saving checkpoint at {now}, path is '{path}'")
    torch.save(dict, path)
    print("saving model... done!")

def load_checkpoint(path, model, optimizer):
    """
    loading the model and optim


    params:
        path: path to the file storing the data
        model: model waiting to load params
        optimizer: optim waiting to load params

    return:
        None
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optim"])
    print("loading model... done!")


