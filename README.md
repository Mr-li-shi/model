# model
模型
功能：可视化模型的的输出。
例如需要输出resnet模型的可视化图
from pythonview import view
model = models.resnet18()
model.load_state_dict(torch.load(path, map_location=device))
view(model, torch.nn.Conv2d, 0, 8)#显示resnet18第一个Conv2d层输出的可视化，每行显示8个通道的图。
