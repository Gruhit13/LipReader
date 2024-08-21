import torch as T
import torchvision as tv
from torchvision.transforms import v2
from model import VisualEfficientConformer
from config import VECConfig
from torchsummary import summary

config = VECConfig

model = VisualEfficientConformer(config)
video_filepath = "./datasets/s1/bgah1s.mpg"
video, _, _ = tv.io.read_video(video_filepath, pts_unit="sec", output_format="TCHW")
video = video.to(T.float)
video = v2.Grayscale()(video)
video = v2.Resize(size=(224, 224))(video)

# [B, T, C, H, W] => [B, C, T, H, W]
input_video = video.unsqueeze(0).transpose(1, 2)

# oup = model(input_video, lengths=10)
# print("Output shape:", oup.shape)

model_summary = summary(model, input_data=input_video, verbose=0)

with open("model_summary2.txt", "w") as summary_writer:
    summary_writer.write(model_summary.__str__())