import torch.nn as nn
from blocks.attention import Attention
from blocks.lstm import BidirectionalLSTM
from blocks.resnet_feature_extract import ResNet_FeatureExtractor

"""
Fixed STR model: ResNet FeatureExtractor + BiLSTM + Attention
"""

class Model(nn.Module):
    def __init__(self,
                 input_channel=1,      # grayscale images
                 output_channel=512,   # feature extractor output channels
                 hidden_size=256,      # BiLSTM hidden units
                 num_class=64          # number of characters (charset size + 1 for EOS)
                 ):

        super(Model, self).__init__()

        """ Feature Extraction (ResNet) """
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # squeeze height

        """ Sequence Modeling (BiLSTM) """
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(output_channel, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )
        sequence_output = hidden_size

        """ Prediction (Attention) """
        self.Prediction = Attention(sequence_output, hidden_size, num_class)

    def forward(self, input, text=None, is_train=True, batch_max_length=25):
        """ Feature extraction """
        visual_feature = self.FeatureExtraction(input)        # [B, C, H, W]
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)            # [B, W, C]

        """ Sequence modeling """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction (Attention decoding) """
        prediction = self.Prediction(
            contextual_feature.contiguous(),
            text,
            is_train,
            batch_max_length=batch_max_length
        )

        return prediction
