from path_block import PathBlock

class Bottleneck(PathBlock):
    """
    1x1 Bottleneck
    """
    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        for i in range(2):
            x = self.de_conv(x)
            x = self.bn(x)
            x = self.relu(x)
            
        return x
