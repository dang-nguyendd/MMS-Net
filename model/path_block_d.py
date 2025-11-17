from path_block import PathBlock

class PathBlockD(PathBlock):
    """
    Path Block D
    F_scale = 2
    Stride = 2
    DF = 2s
    """
    def forward(self, x):
        x = self.dilated_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        for i in range(2):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
        x = self.avg_pool(x)

        for i in range(2):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

        for i in range(2):
            x = self.de_conv(x)
            x = self.bn(x)
            x = self.relu(x)   
            
        return x
