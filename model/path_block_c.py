from path_block import PathBlock

class PathBlockC(PathBlock):
    """
    Path Block C
    F_scale = 4
    Stride = 4
    DF = 4
    """
    def forward(self, x):
        x = self.dilated_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        for i in range(3):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
        x = self.avg_pool(x)

        for i in range(4):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

        x = self.de_conv(x)
        x = self.bn(x)
        x = self.relu(x)   
        return x
