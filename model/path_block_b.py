from path_block import PathBlock

class PathBlockB(PathBlock):
    """
    Path Block B
    F_scale = 1
    Stride = 1
    DF = 1
    """
    def forward(self, x):
        for i in range(4):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
        x = self.avg_pool(x)

        for i in range(3):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
        x = self.avg_pool(x)

        for i in range(3):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)

        # x = self.dilated_conv(x)
        # x = self.pool(x)
        # x = self.de_conv(x)
        return x
