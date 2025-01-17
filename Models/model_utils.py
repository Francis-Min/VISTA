import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.attention.blocks.encoder_layer import EncoderLayer


class TemporalFilter(nn.Module):
    def __init__(
            self,
            num_input=1,
            n_filters=1,
            filter_time_length=25,
            dropout_rate=0.2,
            device='cuda:0',
    ):
        super(TemporalFilter, self).__init__()

        kernel_size = (1, filter_time_length)
        ks0 = int(round((kernel_size[0] - 1) / 2))
        ks1 = int(round((kernel_size[1] - 1) / 2))
        kernel_padding = (ks0, ks1)

        self.temporal_filter = nn.Sequential(
            nn.Conv2d(in_channels=num_input, out_channels=n_filters, kernel_size=kernel_size, padding=kernel_padding),
            nn.BatchNorm2d(num_features=n_filters),
            nn.ELU(),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        x = self.temporal_filter(x)
        # x = x.squeeze()
        return x


class BrainDisentanglementMachine(nn.Module):
    def __init__(
            self,
            n_channel=63,
            time_length=250,
            hidden_dim=512,
            timePatch_size=10,
            att_layers=1,
            dropout_rate=0.2,
            device='cuda:0',
    ):
        super(BrainDisentanglementMachine, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = n_channel
        self.time_length = time_length
        self.timePatch_size = timePatch_size
        self.patch_num = time_length // timePatch_size

        self.time_disentangle = TimeDisentangle(n_channel=n_channel,
                                                time_length=time_length,
                                                hidden_dim=hidden_dim,
                                                timePatch_size=timePatch_size,
                                                device=device)

        self.feature_embedding = nn.Linear(time_length, hidden_dim)

        self.visualNorm = nn.LayerNorm(hidden_dim)
        self.visualLayers = nn.ModuleList([EncoderLayer(d_model=hidden_dim,
                                                        ffn_hidden=hidden_dim * 4,
                                                        n_head=8,
                                                        drop_prob=dropout_rate)
                                           for _ in range(att_layers)])

        self.semanticNorm = nn.LayerNorm(hidden_dim)
        self.semanticLayers = nn.ModuleList([EncoderLayer(d_model=hidden_dim,
                                                          ffn_hidden=hidden_dim * 4,
                                                          n_head=8,
                                                          drop_prob=dropout_rate)
                                             for _ in range(att_layers)])

        visualBrain = torch.empty(self.num_nodes, self.num_nodes, dtype=torch.float, requires_grad=True)
        semanticBrain = torch.empty(self.num_nodes, self.num_nodes, dtype=torch.float, requires_grad=True)
        # Apply Xavier normal initialization
        nn.init.xavier_normal_(visualBrain)
        nn.init.xavier_normal_(semanticBrain)
        # Move the tensor to the device and wrap it in nn.Parameter
        self.visualBrain = nn.Parameter(visualBrain.to(device))
        self.semanticBrain = nn.Parameter(semanticBrain.to(device))

        # GCN
        self.conv = Chebynet(hidden_dim, hidden_dim)
        self.conv_bn = nn.BatchNorm1d(num_features=hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)  # 设置适当的丢弃率

    def forward(self, x):
        x = x.squeeze()
        visual_restored, semantic_restored, visual_binary, semantic_binary, loss_time = self.time_disentangle(x)
        v_x = visual_restored
        s_x = semantic_restored
        # v_x = self.feature_embedding(visual_restored)
        # v_x = self.visualNorm(v_x)
        # s_x = self.feature_embedding(semantic_restored)
        # s_x = self.semanticNorm(s_x)
        # for layer in self.visualLayers:
        #     v_x = layer(v_x)
        # for layer in self.semanticLayers:
        #     s_x = layer(s_x)

        #
        visualBrain = self.normalize_A(self.visualBrain)
        semanticBrain = self.normalize_A(self.semanticBrain)
        v_x = self.conv_bn(v_x.transpose(1, 2)).transpose(1, 2)  # BatchNorm
        s_x = self.conv_bn(s_x.transpose(1, 2)).transpose(1, 2)  # BatchNorm
        v_x = self.conv(v_x, visualBrain)
        v_x = self.dropout(v_x)
        s_x = self.conv(s_x, semanticBrain)
        s_x = self.dropout(s_x)
        visualBrain = 0
        semanticBrain = 0
        # return v_x, s_x, [visual_binary, semantic_binary], [visual_binary, semantic_binary], loss_time
        return v_x, s_x, [visual_binary, semantic_binary], [visualBrain, semanticBrain], loss_time

    def normalize_A(self, A, symmetry=False):
        A = F.relu(A)
        if symmetry:
            A = A + torch.transpose(A, 0, 1)  # A+ A的转置
            d = torch.sum(A, 1)  # 对A的第1维度求和
            d = 1 / torch.sqrt(d + 1e-10)  # d的-1/2次方
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
        else:
            d = torch.sum(A, 1)
            d = 1 / torch.sqrt(d + 1e-10)
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
        return A


class TimeDisentangle(nn.Module):
    def __init__(
        self,
        n_channel=63,
        time_length=250,
        hidden_dim=64,
        timePatch_size=10,
        device='cuda:0',
    ):
        super(TimeDisentangle, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = n_channel
        self.time_length = time_length
        self.patch_size = timePatch_size
        self.patch_num = time_length // self.patch_size
        self.device = device

        feature_size = n_channel * self.patch_size
        self.layer_norm = nn.LayerNorm(feature_size)
        self.batch_norm = nn.BatchNorm1d(feature_size)  # For patches
        self.batch_norm2 = nn.BatchNorm1d(self.patch_num)  # For patches
        self.visual_patch_selection = FeatureToAdjacencyMatrix(feature_size=feature_size, node_size=self.patch_num)
        self.semantic_patch_selection = FeatureToAdjacencyMatrix(feature_size=feature_size, node_size=self.patch_num)
        self.output_norm = nn.LayerNorm(time_length)  # For restored time-series

    def forward(self, x):
        batch_size, channels, time_length = x.size()  # [batch_size, n_channel, time_length]

        # patch
        patches = x.unfold(2, self.patch_size, self.patch_size)  # (batch_size, n_channel, patch_num, patch_size)
        patches = patches.permute(0, 2, 1, 3)  # (batch_size, patch_num, n_channel, patch_size)
        tmp_patches = patches.flatten(2)  # (batch_size, patch_num, feature_size)

        # Normalization for patches
        tmp_patches = self.layer_norm(tmp_patches)  # Normalize per feature
        tmp_patches = self.batch_norm(tmp_patches.transpose(1, 2)).transpose(1, 2)  # Normalize per patch

        # Visual & Semantic Patch
        visual_patch = self.visual_patch_selection(tmp_patches)
        semantic_patch = self.semantic_patch_selection(tmp_patches)

        # activate
        visual_binary = torch.sigmoid(self.batch_norm2(visual_patch))
        semantic_binary = torch.sigmoid(self.batch_norm2(semantic_patch))

        # time loss
        loss_time = self.compute_time_loss(visual_binary, semantic_binary)

        # avg of time patch
        replacement_value = patches.mean(dim=-1, keepdim=True)  # (batch_size, patch_num, 1)

        visual_features = torch.where(visual_binary.unsqueeze(-1) > 0.5, patches, replacement_value)
        semantic_features = torch.where(semantic_binary.unsqueeze(-1) > 0.5, patches, replacement_value)

        # [batch_size, n_channel, time_length]
        visual_features = visual_features.view(batch_size, self.patch_num, channels, self.patch_size).permute(0, 2, 1, 3)
        visual_restored = visual_features.contiguous().view(batch_size, channels, -1)[:, :, :time_length]
        visual_restored = self.output_norm(visual_restored)  # Normalize the restored features

        semantic_features = semantic_features.view(batch_size, self.patch_num, channels, self.patch_size).permute(0, 2, 1, 3)
        semantic_restored = semantic_features.contiguous().view(batch_size, channels, -1)[:, :, :time_length]
        semantic_restored = self.output_norm(semantic_restored)  # Normalize the restored features

        return visual_restored, semantic_restored, visual_binary, semantic_binary, loss_time

    def compute_time_loss(self, visual_binary, semantic_binary, target_coverage=0.8):
        """
        time Loss
        """
        loss_cross = (visual_binary * semantic_binary).sum()

        patch_num = visual_binary.size(1)
        target_active_count = target_coverage * patch_num
        visual_coverage_loss = (visual_binary.sum() - target_active_count).abs()
        semantic_coverage_loss = (semantic_binary.sum() - target_active_count).abs()

        coverage_loss = visual_coverage_loss + semantic_coverage_loss
        total_loss = (coverage_loss + loss_cross) / visual_binary.shape[0]
        # total_loss = loss_cross / visual_binary.shape[0]
        return total_loss


class FeatureToAdjacencyMatrix(nn.Module):
    def __init__(self, feature_size, node_size):
        super(FeatureToAdjacencyMatrix, self).__init__()
        self.input_size = feature_size
        self.node_size = node_size
        self.f_dim = 256

        self.embedding = nn.Sequential(
            nn.Linear(self.input_size, self.f_dim),
            nn.ReLU(inplace=True)
        )

        # query, key, value 的线性变换
        self.query = nn.Linear(self.f_dim, self.f_dim)
        self.key = nn.Linear(self.f_dim, self.f_dim)
        self.value = nn.Linear(self.f_dim, self.f_dim)
        self.pp = nn.Linear(self.node_size, 1)

    def forward(self, x):
        hidden_states = self.embedding(x)
        query_layer = self.query(hidden_states)  # [bs, seqlen, hid_size]
        key_layer = self.key(hidden_states)  # [bs, seqlen, hid_size]
        attention_matrix = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        adjacency_matrix = self.pp(attention_matrix)
        return adjacency_matrix


class Chebynet(nn.Module):
    def __init__(self, input_dim, output_dim, K=2):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc = nn.ModuleList()
        for i in range(K):
            self.gc.append(GraphConvolution(input_dim, output_dim))

    def forward(self, x, adj):
        device = x.device
        adj = self.generate_cheby_adj(adj, self.K, device)
        for i in range(len(self.gc)):
            if i == 0:
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])
        result = F.relu(result)
        return result

    def generate_cheby_adj(self, A, K, device):
        support = []
        for i in range(K):
            if i == 0:
                # support.append(torch.eye(A.shape[1]).cuda())
                temp = torch.eye(A.shape[1])
                temp = temp.to(device)
                support.append(temp.float())
            elif i == 1:
                support.append(A.float())
            else:
                temp = torch.matmul(support[-1], A)
                support.append(temp.float())
        return support


class GraphConvolution(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            bias=True,
            device='cuda:0',
    ):
        super(GraphConvolution, self).__init__()

        self.in_features = input_dim
        self.out_features = output_dim

        weight = torch.empty(input_dim, output_dim, dtype=torch.float)
        # Apply Xavier normal initialization
        nn.init.xavier_normal_(weight)
        # Move the tensor to the device and wrap it in nn.Parameter
        self.weight = nn.Parameter(weight.to(device))

        if bias:
            self.bias = nn.Parameter(nn.init.constant_(
                torch.rand(output_dim), 0.1)).float().to(device)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adjacency_matrix):
        adj = adjacency_matrix.clone()
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            out = output + self.bias
            return out
        else:
            return output
