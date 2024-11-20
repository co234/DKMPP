import torch
import torch.nn as nn

class DeepKernel(nn.Module):
    def __init__(self, input_dim, hidden_dim=10, output_dim=2):
        super(DeepKernel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)

        x2 = self.fc1(x2)
        x2 = self.relu(x2)
        x2 = self.fc2(x2)

        # Compute the similarity between the outputs
        similarity = torch.matmul(x1, x2.T)
        return self.sigmoid(similarity)




if __name__ == "__main__":
    # test:

    input_dim = 10
    x1 = torch.randn(4, input_dim)
    x2 = torch.randn(4, input_dim)
    # hidden_dim = 20
    # output_dim = 5

    deep_kernel = DeepKernel(input_dim)
    similarity_matrix = deep_kernel(x1, x2)
    print(similarity_matrix)
